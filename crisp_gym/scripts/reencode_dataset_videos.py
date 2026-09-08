"""Re-encode a LeRobot dataset's videos so datasets with different codecs can be merged.

Use case: episodes recorded before the codec became configurable carry
``video.codec: "av1"``; episodes recorded after it carry ``"h264"``. LeRobot
refuses to aggregate the two, in two separate places:

1. ``aggregate.validate_all_metadata`` -> ``features_equal_for_merge`` compares
   ``features[<key>]["info"]`` with only the encoder-TUNING keys removed
   (``VIDEO_ENCODER_INFO_KEYS`` = g / crf / preset / fast_decode /
   extra_options / video_backend). ``video.codec`` and ``video.pix_fmt`` are
   NOT in that set, so they are compared and a mismatch raises
   "Same features is expected, but got features=...".
2. Even with matching metadata, ``concatenate_video_files(...,
   compatibility_check=True)`` raises on differing codec / pix_fmt / width /
   height / fps, because it CONCATENATES BY STREAM COPY — no re-encode. That is
   why fixing only info.json produces a corrupt merge rather than an error.

So both the mp4 streams and info.json have to change. This script does both,
then verifies the result.

WHY VERIFICATION MATTERS. In the v3.0 layout a single mp4 holds MANY episodes,
and ``meta/episodes/*.parquet`` locates each one by
``videos/<key>/from_timestamp`` and ``.../to_timestamp`` — offsets in seconds
into that shared file. If a re-encode changes the frame count or shifts the
timeline by even one frame, every episode after the shift silently reads the
wrong images while the dataset still loads cleanly. This script therefore
re-checks, per file: frame count identical, duration within one frame period,
and every episode's stored offsets still landing on the same frame index.

By default it writes a NEW dataset directory and leaves the original alone.

Usage:
    # 1. See what you have and whether it would merge
    python -m crisp_gym.scripts.reencode_dataset_videos --inspect \
        --repo-id data_buffer/old_av1 --repo-id data_buffer/new_h264

    # 2. Convert the odd one out to match the other (recommended: --match
    #    copies codec AND pix_fmt from the reference, so they cannot disagree)
    python -m crisp_gym.scripts.reencode_dataset_videos \
        --repo-id data_buffer/old_av1 \
        --match data_buffer/new_h264 \
        --output-repo-id data_buffer/old_av1_as_h264

    # 3. Or state the target explicitly
    python -m crisp_gym.scripts.reencode_dataset_videos \
        --repo-id data_buffer/old_av1 --output-repo-id data_buffer/old_av1_h264 \
        --vcodec h264_nvenc --crf 21

    # 4. Then merge with lerobot
    python -c "from lerobot.datasets.aggregate import aggregate_datasets; \
        aggregate_datasets(['data_buffer/old_av1_as_h264','data_buffer/new_h264'], \
                           'data_buffer/merged')"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import av
from lerobot.configs.video import VIDEO_CODECS_ALIASES, RGBEncoderConfig
from lerobot.datasets.video_utils import get_video_info, reencode_video
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# NVENC requires gop_size > b_frames + 1 and its presets enable B-frames, so
# lerobot's near-all-intra g=2 cannot open ("Gop Length should be greater than
# number of B frames + 1") unless B-frames are off. Same rule as
# RecordingManager._rgb_encoder — keep the two in step.
NVENC_BF0_MAX_GOP = 4

# Stream-derived keys that must match for lerobot to merge two datasets. These
# are exactly the ones concatenate_video_files' compatibility_check compares.
COMPAT_KEYS = ("video.codec", "video.pix_fmt", "video.width", "video.height", "video.fps")


@dataclass
class VideoProbe:
    """What a single mp4 actually contains, read back from the file."""

    path: Path
    codec: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float

    def timeline_key(self) -> tuple:
        """The part that must survive a re-encode untouched."""
        return (self.width, self.height, self.fps, self.frames)


def resolve_root(repo_id: str, root: Path | None) -> Path:
    """Return the dataset directory for a repo id, or the explicit root."""
    path = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if not (path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"No LeRobot dataset at {path} (meta/info.json missing).")
    return path


def load_info(root: Path) -> dict:
    """Read meta/info.json."""
    return json.loads((root / "meta" / "info.json").read_text())


def video_keys(info: dict) -> list[str]:
    """Feature keys stored as video."""
    return [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


def video_files(root: Path, key: str) -> list[Path]:
    """Every mp4 backing one video key, in chunk/file order.

    Globbed rather than built from info["video_path"] so a dataset written with
    a non-default template still works; the sort keeps chunk-000/file-001
    before chunk-001/file-000.
    """
    return sorted((root / "videos" / key).glob("chunk-*/file-*.mp4"))


def probe(path: Path) -> VideoProbe:
    """Read a file's stream parameters and count its frames.

    Frames are counted by DEMUXING (one packet per frame for h264/hevc/av1 in
    mp4) rather than decoding — a full decode of every file would dominate the
    runtime, and the sampled-frame check below is what catches content drift.
    """
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        codec = stream.codec.canonical_name
        pix_fmt = stream.pix_fmt
        width, height = int(stream.width), int(stream.height)
        fps = float(stream.base_rate)
        frames = sum(1 for packet in container.demux(stream) if packet.size)
    duration_s = frames / fps if fps else 0.0
    return VideoProbe(path, codec, pix_fmt, width, height, fps, frames, duration_s)


def build_encoder(
    vcodec: str,
    pix_fmt: str | None,
    crf: int | None,
    gop: int | None,
    preset: str | int | None,
    extra_options: dict | None,
) -> RGBEncoderConfig:
    """Build the target encoder config, applying the NVENC B-frame rule.

    ``vcodec`` accepts lerobot's stored names too ("av1" -> "libsvtav1"), so a
    value copied straight out of an info.json works.
    """
    vcodec = VIDEO_CODECS_ALIASES.get(vcodec, vcodec)
    options = dict(extra_options or {})

    kwargs: dict = {"vcodec": vcodec}
    if pix_fmt is not None:
        kwargs["pix_fmt"] = pix_fmt
    if crf is not None:
        kwargs["crf"] = crf
    if gop is not None:
        kwargs["g"] = gop
    if preset is not None:
        kwargs["preset"] = preset

    # Construct once to learn the effective gop after defaults are applied,
    # then re-construct with bf=0 folded in if NVENC needs it.
    encoder = RGBEncoderConfig(**kwargs, extra_options=options)
    if (
        encoder.vcodec.endswith("_nvenc")
        and encoder.g is not None
        and encoder.g <= NVENC_BF0_MAX_GOP
        and "bf" not in options
    ):
        options["bf"] = 0
        logger.info(
            "%s with g=%s: forcing bf=0 (NVENC needs the GOP to exceed its B-frame count).",
            encoder.vcodec,
            encoder.g,
        )
        encoder = RGBEncoderConfig(**kwargs, extra_options=options)
    return encoder


def preflight(encoder: RGBEncoderConfig, sizes: set[tuple[int, int]], fps: int = 30) -> None:
    """Open the encoder at every size present, before touching a single file.

    Without this a rejected option combination fails on the first re-encode,
    after the output tree has already been created. Mirrors
    ``RecordingManager._preflight_encoder`` including the VERBOSE retry:
    hardware encoders fail with a bare AVERROR_UNKNOWN and the actual reason
    ("Gop Length should be greater than number of B frames + 1",
    "Cannot load libnvidia-encode") only surfaces at that log level.
    """
    from fractions import Fraction

    options = {k: str(v) for k, v in encoder.get_codec_options().items()}

    def _open(width: int, height: int) -> None:
        ctx = av.CodecContext.create(encoder.vcodec, "w")
        ctx.width, ctx.height = int(width), int(height)
        ctx.pix_fmt = encoder.pix_fmt
        ctx.time_base = Fraction(1, int(fps))
        ctx.options = dict(options)
        ctx.open()
        del ctx  # release the encoder session immediately

    for width, height in sorted(sizes):
        try:
            _open(width, height)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            detail = ""
            previous = None
            try:
                previous = av.logging.get_level()
                av.logging.set_level(av.logging.VERBOSE)
                _open(width, height)
            except Exception as verbose_exc:  # noqa: BLE001 - diagnostic only
                detail = f"\nFFmpeg detail: {verbose_exc}"
            finally:
                if previous is not None:
                    av.logging.set_level(previous)
            raise RuntimeError(
                f"Encoder preflight FAILED: {encoder.vcodec} cannot open for "
                f"{width}x{height} with {options} ({exc}).{detail}\n"
                "Nothing was written."
            ) from exc
    logger.info(
        "Encoder preflight OK: %s opens for %s with %s.",
        encoder.vcodec,
        sorted(sizes),
        options,
    )


def episode_offsets(root: Path, key: str) -> list[tuple[int, int, int, float, float]]:
    """Per-episode (episode_index, chunk_index, file_index, from_ts, to_ts).

    Empty when the episode metadata does not carry video offsets (older
    layouts, or a dataset with one file per episode).
    """
    import pandas as pd

    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        return []
    rows: list[tuple[int, int, int, float, float]] = []
    for path in files:
        frame = pd.read_parquet(path)
        cols = (
            f"videos/{key}/chunk_index",
            f"videos/{key}/file_index",
            f"videos/{key}/from_timestamp",
            f"videos/{key}/to_timestamp",
        )
        if not all(col in frame.columns for col in cols):
            return []
        for _, row in frame.iterrows():
            rows.append(
                (
                    int(row["episode_index"]),
                    int(row[cols[0]]),
                    int(row[cols[1]]),
                    float(row[cols[2]]),
                    float(row[cols[3]]),
                )
            )
    return rows


def verify_timeline(before: VideoProbe, after: VideoProbe) -> list[str]:
    """Return the reasons a re-encoded file is NOT a drop-in replacement."""
    problems: list[str] = []
    if before.frames != after.frames:
        problems.append(f"frame count {before.frames} -> {after.frames}")
    if (before.width, before.height) != (after.width, after.height):
        problems.append(
            f"size {before.width}x{before.height} -> {after.width}x{after.height}"
        )
    if abs(before.fps - after.fps) > 1e-6:
        problems.append(f"fps {before.fps} -> {after.fps}")
    period = 1.0 / before.fps if before.fps else 0.0
    if abs(before.duration_s - after.duration_s) > period:
        problems.append(
            f"duration {before.duration_s:.4f}s -> {after.duration_s:.4f}s "
            f"(more than one frame period, {period:.4f}s)"
        )
    return problems


def verify_episode_offsets(
    offsets: list[tuple[int, int, int, float, float]],
    key: str,
    chunk_index: int,
    file_index: int,
    after: VideoProbe,
) -> list[str]:
    """Check every episode's stored offsets still resolve inside the new file.

    LeRobot reads an episode as ``from_timestamp + t``; if the new file is
    shorter, or a boundary now maps to a different frame index, the episode
    reads someone else's frames.
    """
    problems: list[str] = []
    for episode_index, chunk, file, from_ts, to_ts in offsets:
        if (chunk, file) != (chunk_index, file_index):
            continue
        half_period = 0.5 / after.fps if after.fps else 0.0
        if to_ts > after.duration_s + half_period:
            problems.append(
                f"episode {episode_index} of {key} ends at {to_ts:.4f}s but the "
                f"re-encoded file is only {after.duration_s:.4f}s long"
            )
        from_frame = round(from_ts * after.fps)
        if from_frame >= after.frames:
            problems.append(
                f"episode {episode_index} of {key} starts at frame {from_frame}, "
                f"past the file's {after.frames} frames"
            )
    return problems


def sample_frame_diff(old: Path, new: Path, count: int, total: int) -> float:
    """Mean absolute pixel difference over `count` frames spread across the file.

    Frame counting alone cannot see a timeline SHIFT — the same number of
    frames in a different order still counts right. This decodes matching frame
    indices from both files and compares them. Expect a small non-zero value
    (both codecs are lossy); a large one means the frames no longer line up.
    """
    import numpy as np

    def frames_at(path: Path, wanted: set[int]) -> dict[int, "np.ndarray"]:
        out: dict[int, np.ndarray] = {}
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    out[index] = frame.to_ndarray(format="rgb24").astype(np.int16)
                if len(out) == len(wanted):
                    break
        return out

    if total == 0 or count <= 0:
        return 0.0
    step = max(1, total // count)
    wanted = {i for i in range(0, total, step)}
    if len(wanted) > count:
        wanted = set(sorted(wanted)[:count])

    old_frames = frames_at(old, wanted)
    new_frames = frames_at(new, wanted)
    shared = sorted(set(old_frames) & set(new_frames))
    if not shared:
        return float("inf")
    return float(
        np.mean([np.abs(old_frames[i] - new_frames[i]).mean() for i in shared])
    )


def describe(root: Path, label: str) -> dict[str, dict]:
    """Log what a dataset's videos actually are, and return it per key."""
    info = load_info(root)
    keys = video_keys(info)
    logger.info("%s (%s)", label, root)
    if not keys:
        logger.info("  no video features")
        return {}

    summary: dict[str, dict] = {}
    for key in keys:
        files = video_files(root, key)
        # `.get("info", {})` is not enough: the key can be present and null.
        declared_info = info["features"][key].get("info") or {}
        declared = {k: declared_info.get(k) for k in COMPAT_KEYS}
        if not files:
            logger.info("  %s: declared %s — NO mp4 files found", key, declared)
            summary[key] = {"declared": declared, "actual": None, "files": 0}
            continue
        first = probe(files[0])
        actual = {
            "video.codec": first.codec,
            "video.pix_fmt": first.pix_fmt,
            "video.width": first.width,
            "video.height": first.height,
            "video.fps": int(first.fps),
        }
        logger.info(
            "  %s: %d file(s), stream %s %s %dx%d @%gfps",
            key,
            len(files),
            first.codec,
            first.pix_fmt,
            first.width,
            first.height,
            first.fps,
        )
        mismatched = {k: (declared[k], actual[k]) for k in COMPAT_KEYS if declared[k] != actual[k]}
        if mismatched:
            logger.warning(
                "    info.json disagrees with the file itself: %s "
                "(declared, actual) — the merge compares the DECLARED values.",
                mismatched,
            )
        summary[key] = {"declared": declared, "actual": actual, "files": len(files)}
    return summary


def report_mergeability(summaries: list[tuple[str, dict[str, dict]]]) -> None:
    """Say whether these datasets would aggregate as they stand, and why not."""
    if len(summaries) < 2:
        return
    logger.info("--- mergeability ---")
    (ref_label, ref) = summaries[0]
    blocked = False
    for label, other in summaries[1:]:
        if set(ref) != set(other):
            logger.error(
                "%s and %s declare different video keys (%s vs %s) — that is a "
                "schema difference this script does not fix.",
                ref_label,
                label,
                sorted(ref),
                sorted(other),
            )
            blocked = True
            continue
        for key in ref:
            diffs = {
                k: (ref[key]["declared"][k], other[key]["declared"][k])
                for k in COMPAT_KEYS
                if ref[key]["declared"][k] != other[key]["declared"][k]
            }
            if diffs:
                blocked = True
                logger.error("%s vs %s, %s: %s", ref_label, label, key, diffs)
    if blocked:
        logger.error(
            "Would NOT merge. Re-encode one side to match the other "
            "(--match <reference>)."
        )
    else:
        logger.info("Video parameters match — aggregate_datasets should accept these.")


def copy_without_videos(src: Path, dst: Path) -> None:
    """Copy a dataset tree except the top-level videos/ dir, which is rebuilt."""
    if dst.exists():
        raise FileExistsError(f"Output already exists: {dst}. Remove it or pick another name.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_resolved = src.resolve()

    def _skip_top_level_videos(directory: str, names: list[str]) -> set[str]:
        return {"videos"} if Path(directory).resolve() == src_resolved else set()

    shutil.copytree(src, dst, ignore=_skip_top_level_videos)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id", action="append", default=[], help="Dataset repo id (repeatable for --inspect)."
    )
    parser.add_argument(
        "--root", action="append", default=[], help="Explicit dataset root, positionally paired with --repo-id."
    )
    parser.add_argument("--inspect", action="store_true", help="Report codecs and exit. Changes nothing.")
    parser.add_argument("--match", help="Take the target codec AND pix_fmt from this dataset (repo id or path).")
    parser.add_argument("--vcodec", help="Target codec, e.g. h264_nvenc, h264, libsvtav1, av1, auto.")
    parser.add_argument("--pix-fmt", help="Target pixel format (default: the encoder's own).")
    parser.add_argument("--crf", type=int, help="Quality. CODEC-SPECIFIC: CRF for libsvtav1, constant QP for NVENC.")
    parser.add_argument("--gop", type=int, help="GOP size (lerobot default 2 = near all-intra, fast seeking).")
    parser.add_argument("--preset", help="Codec-specific preset. Do not set alongside --vcodec auto.")
    parser.add_argument("--extra-options", type=json.loads, help='Raw codec options as JSON, e.g. \'{"bf": 0}\'.')
    parser.add_argument("--output-repo-id", help="Write the converted dataset here (default: <repo-id>_<codec>).")
    parser.add_argument("--output-root", help="Explicit output directory, overrides --output-repo-id.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the source dataset's videos. Destructive — there is no undo.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; write nothing.")
    parser.add_argument(
        "--verify-frames",
        type=int,
        default=3,
        help="Decode this many frames per file from both copies and compare (0 disables).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    if not args.repo_id and not args.root:
        parser.error("give at least one --repo-id or --root")

    roots = [
        resolve_root(repo, Path(root) if root else None)
        for repo, root in zip(
            args.repo_id or [None] * len(args.root),
            args.root or [None] * len(args.repo_id),
            strict=False,
        )
    ]
    labels = [r or str(p) for r, p in zip(args.repo_id or [None] * len(roots), roots, strict=False)]

    if args.inspect:
        summaries = [(label, describe(root, label)) for label, root in zip(labels, roots, strict=False)]
        report_mergeability(summaries)
        return 0

    if len(roots) != 1:
        parser.error("converting takes exactly one --repo-id/--root (use --inspect to compare several)")
    src_root, src_label = roots[0], labels[0]

    # ── target encoder ────────────────────────────────────────────────────
    vcodec, pix_fmt = args.vcodec, args.pix_fmt
    if args.match:
        match_root = resolve_root(args.match, Path(args.match) if Path(args.match).exists() else None)
        match_info = load_info(match_root)
        match_keys = video_keys(match_info)
        if not match_keys:
            parser.error(f"--match dataset {match_root} has no video features")
        reference = match_info["features"][match_keys[0]].get("info", {})
        vcodec = vcodec or reference.get("video.codec")
        pix_fmt = pix_fmt or reference.get("video.pix_fmt")
        logger.info("Target from --match %s: codec=%s pix_fmt=%s", match_root, vcodec, pix_fmt)
    if not vcodec:
        parser.error("give --vcodec or --match")

    encoder = build_encoder(vcodec, pix_fmt, args.crf, args.gop, args.preset, args.extra_options)
    logger.info(
        "Target encoder: vcodec=%s pix_fmt=%s options=%s",
        encoder.vcodec,
        encoder.pix_fmt,
        encoder.get_codec_options(as_strings=True),
    )

    info = load_info(src_root)
    keys = video_keys(info)
    if not keys:
        logger.error("%s has no video features — nothing to do.", src_label)
        return 1

    depth_keys = [
        key for key in keys if (info["features"][key].get("info") or {}).get("is_depth_map")
    ]
    if depth_keys:
        logger.error(
            "%s has depth video key(s) %s. Depth is stored 12-bit lossless with "
            "quantisation parameters (depth_min/depth_max/shift/use_log) in its "
            "info block; re-encoding it with an RGB encoder would silently "
            "destroy the depth values. Refusing rather than corrupting them.",
            src_label,
            depth_keys,
        )
        return 1

    plan: list[tuple[str, Path]] = []
    sizes: set[tuple[int, int]] = set()
    for key in keys:
        files = video_files(src_root, key)
        if not files:
            logger.error("%s: no mp4 files under videos/%s", src_label, key)
            return 1
        for path in files:
            plan.append((key, path))
        first = probe(files[0])
        sizes.add((first.width, first.height))
    logger.info("%d file(s) across %d video key(s); sizes %s", len(plan), len(keys), sorted(sizes))

    if args.dry_run:
        logger.info("--dry-run: would re-encode %d file(s) to %s. Nothing written.", len(plan), encoder.vcodec)
        return 0

    preflight(encoder, sizes)

    # ── output tree ───────────────────────────────────────────────────────
    if args.in_place:
        dst_root = src_root
        logger.warning("--in-place: overwriting %s. There is no undo.", src_root)
    else:
        if args.output_root:
            dst_root = Path(args.output_root)
        elif args.output_repo_id:
            dst_root = HF_LEROBOT_HOME / args.output_repo_id
        elif args.repo_id:
            dst_root = HF_LEROBOT_HOME / f"{args.repo_id[0]}_{encoder.vcodec}"
        else:
            # Source was given as a bare --root, so there is no repo id to
            # derive a name from; sit the output beside it.
            dst_root = src_root.parent / f"{src_root.name}_{encoder.vcodec}"
        logger.info("Copying %s -> %s (videos rebuilt, not copied)", src_root, dst_root)
        copy_without_videos(src_root, dst_root)

    offsets_by_key = {key: episode_offsets(src_root, key) for key in keys}
    for key, rows in offsets_by_key.items():
        if not rows:
            logger.warning(
                "No per-episode video offsets found for %s — the offset check "
                "will be skipped for it.",
                key,
            )

    # ── re-encode ─────────────────────────────────────────────────────────
    failures: list[str] = []
    for number, (key, src_path) in enumerate(plan, start=1):
        relative = src_path.relative_to(src_root)
        dst_path = dst_root / relative
        before = probe(src_path)
        logger.info(
            "[%d/%d] %s: %s %s -> %s (%d frames)",
            number,
            len(plan),
            relative,
            before.codec,
            before.pix_fmt,
            encoder.vcodec,
            before.frames,
        )

        if args.in_place:
            staged = src_path.with_suffix(".reencoded.mp4")
            reencode_video(src_path, staged, video_encoder=encoder, overwrite=True)
            after = probe(staged)
        else:
            reencode_video(src_path, dst_path, video_encoder=encoder, overwrite=True)
            after = probe(dst_path)

        problems = verify_timeline(before, after)
        chunk_index = int(relative.parts[-2].split("-")[-1])
        file_index = int(relative.stem.split("-")[-1])
        problems += verify_episode_offsets(
            offsets_by_key.get(key, []), key, chunk_index, file_index, after
        )

        if not problems and args.verify_frames > 0:
            probe_path = staged if args.in_place else dst_path
            diff = sample_frame_diff(src_path, probe_path, args.verify_frames, before.frames)
            logger.info("      sampled frame diff: %.2f / 255 mean abs", diff)
            if diff > 25.0:
                problems.append(
                    f"sampled frames differ by {diff:.1f}/255 on average — the "
                    "timeline may have shifted, or quality collapsed"
                )

        if problems:
            failures.append(f"{relative}: " + "; ".join(problems))
            logger.error("      REJECTED: %s", "; ".join(problems))
            if args.in_place:
                staged.unlink(missing_ok=True)
            break

        if args.in_place:
            staged.replace(src_path)

    if failures:
        logger.error(
            "Stopped after %d failure(s). %s",
            len(failures),
            "The source is untouched."
            if not args.in_place
            else "Files already replaced in-place are NOT rolled back.",
        )
        for failure in failures:
            logger.error("  %s", failure)
        return 1

    # ── metadata ──────────────────────────────────────────────────────────
    dst_info = load_info(dst_root)
    for key in keys:
        first_new = video_files(dst_root, key)[0]
        probed = get_video_info(first_new, video_encoder=encoder)
        existing = dst_info["features"][key].get("info", {}) or {}
        if "is_depth_map" in existing:
            probed["is_depth_map"] = existing["is_depth_map"]
        dst_info["features"][key]["info"] = {**existing, **probed}
        logger.info(
            "info.json %s: codec=%s pix_fmt=%s",
            key,
            probed.get("video.codec"),
            probed.get("video.pix_fmt"),
        )
    (dst_root / "meta" / "info.json").write_text(json.dumps(dst_info, indent=4, ensure_ascii=False))

    logger.info("Done. %d file(s) re-encoded and verified. Dataset: %s", len(plan), dst_root)
    logger.info(
        "Confirm the merge is now accepted with:\n"
        "  python -m crisp_gym.scripts.reencode_dataset_videos --inspect "
        "--root %s --root <the other dataset>",
        dst_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
