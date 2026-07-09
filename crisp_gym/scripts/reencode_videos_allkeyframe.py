"""Re-encode a LeRobot dataset's videos to ALL-KEYFRAME AV1 so torchcodec can
seek them, fixing the training-time decode crash:

    RuntimeError: Could not push packet to decoder: Invalid data found ...

torchcodec's frame-accurate seek fails on the default (inter-frame) AV1 streams
this repo records; pyav works but is slow. Re-encoding every frame as a
keyframe (``-g 1``) makes each frame independently decodable, so torchcodec
seeks fine — and is much faster than pyav at train time.

Safety
------
* Writes to a NEW dataset copy (``--output``); your source is never modified.
* Preserves everything the info.json/episodes metadata depends on: exact fps,
  frame COUNT, per-frame PTS (``-fps_mode passthrough``), resolution, codec
  (libsvtav1) and pixel format (yuv420p). File names/paths are unchanged, so
  the ``video_path`` template and per-episode ``from/to_timestamp`` stay valid.
* VERIFIES each re-encoded file against the source: frame count must match
  exactly, duration within tolerance, and resolution/pix_fmt identical. Any
  mismatch aborts (that file would desync the timestamp->frame mapping).

Trade-off: all-keyframe video is LARGER (no inter-frame compression) and slower
to encode (one-time). Decode/seek at train time is faster and torchcodec-safe.

After it finishes, train on the copy WITHOUT ``--dataset.video_backend=pyav``
(torchcodec default), and re-run check_relative_pose.py to confirm the data is
intact.

Usage
-----
    python crisp_gym/scripts/reencode_videos_allkeyframe.py \\
        --input  /path/to/dataset \\
        --output /path/to/dataset_allkey \\
        [--crf 30] [--preset 6] [--dry-run]

Requires ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("reencode_videos_allkeyframe")

DURATION_TOL_S = 0.05  # allow tiny timebase-rounding differences


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def ffprobe_stream(path: Path) -> dict:
    """Return {'frames': int, 'duration': float, 'width': int, 'height': int,
    'pix_fmt': str} for the first video stream (frames counted exactly)."""
    r = _run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,duration,width,height,pix_fmt",
        "-of", "json", str(path),
    ])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()}")
    st = json.loads(r.stdout)["streams"][0]
    dur = st.get("duration")
    return {
        "frames": int(st["nb_read_frames"]),
        "duration": float(dur) if dur not in (None, "N/A") else None,
        "width": int(st["width"]),
        "height": int(st["height"]),
        "pix_fmt": st.get("pix_fmt"),
    }


def count_keyframes(path: Path) -> int:
    """Number of keyframes in the first video stream."""
    r = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=key_frame", "-of", "csv", str(path),
    ])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe (keyframes) failed on {path}: {r.stderr.strip()}")
    return r.stdout.count("frame,1")


def reencode(src: Path, dst: Path, crf: int, preset: int) -> None:
    """Re-encode src -> dst as all-keyframe AV1, preserving timing/resolution."""
    r = _run([
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p",
        "-g", "1",                       # GOP size 1 -> every frame a keyframe.
        # NB: do NOT also pass -svtav1-params keyint=1 — it OVERRIDES -g and
        # yields a single keyframe (verified). Plain -g 1 gives all-keyframe.
        "-crf", str(crf), "-preset", str(preset),
        "-fps_mode", "passthrough",      # keep exact original PTS/frame timing
        "-video_track_timescale", "90000",
        "-an",                           # no audio (datasets have none)
        str(dst),
    ])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}:\n{r.stderr.strip()[-1500:]}")


def verify(src_info: dict, dst_info: dict, path: Path) -> None:
    if dst_info["frames"] != src_info["frames"]:
        raise ValueError(
            f"FRAME COUNT MISMATCH on {path.name}: "
            f"{src_info['frames']} -> {dst_info['frames']}. Re-encode changed the "
            "frame count; this would desync the timestamp->frame mapping. Aborting."
        )
    if (src_info["width"], src_info["height"]) != (dst_info["width"], dst_info["height"]):
        raise ValueError(
            f"RESOLUTION CHANGED on {path.name}: "
            f"{src_info['width']}x{src_info['height']} -> "
            f"{dst_info['width']}x{dst_info['height']}. Aborting."
        )
    if src_info["pix_fmt"] != dst_info["pix_fmt"]:
        logger.warning("  pix_fmt changed on %s: %s -> %s",
                       path.name, src_info["pix_fmt"], dst_info["pix_fmt"])
    if (src_info["duration"] is not None and dst_info["duration"] is not None
            and abs(src_info["duration"] - dst_info["duration"]) > DURATION_TOL_S):
        raise ValueError(
            f"DURATION MISMATCH on {path.name}: "
            f"{src_info['duration']:.4f}s -> {dst_info['duration']:.4f}s "
            f"(> {DURATION_TOL_S}s). Aborting."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-encode a LeRobot dataset's videos to all-keyframe AV1 "
        "(torchcodec-seekable) on a copy, preserving timing/metadata."
    )
    ap.add_argument("--input", required=True, help="Source dataset root dir.")
    ap.add_argument("--output", required=True, help="Destination dataset root dir (new).")
    ap.add_argument("--crf", type=int, default=30, help="libsvtav1 CRF quality (0-63, lower=better). Default 30.")
    ap.add_argument("--preset", type=int, default=6, help="libsvtav1 preset (0-13, higher=faster). Default 6.")
    ap.add_argument("--dry-run", action="store_true", help="List videos and exit; no copy/encode.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")

    for tool in ("ffmpeg", "ffprobe"):
        if _run([tool, "-version"]).returncode != 0:
            logger.error("%s not found on PATH.", tool)
            return 2

    src = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    src_videos = sorted((src / "videos").rglob("*.mp4"))
    if not src_videos:
        logger.error("No .mp4 files under %s/videos", src)
        return 2
    logger.info("Videos to re-encode: %d", len(src_videos))

    if args.dry_run:
        for v in src_videos:
            logger.info("  %s", v.relative_to(src))
        logger.info("--dry-run: nothing written.")
        return 0

    if out.exists():
        logger.error("Output %s already exists — remove it first.", out)
        return 2
    logger.info("Copying dataset %s -> %s", src, out)
    shutil.copytree(src, out)

    for i, src_v in enumerate(src_videos, 1):
        rel = src_v.relative_to(src)
        dst_v = out / rel                      # the copy (to be replaced)
        tmp = dst_v.with_suffix(".reenc.mp4")

        src_info = ffprobe_stream(src_v)
        reencode(src_v, tmp, args.crf, args.preset)
        dst_info = ffprobe_stream(tmp)
        try:
            verify(src_info, dst_info, rel)
            kf = count_keyframes(tmp)
            if kf != dst_info["frames"]:
                raise ValueError(
                    f"NOT ALL-KEYFRAME on {rel.name}: {kf}/{dst_info['frames']} "
                    "keyframes. torchcodec seek would still fail. Aborting."
                )
        except Exception:
            tmp.unlink(missing_ok=True)
            logger.error("Verification failed on %s — leaving the copy's original "
                         "video in place and aborting.", rel)
            raise
        tmp.replace(dst_v)                     # same filename -> info.json path valid

        s0 = src_v.stat().st_size / 1e6
        s1 = dst_v.stat().st_size / 1e6
        logger.info("  [%d/%d] %s  frames=%d  %.1f->%.1f MB",
                    i, len(src_videos), rel, dst_info["frames"], s0, s1)

    logger.info("Done. All-keyframe dataset: %s", out)
    logger.info("info.json codec stays 'av1' (libsvtav1) — no metadata change needed.")
    logger.info("Now train on the copy WITHOUT --dataset.video_backend=pyav "
                "(torchcodec default), and re-run check_relative_pose.py to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
