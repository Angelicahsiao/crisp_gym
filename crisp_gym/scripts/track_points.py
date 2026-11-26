#!/usr/bin/env python3
"""Track the clicked query points through every episode with RoboTracker.

Step 2 of point-track labels. Step 1, click_track_points.py, recorded for each
episode where each named point is and on which frame. This script runs the
tracker from those clicks and writes, per episode, where every point is in
every frame and whether it is visible. It does not touch the dataset: the
tracks go to their own folder, to be checked (previews, summary) before
anything is built on them.

    python crisp_gym/scripts/track_points.py \\
        --root /workspace/datasets/umi_electricbox/lerobot \\
        --queries /workspace/datasets/umi_electricbox/track_queries.json \\
        --robotrack-repo /workspace/team/anglica/robotrack \\
        --out /workspace/datasets/umi_electricbox/tracks \\
        --preview 0 10 50

Needs, in the Python that has lerobot: a CUDA GPU (RoboTracker moves frames to
the GPU itself), a clone of robotrack (--robotrack-repo, no install needed;
it needs einops, which lerobot's training extras bring), and the weights,
downloaded from Hugging Face on first use (--checkpoint robotracker) or a
local .pth file.

HOW IT TRACKS (robotrack's own benchmark protocol)

    Frames are resized to 384x512, as in robotrack's evaluation. Points that
    share a query frame are tracked together, forward from that frame.
    Before a point's query frame it is not visible (you clicked the earliest
    frame where it can be seen) and its position is held at the click. A
    point is visible when visibility x confidence >= 0.6. A point marked
    "not visible" in the episode has no position (NaN) and is never visible.

OUTPUT (--out)

    episode_000012.npz, one per episode:
        tracks         float32 [T, N, 2]  x, y in pixels of the camera frame
                       (pixel centres at whole numbers, as clicked); NaN for
                       a point not visible in the episode
        visible        bool    [T, N]
        visibility     float32 [T, N]     the tracker's visibility x confidence
        query_frames   int32   [N]        the clicked frame; -1 = not visible
        query_xy       float32 [N, 2]     the click; NaN = not visible
        points, image_size, track_size, checkpoint, episode
      T = the episode's frames in dataset order, N = the points in the
      order of the queries file.
    summary.csv   one row per episode and point: query frame, share of the
                  frames from the query frame on where the point is visible,
                  and suspicious jumps (a visible point moving more than
                  --jump-fraction of the image width in one frame).
    preview_episode_000012.mp4 / .png   tracks drawn on the video (filled =
                  visible, hollow = not visible, with a short trail), and a
                  sheet of 8 frames — for the episodes given to --preview.

Episodes with an .npz already are skipped unless --overwrite, so an
interrupted run continues where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from click_track_points import LerobotFrameSource, to_hwc_uint8  # noqa: E402

logger = logging.getLogger("track_points")

TRACK_SIZE = (384, 512)  # robotrack eval_alltracker.IMAGE_SIZE (H, W)
VIS_THRESHOLD = 0.6  # robotrack evaluate.VIS_THRESHOLD
COLORS_BGR = [
    (48, 59, 255),
    (89, 199, 52),
    (255, 132, 10),
    (0, 204, 255),
    (222, 90, 191),
    (10, 159, 255),
    (255, 210, 100),
    (95, 55, 255),
]


# ── coordinates ──────────────────────────────────────────────────────────────


def scale_xy(xy: np.ndarray, from_hw, to_hw) -> np.ndarray:
    """Pixel coordinates in an image of size `from_hw` -> one of size `to_hw`.

    Pixel centres sit at whole numbers in both, so the first and last pixel
    centres map onto each other: x' = x (W' - 1) / (W - 1). This is how
    robotrack scales its benchmark labels to the tracking size and back.
    """
    xy = np.asarray(xy, dtype=np.float64)
    fy = (to_hw[0] - 1) / max(from_hw[0] - 1, 1)
    fx = (to_hw[1] - 1) / max(from_hw[1] - 1, 1)
    return (xy * np.array([fx, fy])).astype(np.float32)


def episode_queries(labels: dict, points: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One episode's clicks as (query_xy [N, 2] in camera pixels, query_frames [N]).

    A point marked not visible gets NaN and frame -1.
    """
    xy = np.full((len(points), 2), np.nan, np.float32)
    frames = np.full(len(points), -1, np.int32)
    for i, name in enumerate(points):
        click = labels.get(name)
        if click is not None:
            xy[i] = (click["x"], click["y"])
            frames[i] = int(click["frame"])
    return xy, frames


def track_episode(
    frames_trk: np.ndarray, query_xy: np.ndarray, query_frames: np.ndarray, image_hw, track_fn
) -> dict:
    """Run the tracker on one episode and express the result in camera pixels.

    Args:
        frames_trk: [T, h, w, 3] uint8, the episode resized to the tracking size.
        query_xy, query_frames: from episode_queries (camera pixels).
        image_hw: (H, W) of the camera frames.
        track_fn: robotrack's ``track(frames, query_xy, query_frames)`` with
            the model bound: positions [T, n, 2] in tracking pixels and the
            visibility probability [T, n], zero before each query frame.
    """
    T = len(frames_trk)
    N = len(query_frames)
    track_hw = frames_trk.shape[1:3]
    tracks = np.full((T, N, 2), np.nan, np.float32)
    vis = np.zeros((T, N), np.float32)
    live = np.flatnonzero(query_frames >= 0)
    if live.size:
        if int(query_frames[live].max()) >= T:
            raise ValueError(f"a query frame is past the end of the episode ({T} frames)")
        q_trk = scale_xy(query_xy[live], image_hw, track_hw)
        pos, prob = track_fn(frames_trk, q_trk, query_frames[live])
        pos, prob = np.asarray(pos, np.float32), np.asarray(prob, np.float32)
        tracks[:, live] = scale_xy(pos.reshape(-1, 2), track_hw, image_hw).reshape(T, live.size, 2)
        vis[:, live] = prob
        for i in live:
            # Before its query frame a point is not visible and stays at the click.
            q = int(query_frames[i])
            tracks[:q, i] = query_xy[i]
            vis[:q, i] = 0.0
    return {
        "tracks": tracks,
        "visibility": vis,
        "visible": vis >= VIS_THRESHOLD,
        "query_frames": query_frames.astype(np.int32),
        "query_xy": query_xy.astype(np.float32),
    }


def summarize(result: dict, points: list[str], image_hw, jump_fraction: float) -> list[dict]:
    """Per point: query frame, visible share from the query frame on, and jumps."""
    rows = []
    limit = jump_fraction * image_hw[1]
    for i, name in enumerate(points):
        q = int(result["query_frames"][i])
        if q < 0:
            rows.append({"point": name, "query_frame": -1, "visible_share": 0.0, "jumps": 0})
            continue
        vis = result["visible"][q:, i]
        xy = result["tracks"][q:, i]
        both = vis[1:] & vis[:-1]
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        rows.append(
            {
                "point": name,
                "query_frame": q,
                "visible_share": round(float(vis.mean()), 3),
                "jumps": int(np.sum(both & (steps > limit))),
            }
        )
    return rows


def save_episode(
    path: Path, result: dict, *, points, image_hw, track_hw, checkpoint: str, episode: int
) -> None:
    """Write one episode atomically (a crash never leaves a half-written file)."""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **result,
        points=np.asarray(points),
        image_size=np.asarray(image_hw),
        track_size=np.asarray(track_hw),
        checkpoint=np.asarray(str(checkpoint)),
        episode=np.asarray(int(episode)),
    )
    os.replace(tmp, path)


# ── previews ─────────────────────────────────────────────────────────────────


def draw(
    frame_bgr: np.ndarray, result: dict, t: int, points, scale: float, trail: int = 12
) -> np.ndarray:
    """Tracks at frame t on one (preview-size, BGR) frame."""
    import cv2

    out = frame_bgr.copy()
    r = max(4, int(round(out.shape[1] / 160)))
    for i, name in enumerate(points):
        if result["query_frames"][i] < 0:
            continue
        color = COLORS_BGR[i % len(COLORS_BGR)]
        lo = max(int(result["query_frames"][i]), t - trail)
        pts = [result["tracks"][k, i] * scale for k in range(lo, t + 1) if result["visible"][k, i]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(
                out, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), color, 2
            )
        x, y = (int(round(v)) for v in result["tracks"][t, i] * scale)
        visible = bool(result["visible"][t, i])
        cv2.circle(out, (x, y), r, color, -1 if visible else 2)
        cv2.putText(out, name, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, name, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(out, f"frame {t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(out, f"frame {t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def write_preview(
    out_dir: Path,
    episode: int,
    frames_prev: np.ndarray,
    result: dict,
    points,
    scale: float,
    fps: float,
) -> list[Path]:
    """An MP4 of the whole episode and a PNG sheet of 8 frames."""
    import cv2

    T, h, w = frames_prev.shape[:3]
    drawn = [draw(frames_prev[t][:, :, ::-1], result, t, points, scale) for t in range(T)]
    mp4 = out_dir / f"preview_episode_{episode:06d}.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in drawn:
        writer.write(f)
    writer.release()
    picks = np.linspace(0, T - 1, 8).round().astype(int)
    rows = [np.concatenate([drawn[t] for t in picks[k : k + 4]], axis=1) for k in (0, 4)]
    png = out_dir / f"preview_episode_{episode:06d}.png"
    cv2.imwrite(str(png), np.concatenate(rows, axis=0))
    return [mp4, png]


# ── frames of one episode ────────────────────────────────────────────────────


class EpisodeVideo:
    """All frames of one camera for one episode, decoded in chunks and resized.

    Uses lerobot's own per-episode video query (one decoder pass) when this
    lerobot version has the one it was written against, after checking on the
    first episode that it returns exactly what dataset[row] returns; otherwise
    decodes row by row through dataset[row], which is slower but the same
    thing the training loader does.
    """

    def __init__(self, source: LerobotFrameSource, chunk: int = 16):
        self.source, self.ds, self.camera = source, source.ds, source.camera
        self.chunk = int(chunk)
        hf = self.ds.hf_dataset
        try:
            ts = np.asarray(hf.data.column("timestamp").to_numpy(), np.float64)
        except Exception:  # noqa: BLE001
            ts = np.asarray(list(hf.with_format(None)["timestamp"]), np.float64)
        self.timestamps = ts.reshape(-1)
        reader = getattr(self.ds, "reader", None)
        self._query = getattr(reader, "_query_videos", None)
        self._checked = False

    def _rows(self, episode: int, rows) -> list[np.ndarray]:
        return [to_hwc_uint8(self.ds[int(r)][self.camera]) for r in rows]

    def _chunk_fast(self, episode: int, rows) -> list[np.ndarray]:
        ts = [float(self.timestamps[r]) for r in rows]
        frames = self._query({self.camera: ts}, int(episode))[self.camera]
        if frames.ndim == 3:
            frames = frames[None]
        return [to_hwc_uint8(f) for f in frames]

    def frames(self, episode: int, sizes: list[tuple[int, int]]) -> list[np.ndarray]:
        """The episode at each requested (H, W), as [T, H, W, 3] uint8 arrays."""
        import cv2

        lo, hi = self.source.ranges[int(episode)]
        rows = list(range(lo, hi))
        if self._query is not None and not self._checked:
            probe = [lo, (lo + hi - 1) // 2, hi - 1]
            try:
                fast = self._chunk_fast(episode, probe)
                slow = self._rows(episode, probe)
                same = all(
                    a.shape == b.shape and int(np.abs(a.astype(int) - b.astype(int)).max()) <= 1
                    for a, b in zip(fast, slow)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("lerobot's episode video query failed (%s)", exc)
                same = False
            if not same:
                logger.warning("decoding frame by frame through dataset[row] (slower)")
                self._query = None
            self._checked = True
        out = [[] for _ in sizes]
        for k in range(0, len(rows), self.chunk):
            part = rows[k : k + self.chunk]
            frames = self._chunk_fast(episode, part) if self._query else self._rows(episode, part)
            for f in frames:
                for j, (h, w) in enumerate(sizes):
                    out[j].append(
                        f
                        if f.shape[:2] == (h, w)
                        else cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                    )
        return [np.stack(o) for o in out]


# ── the run ──────────────────────────────────────────────────────────────────


def load_queries(path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1:
        raise SystemExit(
            f"{path}: unknown version {data.get('version')!r}; written by click_track_points.py?"
        )
    return data


def load_tracker(robotrack_repo: str | None, checkpoint: str):
    """robotrack's model and track(), with the model bound."""
    if robotrack_repo:
        sys.path.insert(0, str(Path(robotrack_repo).expanduser().resolve()))
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("RoboTracker needs a CUDA GPU (it moves frames to the GPU itself).")
    try:
        from robotrack.eval_alltracker import load_model, track
    except ImportError as exc:
        raise SystemExit(
            f"cannot import robotrack ({exc}). Pass --robotrack-repo with the path of "
            "your robotrack clone, or pip install it."
        ) from exc
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    model = load_model(checkpoint, device="cuda")
    return lambda frames, xy, qf: track(model, frames, xy, qf)


def run(
    video: EpisodeVideo,
    queries: dict,
    track_fn,
    out_dir: Path,
    *,
    episodes=None,
    preview=(),
    overwrite: bool = False,
    checkpoint: str = "",
    jump_fraction: float = 0.05,
    fps: float = 15.0,
    preview_width: int = 640,
) -> list[dict]:
    """Track every requested labelled episode; returns the summary rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    points = list(queries["points"])
    image_hw = tuple(int(v) for v in queries["image_size"])
    labelled = {int(e): v for e, v in queries["episodes"].items()}
    available = set(video.source.ranges)
    wanted = sorted(labelled) if episodes is None else sorted(int(e) for e in episodes)
    unlabelled = [e for e in wanted if e not in labelled]
    if unlabelled:
        logger.warning("no clicks for episode(s) %s: skipped", unlabelled)
    missing = [e for e in wanted if e in labelled and e not in available]
    if missing:
        logger.warning(
            "episode(s) %s are in the queries file but not in the dataset: skipped", missing
        )
    todo = [e for e in wanted if e in labelled and e in available]
    if todo:
        size = to_hwc_uint8(video.source.frame(todo[0], 0)).shape[:2]
        if tuple(size) != image_hw:
            raise SystemExit(
                f"the clicks were made on {image_hw[1]}x{image_hw[0]} frames but "
                f"{queries['camera']} decodes to {size[1]}x{size[0]}"
            )
    preview = {int(e) for e in preview}
    pw = int(preview_width)
    ph = int(round(pw * image_hw[0] / image_hw[1]))
    rows, t_start = [], time.time()
    for n, ep in enumerate(todo, 1):
        path = out_dir / f"episode_{ep:06d}.npz"
        if path.exists() and not overwrite and ep not in preview:
            logger.info("[%d/%d] episode %d: already tracked", n, len(todo), ep)
            with np.load(path) as old:
                result = {k: old[k] for k in ("tracks", "visible", "query_frames")}
            rows += [
                {"episode": ep, **r} for r in summarize(result, points, image_hw, jump_fraction)
            ]
            continue
        t0 = time.time()
        sizes = [TRACK_SIZE] + ([(ph, pw)] if ep in preview else [])
        decoded = video.frames(ep, sizes)
        xy, qf = episode_queries(labelled[ep], points)
        result = track_episode(decoded[0], xy, qf, image_hw, track_fn)
        save_episode(
            path,
            result,
            points=points,
            image_hw=image_hw,
            track_hw=TRACK_SIZE,
            checkpoint=checkpoint,
            episode=ep,
        )
        summary = summarize(result, points, image_hw, jump_fraction)
        rows += [{"episode": ep, **r} for r in summary]
        if ep in preview:
            made = write_preview(out_dir, ep, decoded[1], result, points, pw / image_hw[1], fps)
            logger.info("  preview: %s", ", ".join(p.name for p in made))
        shares = ", ".join(
            f"{r['point']} {r['visible_share']:.0%}"
            + (f" ({r['jumps']} jumps)" if r["jumps"] else "")
            for r in summary
        )
        logger.info(
            "[%d/%d] episode %d: %d frames in %.1f s | visible from the click on: %s",
            n,
            len(todo),
            ep,
            len(decoded[0]),
            time.time() - t0,
            shares,
        )
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "point", "query_frame", "visible_share", "jumps"]
        )
        writer.writeheader()
        writer.writerows(rows)
    flagged = sorted({r["episode"] for r in rows if r["jumps"]})
    logger.info("done: %d episode(s) in %.0f s -> %s", len(todo), time.time() - t_start, out_dir)
    if flagged:
        logger.warning("episodes with suspicious jumps (look at them with --preview): %s", flagged)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track clicked points through every episode with RoboTracker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for the output format.",
    )
    p.add_argument("--root", required=True, help="LeRobot dataset folder")
    p.add_argument("--repo-id", default=None, help="dataset repo id (default: local/<folder name>)")
    p.add_argument("--queries", required=True, help="the JSON written by click_track_points.py")
    p.add_argument("--out", required=True, help="folder for the tracks, summary and previews")
    p.add_argument(
        "--robotrack-repo",
        default=None,
        help="path of a robotrack clone (instead of installing it)",
    )
    p.add_argument(
        "--checkpoint",
        default="robotracker",
        help="robotracker (downloads the weights), alltracker, a URL or a local .pth",
    )
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="only these episodes")
    p.add_argument(
        "--preview", type=int, nargs="*", default=[], help="episodes to draw (MP4 + PNG)"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="track episodes that already have an .npz again"
    )
    p.add_argument(
        "--jump-fraction",
        type=float,
        default=0.05,
        help="flag a visible point moving more than this share of the image width in one frame",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    queries = load_queries(args.queries)
    logger.info(
        "%d labelled episode(s), points %s, camera %s",
        len(queries["episodes"]),
        queries["points"],
        queries["camera"],
    )
    track_fn = load_tracker(args.robotrack_repo, args.checkpoint)
    source = LerobotFrameSource(args.root, queries["camera"], repo_id=args.repo_id)
    fps = float(getattr(source.ds.meta, "fps", 15) or 15)
    run(
        EpisodeVideo(source),
        queries,
        track_fn,
        Path(args.out),
        episodes=args.episodes,
        preview=args.preview,
        overwrite=args.overwrite,
        checkpoint=args.checkpoint,
        jump_fraction=args.jump_fraction,
        fps=fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
