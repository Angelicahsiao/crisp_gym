#!/usr/bin/env python3
"""Suggest every episode's query points from one episode you already clicked.

Optional step before click_track_points.py. You click one reference episode
yourself; this script carries those clicks into every episode of a dataset by
running RoboTracker on a two-image clip: [the reference frame you clicked on,
the episode's frame --frame]. Where each point lands is the suggestion. The
click tool then opens each episode with the suggestions already placed
(--suggestions): Enter accepts them, "redo" re-clicks one.

The reference may come from another dataset of the same task (for example the
UMI handheld clicks for a robot dataset): the points are the same physical
spots. It works when the two views look alike — same object, similar distance
and angle. Nothing is saved without your Enter in the click tool, so a poor
suggestion costs a click, not a wrong label.

    python crisp_gym/scripts/prefill_track_points.py \\
        --root /workspace/datasets/franka_electricbox/lerobot \\
        --reference-root /workspace/datasets/umi_electricbox/lerobot \\
        --reference-queries /workspace/datasets/umi_electricbox/track_queries.json \\
        --reference-episode 0 \\
        --robotrack-repo /workspace/team/anglica/robotrack \\
        --out /workspace/datasets/franka_electricbox/track_suggestions.json

Needs what track_points.py needs: lerobot, a CUDA GPU and a robotrack clone.

OUTPUT (--out, JSON): the click tool's labels format, plus "kind":
"suggestions", the reference it came from, and per point the tracker's
visibility. A point whose visibility is below --min-visibility gets no
suggestion (null): the click tool leaves it for you.
"""

from __future__ import annotations

import argparse
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
from track_points import TRACK_SIZE, load_queries, load_tracker, scale_xy  # noqa: E402

logger = logging.getLogger("prefill_track_points")


def resize(frame: np.ndarray, hw) -> np.ndarray:
    import cv2

    if frame.shape[:2] == tuple(hw):
        return frame
    return cv2.resize(frame, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_LINEAR)


def reference_groups(labels: dict, points: list[str]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """The reference clicks grouped by the frame they were made on: {frame: [(point index, xy)]}."""
    groups: dict[int, list] = {}
    for i, name in enumerate(points):
        click = labels.get(name)
        if click is not None:
            groups.setdefault(int(click["frame"]), []).append(
                (i, np.array([click["x"], click["y"]], np.float32))
            )
    return groups


def suggest(
    track_fn,
    ref_frames_trk: dict[int, np.ndarray],
    groups: dict,
    ref_hw,
    target_trk: np.ndarray,
    target_hw,
    frame: int,
    n_points: int,
    min_visibility: float,
) -> dict[int, dict | None]:
    """Suggestions for one episode: {point index: {frame, x, y, visibility} or None}.

    One two-image clip per reference frame: [reference frame, target frame].
    Positions come back in the target's camera pixels.
    """
    out: dict[int, dict | None] = {i: None for i in range(n_points)}
    for ref_frame, members in groups.items():
        idx = [i for i, _ in members]
        q = scale_xy(np.stack([xy for _, xy in members]), ref_hw, TRACK_SIZE)
        clip = np.stack([ref_frames_trk[ref_frame], target_trk])
        pos, vis = track_fn(clip, q, np.zeros(len(idx), np.int64))
        pos = np.asarray(pos, np.float32)[1]
        vis = np.asarray(vis, np.float32)[1]
        xy = scale_xy(pos, TRACK_SIZE, target_hw)
        H, W = target_hw
        for j, i in enumerate(idx):
            x, y = float(xy[j, 0]), float(xy[j, 1])
            inside = 0 <= x <= W - 1 and 0 <= y <= H - 1
            if vis[j] >= min_visibility and inside:
                out[i] = {
                    "frame": int(frame),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "visibility": round(float(vis[j]), 3),
                }
    return out


def run(
    ref_source,
    ref_episode: int,
    ref_labels: dict,
    target_source,
    points: list[str],
    track_fn,
    *,
    camera: str,
    frame: int = 0,
    episodes=None,
    min_visibility: float = 0.6,
    reference: dict | None = None,
) -> dict:
    """Suggestions for every requested episode of the target, in the suggestions format."""
    groups = reference_groups(ref_labels, points)
    if not groups:
        raise SystemExit("the reference episode has no clicked point (all marked not visible)")
    ref_hw = None
    ref_frames_trk = {}
    for f in groups:
        img = to_hwc_uint8(ref_source.frame(int(ref_episode), f))
        ref_hw = img.shape[:2]
        ref_frames_trk[f] = resize(img, TRACK_SIZE)
    frames = target_source.episode_frames()
    wanted = sorted(frames) if episodes is None else sorted(int(e) for e in episodes)
    missing = [e for e in wanted if e not in frames]
    if missing:
        raise SystemExit(f"episode(s) {missing} are not in the dataset")
    target_hw = None
    out_eps: dict[str, dict] = {}
    placed = np.zeros(len(points), int)
    t0 = time.time()
    for n, ep in enumerate(wanted, 1):
        if frame >= frames[ep]:
            logger.warning("episode %d has only %d frames: no suggestion", ep, frames[ep])
            continue
        img = to_hwc_uint8(target_source.frame(ep, frame))
        if target_hw is None:
            target_hw = img.shape[:2]
        elif img.shape[:2] != target_hw:
            raise SystemExit(
                f"episode {ep} decodes to {img.shape[1]}x{img.shape[0]}, the first to "
                f"{target_hw[1]}x{target_hw[0]}"
            )
        result = suggest(
            track_fn,
            ref_frames_trk,
            groups,
            ref_hw,
            resize(img, TRACK_SIZE),
            target_hw,
            frame,
            len(points),
            min_visibility,
        )
        out_eps[str(ep)] = {points[i]: result[i] for i in range(len(points))}
        placed += np.array([result[i] is not None for i in range(len(points))])
        if n % 10 == 0 or n == len(wanted):
            logger.info("[%d/%d] %.0f s", n, len(wanted), time.time() - t0)
    logger.info(
        "suggested, per point: %s (of %d episodes)",
        ", ".join(f"{p} {c}" for p, c in zip(points, placed)),
        len(out_eps),
    )
    return {
        "version": 1,
        "kind": "suggestions",
        "camera": camera,
        "image_size": list(target_hw) if target_hw else None,
        "points": list(points),
        "reference": reference or {},
        "min_visibility": float(min_visibility),
        "episodes": out_eps,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Suggest query points for every episode from one clicked episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for the workflow.",
    )
    p.add_argument("--root", required=True, help="dataset to suggest points for")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--camera", default=None, help="its image key (default: the reference's camera)")
    p.add_argument("--reference-root", required=True, help="dataset the reference episode is in")
    p.add_argument("--reference-repo-id", default=None)
    p.add_argument(
        "--reference-queries", required=True, help="labels JSON with the reference episode's clicks"
    )
    p.add_argument(
        "--reference-episode",
        type=int,
        default=None,
        help="which clicked episode to use (default: the first one in the labels file)",
    )
    p.add_argument(
        "--out", required=True, help="suggestions JSON, for click_track_points.py --suggestions"
    )
    p.add_argument(
        "--frame", type=int, default=0, help="frame of each episode to place the suggestions on"
    )
    p.add_argument(
        "--min-visibility",
        type=float,
        default=0.6,
        help="leave a point unsuggested below this tracker visibility",
    )
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="only these episodes")
    p.add_argument("--robotrack-repo", default=None, help="path of a robotrack clone")
    p.add_argument("--checkpoint", default="robotracker")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ref_q = load_queries(args.reference_queries)
    labelled = sorted(int(e) for e in ref_q["episodes"])
    if not labelled:
        raise SystemExit(f"{args.reference_queries} has no clicked episode")
    ref_ep = labelled[0] if args.reference_episode is None else int(args.reference_episode)
    if str(ref_ep) not in ref_q["episodes"]:
        raise SystemExit(
            f"episode {ref_ep} has no clicks in {args.reference_queries} (clicked: {labelled[:20]})"
        )
    camera = args.camera or ref_q["camera"]
    track_fn = load_tracker(args.robotrack_repo, args.checkpoint)
    ref_source = LerobotFrameSource(
        args.reference_root, ref_q["camera"], repo_id=args.reference_repo_id
    )
    target = LerobotFrameSource(args.root, camera, repo_id=args.repo_id)
    reference = {
        "root": str(Path(args.reference_root).resolve()),
        "camera": ref_q["camera"],
        "queries": str(Path(args.reference_queries).resolve()),
        "episode": ref_ep,
    }
    logger.info(
        "reference: episode %d of %s; points %s", ref_ep, args.reference_root, ref_q["points"]
    )
    data = run(
        ref_source,
        ref_ep,
        ref_q["episodes"][str(ref_ep)],
        target,
        list(ref_q["points"]),
        track_fn,
        camera=camera,
        frame=args.frame,
        episodes=args.episodes,
        min_visibility=args.min_visibility,
        reference=reference,
    )
    write_json(Path(args.out), data)
    logger.info("wrote %s. Next: click_track_points.py ... --suggestions %s", args.out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
