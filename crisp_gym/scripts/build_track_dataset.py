#!/usr/bin/env python3
"""Build a training copy of a dataset whose action also carries the point tracks.

Step 3 of point-track labels (1: click_track_points.py, 2: track_points.py).
The source dataset is not changed. The copy has the same videos and columns,
except that `action` becomes

    [pose 9,  x, y, visible  for each point,  gripper]

so a policy learns to predict where the points go alongside the actions. The
copy says so in meta/track_targets.json; lerobot_relative_pose.py reads it and
(for multi_task_dit) weights the track dims and leaves hidden positions out of
the loss. Deployment is unchanged: it reads the pose from dims 0-8 and the
gripper from the last dim, and ignores the rest.

    python crisp_gym/scripts/build_track_dataset.py \\
        --root /workspace/datasets/franka_electricbox/lerobot \\
        --tracks /workspace/datasets/franka_electricbox/tracks \\
        --queries /workspace/datasets/franka_electricbox/track_queries.json \\
        --out /workspace/datasets/franka_electricbox_tracks/lerobot

WHAT GOES INTO THE TRACK DIMS

    x, y      the point in the camera frame, as x / (W - 1) and y / (H - 1)
              (0 and 1 are the centres of the first and last pixel).
    visible   1 or 0.
    row       action row t holds the tracks of image row t + 1 (--row-shift),
              the same one-row shift as the action's pose; the last row of an
              episode repeats its last image's tracks.
    hidden    a hidden point keeps its last visible position (before it is
              first seen: its first visible position; never seen: the image
              centre), with visible = 0. The training loss ignores these
              positions; they are only there so the column has no gaps.

CLEANING (a point is set hidden where its track is not trustworthy)

    jumps     a visible point moving more than --jump-fraction of the image
              width in one frame: hidden for --jump-pad frames either side.
    rigidity  points on one rigid object (by default: names sharing the part
              before the last "_", e.g. button_a and button_b) keep a nearly
              constant distance from frame to frame. Where it departs from its
              running median by more than --rigid-rel (or --rigid-abs pixels,
              whichever is larger), both points are hidden at that frame.
    Both are reported per point. Frames the tracker already marked hidden,
    including frames whose image could not be decoded, stay hidden.

Needs lerobot (it writes the copy with lerobot's own parquet writer; written
against lerobot 0.6.1's dataset layout, v3.0).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_track_dataset")

STAMP = "meta/track_targets.json"
STAT_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


# ── cleaning ──────────────────────────────────────────────────────────────────


def hide_jumps(
    tracks: np.ndarray, visible: np.ndarray, image_w: int, fraction: float, pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hide each point around its one-frame jumps. Returns (visible, hidden_by_this [T, N])."""
    T, N = visible.shape
    out = visible.copy()
    for i in range(N):
        step = np.linalg.norm(np.diff(tracks[:, i], axis=0), axis=1)
        both = visible[1:, i] & visible[:-1, i]
        for k in np.flatnonzero(both & (step > fraction * image_w)):
            t = k + 1
            out[max(0, t - pad) : min(T, t + pad + 1), i] = False
    return out, visible & ~out


def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    return np.array([np.median(x[max(0, k - half) : k + half + 1]) for k in range(len(x))])


def hide_rigid_breaks(
    tracks: np.ndarray,
    visible: np.ndarray,
    groups: list[list[int]],
    rel: float,
    abs_px: float,
    window: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide pairs of same-object points where their distance breaks from its running median."""
    out = visible.copy()
    for group in groups:
        for a_pos, a in enumerate(group):
            for b in group[a_pos + 1 :]:
                both = np.flatnonzero(visible[:, a] & visible[:, b])
                if len(both) < 3:
                    continue
                d = np.linalg.norm(tracks[both, a] - tracks[both, b], axis=1)
                med = _running_median(d, window)
                bad = both[np.abs(d - med) > np.maximum(abs_px, rel * med)]
                out[bad, a] = False
                out[bad, b] = False
    return out, visible & ~out


def default_groups(points: list[str]) -> list[list[int]]:
    """Points sharing the name before the last "_" (button_a, button_b) form a group."""
    by_prefix: dict[str, list[int]] = {}
    for i, name in enumerate(points):
        by_prefix.setdefault(name.rsplit("_", 1)[0] if "_" in name else name, []).append(i)
    return [g for g in by_prefix.values() if len(g) >= 2]


# ── track dims of one episode ─────────────────────────────────────────────────


def fill_hidden(xy: np.ndarray, visible: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """[T, 2] positions with hidden frames held at the last visible one (see the docstring)."""
    out = np.array(xy, np.float64, copy=True)
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        out[:] = centre
        return out
    # the last visible frame at or before each frame; before the first, the first
    pos = np.clip(np.searchsorted(idx, np.arange(len(out)), side="right") - 1, 0, None)
    out[:] = xy[idx[pos]]
    return out


def episode_track_dims(
    tracks: np.ndarray, visible: np.ndarray, image_hw, row_shift: int
) -> np.ndarray:
    """[T, 3N] = (x, y, visible) per point, normalized, held while hidden, shifted by row_shift."""
    T, N = visible.shape
    H, W = image_hw
    cols = []
    for i in range(N):
        xy = fill_hidden(tracks[:, i], visible[:, i], ((W - 1) / 2, (H - 1) / 2))
        x = np.clip(xy[:, 0] / (W - 1), 0.0, 1.0)
        y = np.clip(xy[:, 1] / (H - 1), 0.0, 1.0)
        cols += [x, y, visible[:, i].astype(np.float64)]
    dims = np.stack(cols, axis=1)
    rows = np.minimum(np.arange(T) + int(row_shift), T - 1)
    return dims[rows].astype(np.float32)


def action_layout(base_names: list[str], points: list[str]) -> dict:
    """Names and index sets of the new action: [pose 9, (x, y, visible) per point, gripper]."""
    pose, gripper = list(base_names[:9]), list(base_names[9:])
    if len(gripper) != 1:
        raise SystemExit(
            f"expected a 10-D action [pose 9, gripper], got {len(base_names)} dims: {base_names}"
        )
    names, position, vis = list(pose), {}, {}
    for p in points:
        position[p] = [len(names), len(names) + 1]
        vis[p] = len(names) + 2
        names += [f"{p}_x", f"{p}_y", f"{p}_visible"]
    track_dims = list(range(9, len(names)))
    names += gripper
    return {
        "action_names": names,
        "track_dims": track_dims,
        "position_dims": position,
        "visibility_dims": vis,
        "pose_dims": list(range(9)),
        "gripper_dim": len(names) - 1,
    }


def stats_of(values: np.ndarray, keys) -> dict:
    """The statistics lerobot stores for a feature, for the keys it already stores."""
    v = np.asarray(values, np.float64)
    out = {}
    for k in keys:
        if k == "min":
            out[k] = v.min(0)
        elif k == "max":
            out[k] = v.max(0)
        elif k == "mean":
            out[k] = v.mean(0)
        elif k == "std":
            out[k] = v.std(0)
        elif k == "count":
            out[k] = np.array([len(v)])
        elif k in STAT_QUANTILES:
            out[k] = np.quantile(v, STAT_QUANTILES[k], axis=0)
    return out


# ── the build ─────────────────────────────────────────────────────────────────


def compute_new_actions(
    columns: dict,
    tracks_dir: Path,
    *,
    row_shift: int,
    jump_fraction: float,
    jump_pad: int,
    rigid: bool,
    rigid_groups,
    rigid_rel: float,
    rigid_abs: float,
):
    """The new action of every row, from the dataset's columns and the tracks.

    Args:
        columns: numpy arrays "action" [N, 10], "episode_index", "frame_index",
            "index" (global row id), in any row order.
        tracks_dir: the folder track_points.py wrote.

    Returns:
        (new_action [max index + 1, D] indexed by the global row id, layout,
         report {point: {...}}, points, image_hw)
    """
    action = np.asarray(columns["action"], np.float64)
    ep_col, fr_col, ix_col = (
        np.asarray(columns[k]).astype(np.int64) for k in ("episode_index", "frame_index", "index")
    )
    episodes = sorted(set(ep_col.tolist()))
    missing = [e for e in episodes if not (tracks_dir / f"episode_{e:06d}.npz").exists()]
    if missing:
        raise SystemExit(
            f"no tracks for episode(s) {missing[:20]}{' ...' if len(missing) > 20 else ''} "
            f"in {tracks_dir}: run track_points.py on them first"
        )
    points = image_hw = None
    groups = None
    new = None
    report: dict = {}
    for ep in episodes:
        with np.load(tracks_dir / f"episode_{ep:06d}.npz") as d:
            tracks, visible = d["tracks"].astype(np.float64), d["visible"].astype(bool)
            ep_points = [str(p) for p in d["points"]]
            ep_hw = tuple(int(v) for v in d["image_size"])
        if points is None:
            points, image_hw = ep_points, ep_hw
            layout = action_layout([f"a{i}" for i in range(action.shape[1])], points)
            groups = (
                []
                if not rigid
                else [[points.index(n) for n in g] for g in rigid_groups]
                if rigid_groups
                else default_groups(points)
            )
            new = np.zeros((int(ix_col.max()) + 1, len(layout["action_names"])), np.float32)
            report = {
                p: {
                    "frames": 0,
                    "visible_from_tracker": 0,
                    "hidden_by_jumps": 0,
                    "hidden_by_rigidity": 0,
                    "visible_in_target": 0,
                }
                for p in points
            }
        elif ep_points != points or ep_hw != image_hw:
            raise SystemExit(
                f"episode {ep}: tracks for points {ep_points} at {ep_hw}, the first "
                f"episode had {points} at {image_hw}"
            )
        rows = np.flatnonzero(ep_col == ep)
        rows = rows[np.argsort(fr_col[rows])]
        if len(rows) != len(tracks):
            raise SystemExit(f"episode {ep}: {len(rows)} rows but {len(tracks)} tracked frames")
        vis1, by_jump = hide_jumps(tracks, visible, image_hw[1], jump_fraction, jump_pad)
        vis2, by_rigid = (
            hide_rigid_breaks(tracks, vis1, groups, rigid_rel, rigid_abs)
            if groups
            else (vis1, np.zeros_like(vis1))
        )
        dims = episode_track_dims(tracks, vis2, image_hw, row_shift)
        a = action[rows]
        new[ix_col[rows]] = np.concatenate([a[:, :9], dims, a[:, 9:]], axis=1)
        for i, p in enumerate(points):
            r = report[p]
            r["frames"] += len(rows)
            r["visible_from_tracker"] += int(visible[:, i].sum())
            r["hidden_by_jumps"] += int(by_jump[:, i].sum())
            r["hidden_by_rigidity"] += int(by_rigid[:, i].sum())
            r["visible_in_target"] += int(vis2[:, i].sum())
    return new, layout, report, points, image_hw


def _read_columns(root: Path) -> dict:
    import pandas as pd

    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no data/*/*.parquet under {root}")
    frames = [
        pd.read_parquet(f, columns=["action", "episode_index", "frame_index", "index"])
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    return {
        "action": np.stack(df["action"].to_numpy()),
        "episode_index": df["episode_index"].to_numpy(),
        "frame_index": df["frame_index"].to_numpy(),
        "index": df["index"].to_numpy(),
    }


def _rewrite_episode_stats(out_root: Path, new: np.ndarray, columns: dict) -> None:
    """Per-episode action stats in meta/episodes, for the new action."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ep_col = np.asarray(columns["episode_index"]).astype(np.int64)
    ix_col = np.asarray(columns["index"]).astype(np.int64)
    for path in sorted((out_root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        names = [n for n in table.column_names if n.startswith("stats/action/")]
        if not names:
            continue
        eps = table.column("episode_index").to_numpy()
        keys = [n.split("/")[-1] for n in names]
        per_ep = [stats_of(new[ix_col[ep_col == e]], keys) for e in eps]
        for name, key in zip(names, keys, strict=True):
            field = table.schema.field(name)
            value_type = field.type.value_type
            typ = (
                pa.list_(value_type, len(per_ep[0][key]))
                if pa.types.is_fixed_size_list(field.type)
                else pa.list_(value_type)
            )
            column = pa.array([np.asarray(s[key]).tolist() for s in per_ep], type=typ)
            table = table.set_column(
                table.schema.get_field_index(name), field.with_type(typ), column
            )
        pq.write_table(table, path)


def build(
    root: Path,
    tracks_dir: Path,
    out: Path,
    *,
    repo_id: str | None = None,
    out_repo_id: str | None = None,
    queries: Path | None = None,
    row_shift: int = 1,
    jump_fraction: float = 0.05,
    jump_pad: int = 3,
    rigid: bool = True,
    rigid_groups=None,
    rigid_rel: float = 0.35,
    rigid_abs: float = 8.0,
) -> dict:
    """Write the copy and return its track stamp."""
    import pandas as pd
    from lerobot.datasets import dataset_tools as dt
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if out.exists():
        raise SystemExit(
            f"{out} exists already; give a new --out (the source is never overwritten)"
        )
    src = LeRobotDataset(repo_id or f"local/{root.name}", root=root)
    columns = _read_columns(root)
    new, layout, report, points, image_hw = compute_new_actions(
        columns,
        tracks_dir,
        row_shift=row_shift,
        jump_fraction=jump_fraction,
        jump_pad=jump_pad,
        rigid=rigid,
        rigid_groups=rigid_groups,
        rigid_rel=rigid_rel,
        rigid_abs=rigid_abs,
    )
    base_names = list(src.meta.features["action"].get("names") or [f"a{i}" for i in range(10)])
    layout = action_layout(base_names, points)  # the same dims, with the dataset's own names
    features = {k: dict(v) for k, v in src.meta.features.items()}
    features["action"] = {
        **features["action"],
        "shape": [len(layout["action_names"])],
        "names": layout["action_names"],
    }

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=out_repo_id or f"local/{out.name}",
        fps=src.meta.fps,
        features=features,
        robot_type=src.meta.robot_type,
        root=out,
        use_videos=len(src.meta.video_keys) > 0,
    )
    for path in sorted((root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(path)
        df["action"] = list(new[df["index"].to_numpy().astype(np.int64)])
        dst = out / path.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dt._write_parquet(df, dst, new_meta)
    dt._copy_episodes_metadata_and_stats(src, new_meta)
    if new_meta.video_keys:
        dt._copy_videos(src, new_meta)
    # meta files lerobot does not manage (e.g. crisp_gym's record_config.json)
    for f in (root / "meta").iterdir():
        if f.is_file() and not (out / "meta" / f.name).exists():
            shutil.copy2(f, out / "meta" / f.name)
    stats_path = out / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    keys = list(stats.get("action", {})) or ["min", "max", "mean", "std", "count"]
    stats["action"] = {
        k: np.asarray(v).tolist() for k, v in stats_of(new[columns["index"]], keys).items()
    }
    stats_path.write_text(json.dumps(stats, indent=4))
    _rewrite_episode_stats(out, new, columns)

    stamp = {
        "version": 1,
        **layout,
        "points": points,
        "image_size": list(image_hw),
        "camera": json.loads(queries.read_text()).get("camera") if queries else None,
        "coordinates": "x / (W - 1), y / (H - 1) of the camera frame; visible 1 or 0",
        "row_shift": int(row_shift),
        "hidden_fill": "last visible position (first visible before it; image centre if never seen)",
        "cleaning": {
            "jump_fraction": jump_fraction,
            "jump_pad": jump_pad,
            "rigidity": rigid,
            "rigid_groups": [
                [points[i] for i in g]
                for g in (
                    [[points.index(n) for n in g] for g in rigid_groups]
                    if rigid_groups
                    else default_groups(points)
                )
            ]
            if rigid
            else [],
            "rigid_rel": rigid_rel,
            "rigid_abs_px": rigid_abs,
        },
        "report": report,
        "source": {
            "dataset": str(root.resolve()),
            "tracks": str(tracks_dir.resolve()),
            "queries": str(queries.resolve()) if queries else None,
        },
    }
    tmp = out / (STAMP + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=1))
    os.replace(tmp, out / STAMP)
    skipped = sorted(
        p.name for p in root.iterdir() if p.name not in ("data", "meta", "videos", "images")
    )
    if skipped:
        logger.info("not copied (not part of the LeRobot format): %s", ", ".join(skipped))
    return stamp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a dataset copy whose action also carries the point tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for what goes into the track dims.",
    )
    p.add_argument("--root", required=True, help="source LeRobot dataset (not changed)")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--tracks", required=True, help="folder written by track_points.py")
    p.add_argument("--queries", default=None, help="the clicks file, recorded in the stamp")
    p.add_argument("--out", required=True, help="new dataset folder (must not exist)")
    p.add_argument("--out-repo-id", default=None)
    p.add_argument(
        "--row-shift",
        type=int,
        default=1,
        help="action row t takes the tracks of image row t + this",
    )
    p.add_argument("--jump-fraction", type=float, default=0.05)
    p.add_argument("--jump-pad", type=int, default=3)
    p.add_argument(
        "--rigid-groups",
        nargs="*",
        default=None,
        help='points on one rigid object, comma-separated, e.g. "button_a,button_b" "door_a,door_b" '
        "(default: by name before the last _)",
    )
    p.add_argument("--no-rigid-check", action="store_true")
    p.add_argument("--rigid-rel", type=float, default=0.35)
    p.add_argument("--rigid-abs", type=float, default=8.0, help="pixels")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    groups = [g.split(",") for g in args.rigid_groups] if args.rigid_groups else None
    stamp = build(
        Path(args.root),
        Path(args.tracks),
        Path(args.out),
        repo_id=args.repo_id,
        out_repo_id=args.out_repo_id,
        queries=Path(args.queries) if args.queries else None,
        row_shift=args.row_shift,
        jump_fraction=args.jump_fraction,
        jump_pad=args.jump_pad,
        rigid=not args.no_rigid_check,
        rigid_groups=groups,
        rigid_rel=args.rigid_rel,
        rigid_abs=args.rigid_abs,
    )
    logger.info("action: %d dims %s", len(stamp["action_names"]), stamp["action_names"])
    logger.info(
        "%-10s %8s %10s %10s %10s %10s", "point", "frames", "tracker", "-jumps", "-rigid", "target"
    )
    for p, r in stamp["report"].items():
        logger.info(
            "%-10s %8d %9.1f%% %10d %10d %9.1f%%",
            p,
            r["frames"],
            100 * r["visible_from_tracker"] / max(r["frames"], 1),
            r["hidden_by_jumps"],
            r["hidden_by_rigidity"],
            100 * r["visible_in_target"] / max(r["frames"], 1),
        )
    logger.info(
        "wrote %s. Train with lerobot_relative_pose.py --dataset.root=%s (multi_task_dit; "
        "--track-loss-weight to change the weight of the track dims).",
        args.out,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
