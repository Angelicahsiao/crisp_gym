"""Migrate a legacy (Euler + delta-command) LeRobot dataset to the UMI
absolute-on-disk rot6d schema so it can be trained with
``lerobot_relative_pose.py``.

Why this exists
---------------
Datasets collected with the OLD recording scripts store:
  * ``observation.state.cartesian`` = ``[x, y, z, roll, pitch, yaw]``  (Euler, 6d)
  * ``action``                      = a **delta pose command** (already relative,
                                       produced by the teleop ``stream_fn``)

``lerobot_relative_pose.py`` (the rot6d + relative-at-train wrapper) requires
the UMI convention instead:
  * ``observation.state.cartesian`` = ``[x, y, z, rot6d(6)]``            (9d)
  * ``action``                      = the **absolute** ``next_tcp_pose``
                                       (``pos(3) + rot6d(6) [+ gripper]``); the
                                       trainer relativises it itself.

How it works — file surgery, NO video re-encode
-----------------------------------------------
This script copies the dataset directory verbatim (so the **camera videos are
byte-identical** — no lossy AV1 re-encode, no decoder corruption) and rewrites
only the low-dim Parquet columns plus ``meta/info.json`` (and stats). It mirrors
``postprocess_align_datasets.py``'s proven approach.

Per frame it:
  1. converts ``observation.state.cartesian`` Euler(6) -> rot6d(9)
     (scipy ``from_euler("xyz")`` -> first two rows of R, the UMI convention
     used everywhere else in this repo — see crisp_py ``utils/geometry.py``);
  2. rebuilds the concatenated ``observation.state`` from its sub-features;
  3. **discards the old delta action** and sets
     ``action[t] = absolute measured TCP at t+1`` (``next_tcp_pose``,
     lookahead 1) from the converted cartesian pose, with the gripper from
     ``observation.state.gripper`` at ``t+1``. The last frame of each episode
     repeats its own pose (zero relative motion after conversion).

Images, sensors, videos and every other column/file are copied unchanged.
Stats for the three rewritten keys are recomputed from the new Parquet values;
``lerobot_relative_pose.py`` further recomputes the relative-pose stats at load.

Assumptions (validated at start; the script aborts with a clear message if not)
------------------------------------------------------------------------------
  * ``observation.state.cartesian`` exists, dim 6, last 3 dims Euler xyz radians
    (matching ``Pose.to_pos_euler_array`` in crisp_py).
  * a gripper scalar is available as ``observation.state.gripper`` (used for the
    reconstructed action's gripper channel). If absent, pass
    ``--no-action-gripper`` to emit a pose-only (9d) action.

Run it in the lerobot environment (no ROS needed). ``--input``/``--output`` are
dataset **root directories** (each with ``data/`` and ``meta/``), NOT repo ids.

Example
-------
    python crisp_gym/scripts/migrate_euler_delta_to_rot6d.py \\
        --input  /path/to/old_euler_delta_demo \\
        --output /path/to/old_euler_delta_demo_rot6d \\
        --dry-run          # inspect the planned schema change first

Then train as in USAGE.md §8:
    python lerobot_relative_pose.py \\
        --dataset.repo_id=/path/to/old_euler_delta_demo_rot6d \\
        --policy.type=diffusion --output_dir=outputs/train/umi ...
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

logger = logging.getLogger("migrate_euler_delta_to_rot6d")

CARTESIAN_KEY = "observation.state.cartesian"
GRIPPER_KEY = "observation.state.gripper"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

ROT6D_NAMES = [f"rot6d_{i}" for i in range(6)]
CART_NAMES = ["x", "y", "z"] + ROT6D_NAMES


# ── pose conversion ───────────────────────────────────────────────────────────

def euler_pose_to_rot6d(pose6: np.ndarray) -> np.ndarray:
    """``[x,y,z, roll,pitch,yaw]`` (xyz radians) -> ``[x,y,z, rot6d(6)]``.

    rot6d = first two ROWS of the rotation matrix, flattened row-major — the
    UMI/pytorch3d convention used across this repo.
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    pos = pose6[:3]
    mat = Rotation.from_euler("xyz", pose6[3:6]).as_matrix()
    rot6d = mat[:2, :].flatten()
    return np.concatenate([pos, rot6d]).astype(np.float32)


# ── dataset helpers (mirrors postprocess_align_datasets.py) ───────────────────

def data_parquet_files(dataset_dir: Path) -> list[Path]:
    """All frame-data parquet files, layout-agnostic.

    v2.x: data/chunk-*/episode_XXXXXX.parquet (one episode per file).
    v3.0: data/chunk-*/file-XXX.parquet (many episodes concatenated per file).
    Both are found by globbing every *.parquet under data/; episodes are then
    separated by the episode_index column, so the same code handles both.
    """
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No frame parquet files under {dataset_dir}/data")
    return files


def load_info(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def state_subfeature_order(features: dict) -> list[str]:
    """Sub-features that make up the concatenated ``observation.state`` vector,
    in info.json order (matches how recording concatenated them)."""
    return [
        k for k in features
        if k.startswith("observation.state.") and k != STATE_KEY
    ]


def feature_dim(feat: dict) -> int:
    return int(np.prod(feat["shape"]))


# ── info.json feature rewriting ───────────────────────────────────────────────

def rewrite_features(features: dict, action_has_gripper: bool) -> dict:
    if CARTESIAN_KEY not in features:
        raise KeyError(
            f"'{CARTESIAN_KEY}' not in dataset features {list(features)}. "
            "This script expects a cartesian pose state from the old recorder."
        )
    if feature_dim(features[CARTESIAN_KEY]) != 6:
        raise ValueError(
            f"'{CARTESIAN_KEY}' has dim {feature_dim(features[CARTESIAN_KEY])}, "
            "expected 6 ([x,y,z,roll,pitch,yaw]). If it is already 9 the data is "
            "likely already rot6d — no migration needed."
        )

    # cartesian 6 -> 9
    features[CARTESIAN_KEY]["shape"] = [9]
    features[CARTESIAN_KEY]["names"] = list(CART_NAMES)

    # concatenated observation.state: rebuild names/length from sub-features
    if STATE_KEY in features:
        names: list[str] = []
        for k in state_subfeature_order(features):
            sub = features[k]
            sub_names = sub.get("names") or [
                f"{k.split('.')[-1]}_{i}" for i in range(feature_dim(sub))
            ]
            names += list(sub_names)
        features[STATE_KEY]["shape"] = [len(names)]
        features[STATE_KEY]["names"] = names

    # action: pos(3) + rot6d(6) [+ gripper]
    act_dim = 9 + (1 if action_has_gripper else 0)
    features[ACTION_KEY]["shape"] = [act_dim]
    features[ACTION_KEY]["names"] = list(CART_NAMES) + (["gripper"] if action_has_gripper else [])
    return features


# ── stats recomputation for the rewritten keys ────────────────────────────────

def _compute_stats(values: np.ndarray, stat_names) -> dict:
    """Per-dimension stats for a [N, D] array, for the requested stat names.

    Supports LeRobot's names: mean/std/min/max/count and quantiles qNN
    (q01, q10, q50, q90, q99). Returns each as a np.ndarray (count as [N]).
    Unknown names map to None (left untouched by callers).
    """
    values = np.asarray(values, dtype=np.float64)
    out: dict = {}
    for s in stat_names:
        if s == "mean":
            out[s] = values.mean(axis=0)
        elif s == "std":
            out[s] = values.std(axis=0) + 1e-8
        elif s == "min":
            out[s] = values.min(axis=0)
        elif s == "max":
            out[s] = values.max(axis=0)
        elif s == "count":
            out[s] = np.array([values.shape[0]])
        elif len(s) > 1 and s[0] == "q" and s[1:].isdigit():
            out[s] = np.quantile(values, int(s[1:]) / 100.0, axis=0)
        else:
            out[s] = None
    return out


def update_stats_json(out_dir: Path, agg: dict[str, np.ndarray]):
    """Patch aggregate meta/stats.json for the rewritten keys (preserving each
    key's existing set of stat names, incl. quantiles)."""
    stats_path = out_dir / "meta" / "stats.json"
    if not stats_path.exists():
        return
    with open(stats_path) as f:
        stats = json.load(f)
    for k, vals in agg.items():
        if k not in stats:
            continue
        names = list(stats[k].keys())
        cs = _compute_stats(vals, names)
        for name in names:
            if cs.get(name) is not None:
                stats[k][name] = cs[name].astype(np.float64).tolist()
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info("  updated meta/stats.json for %s", list(agg))


def update_v3_episode_stats(out_dir: Path, per_ep: dict[int, dict[str, np.ndarray]]):
    """Patch v3.0 meta/episodes/**/*.parquet per-episode stats columns
    (``stats/<key>/<stat>``) for the rewritten keys."""
    ep_files = sorted((out_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        return
    keys = [CARTESIAN_KEY, STATE_KEY, ACTION_KEY]
    for ef in ep_files:
        df = pd.read_parquet(ef)
        ep_col = df["episode_index"].to_numpy()

        # Build new whole-column value lists (assigning per-cell via df.at
        # unwraps 1-element arrays into 0-d scalars, which pyarrow rejects).
        cols = {}
        for k in keys:
            prefix = f"stats/{k}/"
            for c in df.columns:
                if c.startswith(prefix):
                    cols[c] = list(df[c].to_list())

        changed = False
        for i in range(len(df)):
            ep = int(np.asarray(ep_col[i]))
            if ep not in per_ep:
                continue
            for k in keys:
                if k not in per_ep[ep]:
                    continue
                prefix = f"stats/{k}/"
                names = [c[len(prefix):] for c in df.columns if c.startswith(prefix)]
                if not names:
                    continue
                cs = _compute_stats(per_ep[ep][k], names)
                n = int(per_ep[ep][k].shape[0])
                for name in names:
                    if cs.get(name) is None:
                        continue
                    col = prefix + name
                    ref = cols[col][i]
                    ref_is_arr = isinstance(ref, np.ndarray) and np.asarray(ref).ndim >= 1
                    if name == "count":
                        # preserve scalar-vs-1d-array form of the original cell
                        cols[col][i] = (
                            np.full(np.asarray(ref).shape, n, dtype=np.asarray(ref).dtype)
                            if ref_is_arr else type(ref)(n) if ref is not None else n
                        )
                    else:
                        arr = np.asarray(cs[name]).reshape(-1)
                        if ref_is_arr:
                            arr = arr.astype(np.asarray(ref).dtype)
                        cols[col][i] = arr
                    changed = True

        if changed:
            for col, vals in cols.items():
                df[col] = vals
            df.to_parquet(ef, index=False)
            logger.info("  updated %s", ef.relative_to(out_dir))


def update_stats_files(out_dir: Path, per_ep: dict[int, dict[str, np.ndarray]]):
    """Update aggregate (stats.json) and per-episode stats (v3.0 episodes
    parquet, or v2.x episodes_stats.jsonl) for the rewritten keys."""
    keys = [CARTESIAN_KEY, STATE_KEY, ACTION_KEY]
    agg = {
        k: np.concatenate([per_ep[e][k] for e in sorted(per_ep) if k in per_ep[e]], axis=0)
        for k in keys
        if any(k in per_ep[e] for e in per_ep)
    }
    update_stats_json(out_dir, agg)
    update_v3_episode_stats(out_dir, per_ep)

    # v2.x per-episode stats (older datasets)
    ep_stats_path = out_dir / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.exists():
        lines = []
        with open(ep_stats_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep = rec.get("episode_index")
                if ep in per_ep and "stats" in rec:
                    for k in keys:
                        if k in per_ep[ep] and k in rec["stats"]:
                            names = list(rec["stats"][k].keys())
                            cs = _compute_stats(per_ep[ep][k], names)
                            for name in names:
                                if cs.get(name) is not None:
                                    rec["stats"][k][name] = cs[name].astype(np.float64).tolist()
                lines.append(json.dumps(rec))
        with open(ep_stats_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("  updated meta/episodes_stats.jsonl")


# ── main migration ────────────────────────────────────────────────────────────

def migrate(args) -> int:
    src = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        logger.error("No meta/info.json under %s — is this a LeRobot dataset root?", src)
        return 2

    info = load_info(src)
    features = info["features"]
    action_has_gripper = not args.no_action_gripper and GRIPPER_KEY in features
    if not args.no_action_gripper and GRIPPER_KEY not in features:
        logger.warning(
            "'%s' not found; reconstructed action will be pose-only (9d). "
            "Pass --no-action-gripper to silence this.", GRIPPER_KEY,
        )

    # plan
    new_features = rewrite_features(json.loads(json.dumps(features)), action_has_gripper)
    logger.info("Planned feature changes:")
    for key in (CARTESIAN_KEY, STATE_KEY, ACTION_KEY):
        if key in features:
            logger.info("  %-28s %s -> %s", key,
                        tuple(features[key]["shape"]), tuple(new_features[key]["shape"]))
    logger.info("  action gripper channel: %s", "yes" if action_has_gripper else "no")

    data_files = data_parquet_files(src)
    logger.info("Frame parquet files: %d (episodes separated by episode_index)", len(data_files))

    if args.dry_run:
        logger.info("--dry-run: no data written.")
        return 0

    if out.exists():
        logger.error("Output %s already exists — remove it first.", out)
        return 2
    logger.info("Copying %s -> %s (videos copied byte-identical)", src, out)
    shutil.copytree(src, out)

    state_order = state_subfeature_order(new_features) if STATE_KEY in new_features else []
    per_ep_frames: dict[int, dict[str, list]] = {}
    state_verified = False

    for pf in data_parquet_files(out):
        df = pd.read_parquet(pf).reset_index(drop=True)

        # original Euler cartesian (kept for the state-composition sanity check)
        cart6 = [np.asarray(v, np.float32).reshape(-1) for v in df[CARTESIAN_KEY]]

        # Sanity check (once): observation.state must be the ordered concat of
        # its sub-features, or rebuilding it would silently reorder the vector.
        if not state_verified and STATE_KEY in df.columns and state_order:
            old_parts = [
                cart6[0] if sk == CARTESIAN_KEY else np.asarray(df[sk].iloc[0], np.float32).reshape(-1)
                for sk in state_order
            ]
            old_concat = np.concatenate(old_parts)
            existing = np.asarray(df[STATE_KEY].iloc[0], np.float32).reshape(-1)
            if old_concat.shape != existing.shape or not np.allclose(old_concat, existing, atol=1e-4):
                raise ValueError(
                    f"'{STATE_KEY}' is not the ordered concat of its sub-features "
                    f"{state_order} (got len {existing.shape} vs rebuilt {old_concat.shape}). "
                    "Refusing to rebuild it — the info.json sub-feature order does "
                    "not match the stored state vector."
                )
            state_verified = True

        # 1. cartesian Euler(6) -> rot6d(9)  (per row, order-independent)
        cart9 = np.stack([euler_pose_to_rot6d(v) for v in cart6])
        df[CARTESIAN_KEY] = list(cart9)

        # gripper (for action + state rebuild)
        grip = None
        if GRIPPER_KEY in df.columns:
            grip = np.stack([np.asarray(v, np.float32).reshape(-1) for v in df[GRIPPER_KEY]])

        # 2. rebuild concatenated observation.state from sub-features (per row)
        if STATE_KEY in df.columns and state_order:
            parts_per_row = []
            for i in range(len(df)):
                parts = []
                for sk in state_order:
                    if sk == CARTESIAN_KEY:
                        parts.append(cart9[i])
                    elif sk in df.columns:
                        parts.append(np.asarray(df[sk].iloc[i], np.float32).reshape(-1))
                    else:
                        raise KeyError(
                            f"state sub-feature '{sk}' has no parquet column; "
                            "cannot rebuild observation.state."
                        )
                parts_per_row.append(np.concatenate(parts).astype(np.float32))
            df[STATE_KEY] = parts_per_row

        # 3. action = next_tcp_pose (absolute pose at t+1), PER EPISODE.
        #    A single v3.0 parquet holds many episodes; the lookahead must not
        #    cross an episode boundary, so group by episode_index and, within
        #    each episode, order by frame_index. Last frame of each episode
        #    repeats its own pose.
        actions: list = [None] * len(df)
        ep_col = (
            df["episode_index"].to_numpy()
            if "episode_index" in df.columns
            else np.zeros(len(df), dtype=int)
        )
        fi_col = (
            df["frame_index"].to_numpy()
            if "frame_index" in df.columns
            else np.arange(len(df))
        )
        for ep in np.unique(ep_col):
            pos = np.where(ep_col == ep)[0]
            pos = pos[np.argsort(fi_col[pos])]  # episode rows in temporal order
            n = len(pos)
            for j in range(n):
                nxt = pos[min(j + 1, n - 1)]
                a = cart9[nxt].astype(np.float32)
                if action_has_gripper and grip is not None:
                    a = np.concatenate([a, grip[nxt].reshape(-1)]).astype(np.float32)
                actions[pos[j]] = a

            ep_key = int(ep)
            frames = per_ep_frames.setdefault(ep_key, {CARTESIAN_KEY: [], ACTION_KEY: [], STATE_KEY: []})
            frames[CARTESIAN_KEY].extend(cart9[pos])
            frames[ACTION_KEY].extend([actions[p] for p in pos])
            if STATE_KEY in df.columns and state_order:
                frames[STATE_KEY].extend([df[STATE_KEY].iloc[p] for p in pos])

        df[ACTION_KEY] = actions
        df.to_parquet(pf, index=False)

    # collapse per-episode frame lists to arrays
    per_ep_arrays: dict[int, dict[str, np.ndarray]] = {}
    for ep, d in per_ep_frames.items():
        per_ep_arrays[ep] = {
            k: np.stack(v) for k, v in d.items() if v
        }

    # 4. meta updates
    info["features"] = new_features
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    update_stats_files(out, per_ep_arrays)

    logger.info("Done. New dataset: %s", out)
    logger.info("Videos were copied unchanged; only low-dim columns were rewritten.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a legacy Euler + delta-command LeRobot dataset to "
        "the UMI absolute rot6d schema for lerobot_relative_pose.py training "
        "(file surgery — copies videos unchanged, rewrites low-dim columns)."
    )
    parser.add_argument("--input", required=True, help="Source dataset root dir.")
    parser.add_argument("--output", required=True, help="Destination dataset root dir.")
    parser.add_argument(
        "--no-action-gripper", action="store_true",
        help="Emit a pose-only (9d) action instead of pos+rot6d+gripper (10d).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the planned schema change without writing.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    return migrate(args)


if __name__ == "__main__":
    sys.exit(main())
