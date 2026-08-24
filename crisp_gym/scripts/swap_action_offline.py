"""Offline: write a copy of a dataset whose action ARM dims are the commanded pose.

The online wrapper (train_action_from_target_cartesian.py) fails on lerobot
versions that fix their windowed-key set at dataset construction from the
policy's features — extra.target_cartesian is not a policy feature, so it is
never windowed and cannot be swapped into the action horizon at load time. This
script sidesteps lerobot's dataloader entirely: it rewrites the `action` column
ON DISK so the arm dims come from extra.target_cartesian, then you train the
result with the plain launcher (train_absolute_next_pose.py). lerobot then sees
an ordinary dataset with an ordinary action column.

    action[t]  :  [ next_tcp_pose(9) , gripper ]         (recorded: achieved)
      becomes  :  [ target_cartesian[t](9) , gripper ]   (commanded)

Same-frame swap: it replaces action[t]'s 9 arm dims with target_cartesian[t] and
keeps action[t]'s gripper. Because lerobot builds the diffusion action chunk from
consecutive `action` rows, the resulting chunk is exactly the commanded-pose
sequence — identical to what the online window-swap intended.

File surgery, mirroring migrate_euler_delta_to_rot6d.py: videos are copied
byte-for-byte (never re-encoded), only the low-dim `action` column is rewritten,
and the `action` stats are recomputed (aggregate meta/stats.json + v3.0
per-episode meta/episodes/*.parquet + v2.x episodes_stats.jsonl). The action
SCHEMA is unchanged (still 10-D rot6d+gripper, same names), so meta/info.json is
left as-is; a meta/action_repr.json marks what the column now means.

REFUSES on FACTR/JIC data (target_cartesian constant per episode) — same guard
as the online script.

GPU-PC / preprocessing script: numpy + pandas + pyarrow, crisp-import-free.
Reuses the stats helpers from migrate_euler_delta_to_rot6d.py (same directory).

Usage:

    python crisp_gym/crisp_gym/scripts/swap_action_offline.py \\
        --input datasets/franka_electricbox/lerobot \\
        --output datasets/franka_electricbox_targetcmd/lerobot

    python crisp_gym/crisp_gym/scripts/train_absolute_next_pose.py \\
        --dataset.repo_id=franka_electricbox_targetcmd \\
        --dataset.root=datasets/franka_electricbox_targetcmd/lerobot \\
        --policy.type=diffusion --policy.push_to_hub=false \\
        --output_dir=outputs/train/franka_targetcmd --batch_size=64 --steps=200000
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

# Reuse the tested v2.x/v3.0 stats machinery from the sibling migration script.
# sys.path[0] is this file's directory (crisp_gym/.../scripts) when run as a
# script, so the flat import resolves without packaging.
from migrate_euler_delta_to_rot6d import (  # noqa: E402
    data_parquet_files,
    load_info,
    update_stats_files,
)

logger = logging.getLogger(__name__)

ACTION_KEY = "action"
TARGET_KEY = "extra.target_cartesian"
ARM_DIMS = 9          # pos(3) + rot6d(6); action dim 9 is the gripper, kept as-is


def swap_arm(act: np.ndarray, tc: np.ndarray, arm_dims: int = ARM_DIMS) -> np.ndarray:
    """Return a copy of `act` [N, A] with its first `arm_dims` cols set to `tc`.

    The remaining columns of `act` (the gripper) are untouched. `tc` is [N, >=arm_dims].
    """
    out = np.asarray(act, dtype=np.float32).copy()
    out[:, :arm_dims] = np.asarray(tc, dtype=np.float32)[:, :arm_dims]
    return out


def _col_to_2d(series, width: int | None = None) -> np.ndarray:
    """Stack a parquet column of per-row vectors into [N, D]."""
    arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in series])
    if width is not None and arr.shape[1] != width:
        raise ValueError(f"expected width {width}, got {arr.shape[1]}")
    return arr


def _assert_features(info: dict) -> None:
    features = info.get("features", {})
    for key, dims in ((ACTION_KEY, ARM_DIMS + 1), (TARGET_KEY, ARM_DIMS)):
        if key not in features:
            raise KeyError(
                f"dataset has no '{key}' feature; need both '{ACTION_KEY}' and "
                f"'{TARGET_KEY}'. present: {sorted(features)}"
            )
        d = int(np.prod(features[key]["shape"]))
        if d != dims:
            raise ValueError(
                f"'{key}' is {d}D, expected {dims}D "
                f"({'[pos3,rot6d6,gripper]' if key == ACTION_KEY else '[pos3,rot6d6]'})."
            )


def _assert_target_varies(src: Path, atol: float) -> None:
    """Refuse the FACTR/JIC dead-column case — target constant within episodes."""
    max_spread = 0.0
    for pf in data_parquet_files(src):
        df = pd.read_parquet(pf, columns=[TARGET_KEY, "episode_index"])
        tc = _col_to_2d(df[TARGET_KEY])
        ep = df["episode_index"].to_numpy()
        for e in np.unique(ep):
            seg = tc[ep == e]
            if seg.shape[0] > 1:
                max_spread = max(max_spread, float(np.ptp(seg, axis=0).max()))
    if max_spread <= atol:
        raise ValueError(
            f"'{TARGET_KEY}' is CONSTANT within every episode (max per-episode "
            f"spread {max_spread:.2e} <= {atol:.0e}). This is the FACTR/JIC "
            "signature: robot._target_pose is only written by Cartesian control, "
            "so on joint-driven data it is the post-home pose frozen per episode. "
            "Nothing to swap. Use a Cartesian-driven dataset, or train the "
            "recorded action with train_absolute_next_pose.py."
        )
    logger.info("%s varies (max per-episode spread %.3g) — OK.", TARGET_KEY, max_spread)


def run(args) -> int:
    src = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        logger.error("No meta/info.json under %s — is this a LeRobot dataset root?", src)
        return 2

    info = load_info(src)
    _assert_features(info)
    _assert_target_varies(src, args.atol)

    if args.dry_run:
        logger.info("--dry-run: features OK, target varies. No data written.")
        return 0

    if out.exists():
        logger.error("Output %s already exists — remove it first.", out)
        return 2
    logger.info("Copying %s -> %s (videos copied byte-identical)", src, out)
    shutil.copytree(src, out)

    per_ep: dict[int, dict[str, list]] = {}
    total = 0
    for pf in data_parquet_files(out):
        df = pd.read_parquet(pf).reset_index(drop=True)
        act = _col_to_2d(df[ACTION_KEY], ARM_DIMS + 1)
        tc = _col_to_2d(df[TARGET_KEY], ARM_DIMS)
        new_act = swap_arm(act, tc)
        df[ACTION_KEY] = list(new_act)
        df.to_parquet(pf, index=False)
        total += len(df)

        ep_col = (
            df["episode_index"].to_numpy()
            if "episode_index" in df.columns
            else np.zeros(len(df), dtype=int)
        )
        for e in np.unique(ep_col):
            per_ep.setdefault(int(e), {ACTION_KEY: []})[ACTION_KEY].extend(new_act[ep_col == e])

    per_ep_arrays = {e: {ACTION_KEY: np.stack(d[ACTION_KEY])} for e, d in per_ep.items()}
    update_stats_files(out, per_ep_arrays)

    # Provenance: the action column's arm dims are now the COMMANDED pose. Schema
    # is unchanged, so info.json is untouched; this marks the semantics.
    (out / "meta" / "action_repr.json").write_text(json.dumps({
        "action": {
            "source": TARGET_KEY,
            "pose_repr": "absolute",
            "layout": "[x, y, z, rot6d(6), gripper]",
            "note": "ARM dims (0-8) are extra.target_cartesian (pose COMMANDED to "
                    "the CIC); gripper (dim 9) is the recorded next_tcp_pose "
                    "gripper. Same-frame swap. Observations unchanged.",
        },
        "derived_from": str(src),
    }, indent=2))

    logger.info("Done: %d frames, %d episodes. New dataset: %s", total, len(per_ep_arrays), out)
    logger.info("Train it with train_absolute_next_pose.py (--dataset.root=%s).", out)
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Write a copy of a LeRobot dataset whose action arm dims are "
        "extra.target_cartesian (commanded pose). File surgery — videos copied "
        "unchanged, only the action column rewritten + its stats."
    )
    p.add_argument("--input", required=True, help="Source dataset root (has data/ and meta/).")
    p.add_argument("--output", required=True, help="Destination dataset root (must not exist).")
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Per-episode spread below which target_cartesian is 'constant' (refuse).")
    p.add_argument("--dry-run", action="store_true", help="Validate only; write nothing.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
