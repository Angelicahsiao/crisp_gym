"""Post-process a recorded LeRobot dataset to convert absolute poses to relative poses.

Converts both observation cartesian state and actions from absolute TCP poses to
poses expressed relative to the current (most recent) observed TCP pose — matching
UMI's training representation: T_rel = T_current^{-1} @ T_target.

This is equivalent to umi_dataset.py's convert_pose_mat_rep(..., pose_rep='relative')
applied per episode during load time, but baked into the Parquet files so the
training dataloader can treat them as regular absolute values.

The original dataset is NOT modified. Output is written to a new directory.

Usage:
    python postprocess_relative_pose.py \\
        --input-dir ~/.cache/huggingface/lerobot/my_org/umi_demo \\
        --output-dir ~/.cache/huggingface/lerobot/my_org/umi_demo_relative \\
        --obs-window 2

The script handles the 1-step lookahead (action[t] = pose[t+1]) by treating each
row's action as a future absolute pose and converting it relative to the row's
current observation pose.

Columns processed:
    observation.state.cartesian  [x, y, z, roll, pitch, yaw]  →  relative
    action                       [x, y, z, roll, pitch, yaw, gripper]  →  pos+rot relative, gripper unchanged
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ── Pose math helpers ──────────────────────────────────────────────────────────

def euler_xyz_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, roll, pitch, yaw] (Euler XYZ) to 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def mat_to_euler_xyz(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous matrix to [x, y, z, roll, pitch, yaw] (Euler XYZ)."""
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return np.concatenate([pos, rot]).astype(np.float32)


def make_relative(base_pose6d: np.ndarray, target_pose6d: np.ndarray) -> np.ndarray:
    """Express target_pose relative to base_pose: T_rel = T_base^{-1} @ T_target."""
    T_base = euler_xyz_to_mat(base_pose6d)
    T_target = euler_xyz_to_mat(target_pose6d)
    T_rel = np.linalg.inv(T_base) @ T_target
    return mat_to_euler_xyz(T_rel)


# ── Per-episode processing ─────────────────────────────────────────────────────

def process_episode(df: pd.DataFrame, obs_col: str, action_col: str) -> pd.DataFrame:
    """Convert absolute poses in one episode DataFrame to relative poses.

    For each row t:
        obs_relative[t]    = T_obs[t]^{-1} @ T_obs[t]          = identity (zeroed)
                             ... but we keep a sliding window of obs_window frames
        action_relative[t] = T_obs_current[t]^{-1} @ T_action[t]

    The observation state[t] is expressed relative to itself (→ zeros for pos,
    identity for rot), which is what UMI does: base_pose_mat = pose_mat[-1] (current).

    For a temporal obs window > 1, older observations in the window are expressed
    relative to the *current* frame (the most recent one), which is what UMI's
    dataloader does at training time.

    Args:
        df: Episode DataFrame sorted by frame index.
        obs_col: Column name for cartesian observation [x,y,z,rx,ry,rz].
        action_col: Column name for action [x,y,z,rx,ry,rz,gripper].

    Returns:
        Modified DataFrame with relative pose columns.
    """
    df = df.copy()
    n = len(df)

    obs_abs = np.stack(df[obs_col].values).astype(np.float64)    # (N, 6)
    action_abs = np.stack(df[action_col].values).astype(np.float64)  # (N, 7)

    obs_rel = np.zeros_like(obs_abs)
    action_rel = np.zeros_like(action_abs)

    for t in range(n):
        base = obs_abs[t]  # current frame is the reference

        # obs relative to current frame — itself → zeroed position, identity rot
        obs_rel[t] = make_relative(base, obs_abs[t])

        # action relative to current obs frame
        action_rel[t, :6] = make_relative(base, action_abs[t, :6])
        action_rel[t, 6] = action_abs[t, 6]  # gripper unchanged

    df[obs_col] = list(obs_rel.astype(np.float32))
    df[action_col] = list(action_rel.astype(np.float32))

    return df


# ── Dataset-level processing ───────────────────────────────────────────────────

def find_episode_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode_*.parquet files found under {data_dir}")
    return files


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    obs_col: str,
    action_col: str,
    dry_run: bool,
) -> None:
    data_in = input_dir / "data"
    if not data_in.exists():
        raise FileNotFoundError(f"Expected data/ subdirectory in {input_dir}")

    if not dry_run:
        if output_dir.exists():
            logger.warning(f"Output directory already exists: {output_dir} — overwriting")
        # Copy everything first (videos, meta, etc.), then overwrite Parquet files
        if output_dir != input_dir:
            logger.info(f"Copying dataset structure {input_dir} → {output_dir}")
            shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)

    episode_files = find_episode_files(data_in)
    logger.info(f"Found {len(episode_files)} episode files")

    for ep_path in episode_files:
        rel_path = ep_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        df = pd.read_parquet(ep_path)

        missing = [c for c in [obs_col, action_col] if c not in df.columns]
        if missing:
            logger.warning(f"Skipping {ep_path.name}: missing columns {missing}")
            continue

        n_frames = len(df)
        df_out = process_episode(df, obs_col=obs_col, action_col=action_col)

        if dry_run:
            first_obs_in  = np.stack(df[obs_col].values)[0]
            first_obs_out = np.stack(df_out[obs_col].values)[0]
            first_act_in  = np.stack(df[action_col].values)[0]
            first_act_out = np.stack(df_out[action_col].values)[0]
            logger.info(
                f"[DRY RUN] {ep_path.name} ({n_frames} frames)\n"
                f"  obs[0]    before: {np.round(first_obs_in, 4)}\n"
                f"  obs[0]    after:  {np.round(first_obs_out, 4)}\n"
                f"  action[0] before: {np.round(first_act_in, 4)}\n"
                f"  action[0] after:  {np.round(first_act_out, 4)}"
            )
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_parquet(out_path, index=False)
            logger.info(f"  ✓ {rel_path}  ({n_frames} frames)")

    if not dry_run:
        # Write a marker so it's clear this dataset was post-processed
        marker = output_dir / "meta" / "postprocess_relative_pose.json"
        import json, datetime
        marker.parent.mkdir(parents=True, exist_ok=True)
        with open(marker, "w") as f:
            json.dump({
                "source": str(input_dir),
                "obs_col": obs_col,
                "action_col": action_col,
                "pose_rep": "relative",
                "processed_at": datetime.datetime.utcnow().isoformat(),
            }, f, indent=4)
        logger.info(f"Wrote postprocess marker to {marker}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert absolute poses in a LeRobot dataset to relative poses."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory of the recorded LeRobot dataset (contains data/, meta/, videos/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the converted dataset. "
            "Defaults to <input-dir>_relative. "
            "Pass the same path as --input-dir to convert in place (destructive)."
        ),
    )
    parser.add_argument(
        "--obs-col",
        type=str,
        default="observation.state.cartesian",
        help="Parquet column name for the cartesian observation [x,y,z,roll,pitch,yaw].",
    )
    parser.add_argument(
        "--action-col",
        type=str,
        default="action",
        help="Parquet column name for the action [x,y,z,roll,pitch,yaw,gripper].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing any files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "_relative")
    output_dir = output_dir.expanduser().resolve()

    logger.info(f"Input:    {input_dir}")
    logger.info(f"Output:   {output_dir}")
    logger.info(f"Obs col:  {args.obs_col}")
    logger.info(f"Act col:  {args.action_col}")
    if args.dry_run:
        logger.info("Mode:     DRY RUN — no files will be written")

    process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        obs_col=args.obs_col,
        action_col=args.action_col,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        logger.info("Done.")


if __name__ == "__main__":
    main()
