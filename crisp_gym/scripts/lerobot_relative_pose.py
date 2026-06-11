"""Relative-pose dataset wrapper and training launcher for LeRobot 0.4.4.

Implements UMI's relative pose representation at dataloader level: every pose
(observation history and action horizon) is re-expressed relative to the
current observed TCP pose — T_rel = T_current^{-1} @ T_target — after
LeRobot's temporal sampling (delta_timestamps) has stacked the frames.

The dataset on disk stays in absolute poses. With diffusion policy's default
n_obs_steps=2, the wrapper produces:

    observation.state.cartesian  (2, 6):  [t-1 rel. to t,  identity (zeros)]
    action                       (16, 7): each step rel. to t, gripper untouched

Normalization stats are recomputed on the relative values (LeRobot normalizes
with dataset-wide stats which were computed on absolute poses and would
otherwise be wrong).

Usage on the training PC (lerobot==0.4.4 installed):

    python lerobot_relative_pose.py \\
        --dataset.repo_id=my_org/umi_demo \\
        --policy.type=diffusion \\
        --output_dir=outputs/train/umi_relative \\
        ... (any other lerobot-train args)

This script wraps `lerobot-train`: it patches dataset construction and then
hands control to the standard training loop.

At inference, invert with: T_cmd = T_current_robot_tcp @ T_rel  (compose the
policy's relative output with the robot's current TCP pose in its own base
frame — no OptiTrack-to-robot calibration needed).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Keys holding absolute poses as [x, y, z, roll, pitch, yaw(, gripper...)]
POSE_OBS_KEYS = ["observation.state.cartesian"]
POSE_ACTION_KEY = "action"
# Which key provides the "current" base pose (last frame of its window)
BASE_KEY = "observation.state.cartesian"


# ── Pose math (Euler XYZ <-> 4x4) ─────────────────────────────────────────────

def _pose6d_to_mat(pose: np.ndarray) -> np.ndarray:
    """[..., 6] euler-xyz pose -> [..., 4, 4] homogeneous matrices."""
    pose = np.asarray(pose, dtype=np.float64)
    batch = pose.shape[:-1]
    T = np.tile(np.eye(4), (*batch, 1, 1))
    T[..., :3, :3] = Rotation.from_euler("xyz", pose[..., 3:6].reshape(-1, 3)).as_matrix().reshape(*batch, 3, 3)
    T[..., :3, 3] = pose[..., :3]
    return T


def _mat_to_pose6d(T: np.ndarray) -> np.ndarray:
    """[..., 4, 4] homogeneous matrices -> [..., 6] euler-xyz pose."""
    batch = T.shape[:-2]
    pos = T[..., :3, 3]
    rot = Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_euler("xyz").reshape(*batch, 3)
    return np.concatenate([pos, rot], axis=-1)


def make_relative(base_pose6d: np.ndarray, poses6d: np.ndarray) -> np.ndarray:
    """Express poses6d [..., 6] relative to base_pose6d [6]: T_base^-1 @ T."""
    T_base_inv = np.linalg.inv(_pose6d_to_mat(base_pose6d))
    T = _pose6d_to_mat(poses6d)
    return _mat_to_pose6d(T_base_inv @ T)


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class RelativePoseDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset; converts absolute poses to relative in __getitem__.

    The base frame is the LAST timestep of BASE_KEY's window (the current
    observation, index 0 in delta terms), matching UMI's pose_mat[-1].
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name):
        # Delegate everything else (.meta, .num_frames, .episodes, ...)
        return getattr(self._dataset, name)

    @staticmethod
    def convert_item(item: dict) -> dict:
        base = item[BASE_KEY]
        # (T, 6) stacked window or (6,) single frame
        base_np = base.numpy() if isinstance(base, torch.Tensor) else np.asarray(base)
        base_pose = base_np[-1] if base_np.ndim == 2 else base_np

        for key in POSE_OBS_KEYS:
            if key not in item:
                continue
            v = item[key]
            v_np = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            rel = make_relative(base_pose, v_np)
            item[key] = torch.from_numpy(rel.astype(np.float32))

        if POSE_ACTION_KEY in item:
            a = item[POSE_ACTION_KEY]
            a_np = a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
            rel = make_relative(base_pose, a_np[..., :6])
            out = np.concatenate([rel, a_np[..., 6:]], axis=-1)
            item[POSE_ACTION_KEY] = torch.from_numpy(out.astype(np.float32))

        return item

    def __getitem__(self, idx: int) -> dict:
        return self.convert_item(self._dataset[idx])


# ── Stats recomputation ───────────────────────────────────────────────────────

def recompute_relative_stats(wrapped: RelativePoseDataset, num_samples: int = 2000) -> None:
    """Patch dataset.meta.stats for pose keys with stats of the RELATIVE values.

    LeRobot's NormalizerProcessor uses dataset-wide stats computed on the raw
    (absolute) Parquet values. After the relative conversion the distributions
    are completely different, so we sample the wrapped dataset and overwrite
    the stats for the converted keys.
    """
    keys = [k for k in POSE_OBS_KEYS if k in wrapped.meta.stats] + [POSE_ACTION_KEY]
    n = len(wrapped)
    indices = np.linspace(0, n - 1, min(num_samples, n)).astype(int)

    collected: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for i in indices:
        item = wrapped[int(i)]
        for k in keys:
            if k in item:
                v = item[k]
                v = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                collected[k].append(v.reshape(-1, v.shape[-1]))

    for k, chunks in collected.items():
        if not chunks:
            continue
        data = np.concatenate(chunks, axis=0)
        wrapped.meta.stats[k] = {
            "mean": torch.from_numpy(data.mean(axis=0).astype(np.float32)),
            "std": torch.from_numpy((data.std(axis=0) + 1e-8).astype(np.float32)),
            "min": torch.from_numpy(data.min(axis=0).astype(np.float32)),
            "max": torch.from_numpy(data.max(axis=0).astype(np.float32)),
        }
        logger.info(f"Recomputed relative-pose stats for '{k}' over {len(data)} frames.")


# ── Training launcher ─────────────────────────────────────────────────────────

def main():
    """Run lerobot-train with the relative-pose dataset wrapper injected."""
    import lerobot.scripts.lerobot_train as lerobot_train

    original_make_dataset = lerobot_train.make_dataset

    def make_dataset_with_relative(cfg):
        dataset = original_make_dataset(cfg)
        wrapped = RelativePoseDataset(dataset)
        logger.info("Wrapped dataset with RelativePoseDataset (UMI-style relative poses).")
        recompute_relative_stats(wrapped)
        return wrapped

    lerobot_train.make_dataset = make_dataset_with_relative

    logging.basicConfig(level=logging.INFO)
    lerobot_train.main()


if __name__ == "__main__":
    main()
