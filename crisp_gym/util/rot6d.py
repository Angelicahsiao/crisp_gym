"""Canonical rot6d <-> matrix helpers (UMI / pytorch3d convention).

THE convention (HANDOFF §1.1 — breaking it silently corrupts data/training):
  encode: first two ROWS of the rotation matrix, flattened row-major
          [R00, R01, R02, R10, R11, R12]
  decode: Gram-Schmidt the two 3-vectors, third row = b1 x b2

This module is the single in-package implementation, used by the robot-PC
code (remote_policy, umi_handheld_env). Two scripts intentionally keep a
LOCAL copy of the same math because they must stay import-free of crisp_gym:
  - scripts/lerobot_relative_pose.py   (GPU PC: lerobot+torch+numpy only)
  - scripts/migrate_euler_delta_to_rot6d.py (standalone file surgery)
If you change anything here, change those too — tests/test_pose_math.py
pins the convention numerically.

Pure numpy — no ROS, no torch.
"""

from __future__ import annotations

import numpy as np


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotation matrix -> [..., 6]: first two rows flattened."""
    mat = np.asarray(mat)
    batch = mat.shape[:-2]
    return mat[..., :2, :].reshape(*batch, 6).copy()


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """[..., 6] rot6d -> [..., 3, 3] via Gram-Schmidt (pytorch3d/UMI)."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def pose9d_to_mat(pose: np.ndarray) -> np.ndarray:
    """[..., 9] (pos + rot6d) -> [..., 4, 4] homogeneous matrices."""
    pose = np.asarray(pose, dtype=np.float64)
    batch = pose.shape[:-1]
    T = np.tile(np.eye(4), (*batch, 1, 1))
    T[..., :3, :3] = rot6d_to_mat(pose[..., 3:9])
    T[..., :3, 3] = pose[..., :3]
    return T


def mat_to_pose9d(T: np.ndarray) -> np.ndarray:
    """[..., 4, 4] -> [..., 9] (pos + rot6d)."""
    T = np.asarray(T)
    return np.concatenate([T[..., :3, 3], mat_to_rot6d(T[..., :3, :3])], axis=-1)
