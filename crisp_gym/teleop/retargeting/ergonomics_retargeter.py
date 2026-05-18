"""Manus ergonomics (joint angles in degrees) → multi-DOF gripper targets.

Joint-space, rule-based retargeting. Mirrors the approach used by
tesollodelto/delto_m_ros2 manus_retarget.py, generalized for an arbitrary
multi-DOF gripper (DG3F by default).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from crisp_gym.teleop.retargeting.base import GloveRetargeter

# Manus ergonomics field suffix per joint within a finger:
#   {Finger}MCPSpread  - lateral spread at MCP
#   {Finger}MCPStretch - flexion at MCP
#   {Finger}PIPStretch - flexion at PIP
#   {Finger}DIPStretch - flexion at DIP
_JOINT_SUFFIXES_DEFAULT = ["MCPSpread", "MCPStretch", "PIPStretch", "DIPStretch"]


class ErgonomicsRetargeter(GloveRetargeter):
    """Read Manus glove ergonomics and emit normalized gripper joint targets.

    Each gripper finger is mapped to a Manus finger via ``finger_mapping``.
    Per-finger ``scale`` and ``direction`` adjust the raw degrees; output is
    converted to radians, clamped to the gripper's hardware limits, and
    normalized to [0, 1] for ``MultiDofGripper.set_target``.
    """

    def __init__(
        self,
        finger_mapping: Dict[str, str],
        scale: Dict[str, List[float]],
        direction: Dict[str, List[int]],
        min_values: List[float],
        max_values: List[float],
        joint_suffixes: Optional[List[str]] = None,
        offset_deg: float = 0.0,
    ):
        self._finger_mapping = finger_mapping
        self._suffixes = joint_suffixes or _JOINT_SUFFIXES_DEFAULT
        self._offset_deg = float(offset_deg)
        self._min = np.asarray(min_values, dtype=np.float32)
        self._max = np.asarray(max_values, dtype=np.float32)
        self._num_joints = int(self._min.shape[0])

        if self._max.shape != self._min.shape:
            raise ValueError("min_values and max_values must have the same length")
        if len(self._suffixes) == 0:
            raise ValueError("joint_suffixes must be non-empty")

        expected = len(finger_mapping) * len(self._suffixes)
        if expected != self._num_joints:
            raise ValueError(
                f"finger_mapping ({len(finger_mapping)}) × joint_suffixes "
                f"({len(self._suffixes)}) = {expected}, but gripper has "
                f"{self._num_joints} joints"
            )

        # Cache field names and per-joint scale/direction in gripper-joint order
        # (finger-major: F1J1, F1J2, ..., F2J1, ...).
        field_names: List[str] = []
        scales: List[float] = []
        dirs: List[int] = []
        for gripper_finger, manus_finger in finger_mapping.items():
            if gripper_finger not in scale or gripper_finger not in direction:
                raise KeyError(
                    f"finger '{gripper_finger}' missing from scale or direction"
                )
            if len(scale[gripper_finger]) != len(self._suffixes):
                raise ValueError(
                    f"scale['{gripper_finger}'] length {len(scale[gripper_finger])} "
                    f"!= joint_suffixes length {len(self._suffixes)}"
                )
            if len(direction[gripper_finger]) != len(self._suffixes):
                raise ValueError(
                    f"direction['{gripper_finger}'] length {len(direction[gripper_finger])} "
                    f"!= joint_suffixes length {len(self._suffixes)}"
                )
            for i, suffix in enumerate(self._suffixes):
                field_names.append(f"{manus_finger}{suffix}")
                scales.append(scale[gripper_finger][i])
                dirs.append(direction[gripper_finger][i])

        self._field_names = field_names
        self._scales = np.asarray(scales, dtype=np.float32)
        self._dirs = np.asarray(dirs, dtype=np.float32)

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        min_values: List[float],
        max_values: List[float],
    ) -> "ErgonomicsRetargeter":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(
            finger_mapping=cfg["finger_mapping"],
            scale=cfg["scale"],
            direction=cfg["direction"],
            min_values=min_values,
            max_values=max_values,
            joint_suffixes=cfg.get("joint_suffixes"),
            offset_deg=cfg.get("offset_deg", 0.0),
        )

    @property
    def topic_type(self) -> type:
        # Lazy import: manus_ros2_msgs is not always installed.
        from manus_ros2_msgs.msg import ManusGlove

        return ManusGlove

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def field_names(self) -> List[str]:
        return list(self._field_names)

    def retarget(self, msg: Any) -> np.ndarray:
        ergonomics = {e.type: float(e.value) for e in msg.ergonomics}
        raw_deg = np.asarray(
            [ergonomics.get(name, 0.0) for name in self._field_names],
            dtype=np.float32,
        )
        # Scale + direction in degree space, then convert to radians.
        rad = np.deg2rad((raw_deg + self._offset_deg) * self._dirs * self._scales)
        rad = np.clip(rad, self._min, self._max)
        span = self._max - self._min
        # Guard against zero-span axes.
        span = np.where(span == 0, 1.0, span)
        norm = (rad - self._min) / span
        return norm.astype(np.float32)
