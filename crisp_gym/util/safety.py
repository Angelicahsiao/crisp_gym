"""Limits applied to a Cartesian target before it is sent to the controller.

Two independent guards, both configured per env YAML:

  * the WORKSPACE BOX (`min_x` ... `max_z`): every commanded TCP position is
    clipped into it, in metres, in the robot base frame;
  * the STEP LIMIT (`max_target_step_m`, `max_target_step_deg`): a new target
    may move at most this far from the PREVIOUS TARGET in one `env.step`. At a
    15 Hz loop, 0.03 m and 5 deg per step are 0.45 m/s and 75 deg/s.

The step limit is measured against the previous target, not the measured pose,
on purpose: under impedance control the target runs ahead of the arm (that gap
is the contact force), and a limit on target-minus-measured would cap the very
quantity the command-action work records. A limit on target-to-target is a
limit on how fast the command itself may move.

Pure numpy + scipy, so it is testable without ROS.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def workspace_box(
    min_x: float | None = None,
    min_y: float | None = None,
    min_z: float | None = None,
    max_x: float | None = None,
    max_y: float | None = None,
    max_z: float | None = None,
) -> dict[str, np.ndarray]:
    """Lower and upper corners of the workspace box; None = unbounded.

    A bound of 0.0 is a real bound. (It used to be read with `value or inf`,
    which turns 0.0 into "no bound".)
    """

    def bound(value: float | None, unbounded: float) -> float:
        return unbounded if value is None else float(value)

    inf = float("inf")
    return {
        "lower": np.array([bound(min_x, -inf), bound(min_y, -inf), bound(min_z, -inf)]),
        "upper": np.array([bound(max_x, inf), bound(max_y, inf), bound(max_z, inf)]),
    }


def limit_target_step(
    prev_position: np.ndarray,
    prev_orientation: Rotation,
    position: np.ndarray,
    orientation: Rotation,
    max_step_m: float | None = None,
    max_step_deg: float | None = None,
) -> tuple[np.ndarray, Rotation, bool, bool]:
    """Shorten a target that moves too far from the previous one.

    Translation and rotation are limited independently. A step that is too
    long keeps its direction and is cut to the limit: for the translation along
    the straight line, for the rotation along the shortest rotation between
    the two orientations.

    Returns:
        (position, orientation, translation_was_cut, rotation_was_cut)
    """
    prev_position = np.asarray(prev_position, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    cut_translation = cut_rotation = False

    if max_step_m is not None:
        delta = position - prev_position
        length = float(np.linalg.norm(delta))
        if length > max_step_m:
            position = prev_position + delta * (max_step_m / length)
            cut_translation = True

    if max_step_deg is not None:
        delta_rot = (orientation * prev_orientation.inv()).as_rotvec()
        angle = float(np.linalg.norm(delta_rot))
        limit = np.radians(max_step_deg)
        if angle > limit:
            orientation = Rotation.from_rotvec(delta_rot * (limit / angle)) * prev_orientation
            cut_rotation = True

    return position, orientation, cut_translation, cut_rotation
