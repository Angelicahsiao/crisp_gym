"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from crisp_gym.util.control_type import ControlType
from crisp_gym.util.gripper_mode import GripperMode

if TYPE_CHECKING:
    from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, ManipulatorCartesianEnv
    from crisp_gym.teleop.teleop_robot import TeleopRobot
    from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose


logger = logging.getLogger(__name__)


def _leader_gripper_to_action(
    leader_value: float,
    follower_value: float,
    control_mode: GripperMode | str,
) -> float:
    """Convert the leader gripper value to an action for the follower gripper.

    Args:
        leader_value (float): The current value of the leader gripper.
        follower_value (float): The current value of the follower gripper.
        control_mode (GripperMode): The control mode of the gripper.

    Returns:
        float: The computed gripper action for the follower.
    """
    if isinstance(control_mode, str):
        control_mode = GripperMode(control_mode)

    if control_mode in [GripperMode.ABSOLUTE_BINARY, GripperMode.ABSOLUTE_CONTINUOUS]:
        return leader_value
    elif control_mode in [GripperMode.RELATIVE_BINARY, GripperMode.RELATIVE_CONTINUOUS]:
        return leader_value - follower_value
    elif control_mode == GripperMode.NONE:
        return 0.0
    else:
        raise ValueError(f"Unsupported gripper control mode: {control_mode}")


def _drive_fn_to_teleop_fn(env, drive_fn: Callable) -> Callable:
    """Wrap a drive-fn (command computation only) into a legacy teleop fn that
    also steps the env and returns (obs, action) for RecordingManager."""

    def _fn() -> tuple:
        action = drive_fn()
        if action is None:  # warm-up tick (no previous pose to diff against)
            return None, None
        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def make_teleop_streamer_fn(env: ManipulatorCartesianEnv, leader: TeleopStreamedPose) -> Callable:
    """Legacy streamed-pose teleop fn — thin wrapper over make_streamer_drive_fn."""
    return _drive_fn_to_teleop_fn(env, make_streamer_drive_fn(env, leader))


def make_record_fn(
    env,
    record_config,
    drive_fn: Callable | None = None,
) -> Callable:
    """Generic, config-driven recording function.

    Decouples WHAT is recorded (record_config: observation sources, action
    definition) from HOW the robot is driven (drive_fn: teleop deltas, FACTR
    joints, nothing for a handheld device).

    Args:
        env: Any env exposing the sources named in the config (robot envs,
            UmiHandheldEnv, ...). env.step(action) is called every tick.
        record_config: A RecordConfig (see record/record_config.py).
        drive_fn: Optional zero-arg callable returning the command vector to
            send to env.step() this tick (or None). It must NOT step the env
            itself. For passive recording (handheld), leave None.

    Returns:
        Callable returning (obs, action) per tick — or (None, None) while the
        lookahead buffer fills — for RecordingManager.record_episode.
    """
    from collections import deque

    from crisp_gym.record.record_config import SOURCE_REGISTRY

    act_cfg = record_config.action
    lookahead = act_cfg.lookahead
    obs_buffer: deque = deque(maxlen=lookahead + 1)

    def _resize_image(img: np.ndarray, shape) -> np.ndarray:
        """Resize an (H, W, C) image to the contract's declared (H, W, C).

        The camera produces images at the env config's `resolution` target,
        which need not equal the record contract's declared image shape. The
        old record path cropped/resized the image afterward; do the same here
        so a resolution mismatch never raises a LeRobot shape error.
        """
        if img is None or shape is None or len(shape) != 3:
            return img
        target_h, target_w = int(shape[0]), int(shape[1])
        if img.shape[0] == target_h and img.shape[1] == target_w:
            return img
        import cv2

        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _collect_obs() -> dict:
        obs = {"task": getattr(env, "task", "")}
        for o in record_config.observations:
            val = SOURCE_REGISTRY[o.source](env, **o.params)
            if o.key.startswith("observation.images.") and o.shape is not None:
                val = _resize_image(val, o.resolved_shape())
            obs[o.key] = val
        return obs

    def _action_value() -> np.ndarray:
        if act_cfg.definition == "next_tcp_pose":
            pose = SOURCE_REGISTRY["robot.tcp_pose"](
                env, representation=act_cfg.representation
            )
            parts = [pose]
        elif act_cfg.definition == "next_joint_positions":
            parts = [SOURCE_REGISTRY["robot.joint_positions"](env)]
        else:
            raise RuntimeError("command actions are handled inline")
        if act_cfg.include_gripper:
            parts.append(
                SOURCE_REGISTRY["gripper.width_normalized"](env, **act_cfg.gripper_params)
            )
        return np.concatenate(parts).astype(np.float32)

    def _fn() -> tuple:
        cmd = drive_fn() if drive_fn is not None else None
        if drive_fn is not None and cmd is None:
            # Drive warm-up tick (e.g. teleop needs a previous pose to diff).
            return None, None
        env.step(cmd, block=False)

        obs = _collect_obs()

        if act_cfg.definition == "command":
            if cmd is None:
                raise ValueError(
                    "action.definition 'command' requires a drive_fn "
                    "returning the command vector."
                )
            return obs, np.asarray(cmd, dtype=np.float32)

        # next_*: pair obs[t-lookahead] with the value measured NOW.
        obs_buffer.append(obs)
        if len(obs_buffer) <= lookahead:
            return None, None
        return obs_buffer[0], _action_value()

    return _fn


def make_teleop_drive_fn(env: ManipulatorBaseEnv, leader: TeleopRobot) -> Callable:
    """Drive-fn adapter for make_record_fn: computes the teleop command each
    tick (same semantics as make_teleop_fn) WITHOUT stepping the env.

    Returns None on the first tick (no previous pose to diff against).
    """
    state = {"prev_pose": None, "prev_joint": None}

    def _drive():
        pose = leader.robot.end_effector_pose
        joint = leader.robot.joint_values

        if state["prev_pose"] is None:
            state["prev_pose"], state["prev_joint"] = pose, joint
            return None

        if env.config.use_relative_actions:
            action_pose = pose - state["prev_pose"]
            action_joint = joint - state["prev_joint"]
        else:
            action_pose, action_joint = pose, joint
        state["prev_pose"], state["prev_joint"] = pose, joint

        gripper_action = _leader_gripper_to_action(
            leader_value=leader.gripper.value if leader.gripper is not None else 0.0,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )

        if env.ctrl_type is ControlType.CARTESIAN:
            vec = action_pose.to_array(env.config.orientation_representation)
        elif env.ctrl_type is ControlType.JOINT:
            vec = action_joint
        else:
            raise ValueError(f"Unsupported control type: {env.ctrl_type}")
        return np.concatenate([vec, [gripper_action]])

    return _drive


def make_streamer_drive_fn(
    env: "ManipulatorCartesianEnv", leader: "TeleopStreamedPose"
) -> Callable:
    """Drive-fn adapter for streamed-pose teleop (phone/VR): delta pose command."""
    state = {"prev_pose": None}

    def _drive():
        pose = leader.last_pose
        if state["prev_pose"] is None:
            state["prev_pose"] = pose
            return None
        action_pose = pose - state["prev_pose"]
        state["prev_pose"] = pose
        gripper = leader.last_gripper if leader.last_gripper is not None else 0.0
        vec = action_pose.to_array(env.config.orientation_representation)
        return np.concatenate([vec, [gripper]])

    return _drive


def make_factr_drive_fn(factr) -> Callable:
    """Drive-fn adapter for a FACTR leader arm (joint-space teleop).

    Computes delta-joint commands from the FACTR stream's absolute joint
    positions (same logic as examples/09_factr_ur7e_teleop.py) WITHOUT
    stepping the env. Returns None on the warm-up tick. The command layout is
    [dtheta_1..dtheta_N, gripper] with the gripper trigger already inverted
    and clamped to [0, 1] by FACTRStreamedJoints.

    Args:
        factr: A crisp_gym.teleop.teleop_factr_stream.FACTRStreamedJoints.
    """
    state = {"prev": None}

    def _drive():
        current = factr.last_joint_pos
        if state["prev"] is None:
            state["prev"] = current
            return None
        delta = current - state["prev"]
        state["prev"] = current
        gripper = factr.last_gripper if factr.last_gripper is not None else 0.0
        return np.append(delta, gripper).astype(np.float32)

    return _drive


def make_teleop_fn(env: ManipulatorBaseEnv, leader: TeleopRobot) -> Callable:
    """Legacy leader-follower teleop fn — thin wrapper over make_teleop_drive_fn.

    Computes the delta (or absolute) command from the leader, steps the env,
    and returns (obs, action) for RecordingManager. The command math lives in
    make_teleop_drive_fn (single source).
    """
    return _drive_fn_to_teleop_fn(env, make_teleop_drive_fn(env, leader))
