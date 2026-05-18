"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from crisp_gym.util.control_type import ControlType
from crisp_gym.util.gripper_mode import GripperMode

if TYPE_CHECKING:
    from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, ManipulatorCartesianEnv
    from crisp_gym.teleop.teleop_manus_glove import ManusGloveTeleop
    from crisp_gym.teleop.teleop_robot import TeleopRobot
    from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose


logger = logging.getLogger(__name__)

GripperValue = Union[float, np.ndarray]


def _leader_gripper_to_action(
    leader_value: GripperValue,
    follower_value: GripperValue,
    control_mode: GripperMode | str,
) -> GripperValue:
    """Convert the leader gripper value to an action for the follower gripper.

    Works element-wise for both scalar (1-DOF) and ``np.ndarray`` (multi-DOF)
    inputs. Output shape matches the leader's shape; ``GripperMode.NONE``
    returns zeros of the same shape.

    Args:
        leader_value: Current value of the leader gripper (scalar or array).
        follower_value: Current value of the follower gripper (same shape).
        control_mode: Gripper control mode.

    Returns:
        Gripper action matching the leader's shape.
    """
    if isinstance(control_mode, str):
        control_mode = GripperMode(control_mode)

    if control_mode in [GripperMode.ABSOLUTE_BINARY, GripperMode.ABSOLUTE_CONTINUOUS]:
        return leader_value
    elif control_mode in [GripperMode.RELATIVE_BINARY, GripperMode.RELATIVE_CONTINUOUS]:
        return leader_value - follower_value
    elif control_mode == GripperMode.NONE:
        if isinstance(leader_value, np.ndarray):
            return np.zeros_like(leader_value)
        return 0.0
    else:
        raise ValueError(f"Unsupported gripper control mode: {control_mode}")


def _fit_gripper_action_dim(gripper_action: GripperValue, expected_dim: int) -> np.ndarray:
    """Reshape a gripper action to match the env's expected gripper action dim.

    - Scalar or ``(1,)`` input + ``expected_dim > 1``: broadcast (same value for
      every joint; e.g. a phone open/close slider applied to all 12 DG3F joints).
    - ``(expected_dim,)`` input: passed through as float32.
    - Anything else: raises ``ValueError``.
    """
    arr = np.atleast_1d(np.asarray(gripper_action, dtype=np.float32))
    if arr.shape[0] == expected_dim:
        return arr
    if arr.shape[0] == 1:
        return np.full(expected_dim, arr[0], dtype=np.float32)
    raise ValueError(
        f"Gripper action has {arr.shape[0]} dim(s), env expects {expected_dim}."
    )


def make_teleop_streamer_fn(env: ManipulatorCartesianEnv, leader: TeleopStreamedPose) -> Callable:
    """Create a teleoperation function for the leader robot using streamed pose data."""
    prev_pose = leader.last_pose
    first_step = True

    def _fn() -> tuple:
        nonlocal prev_pose, first_step
        if first_step:
            first_step = False
            prev_pose = leader.last_pose
            return None, None

        pose = leader.last_pose
        action_pose = pose - prev_pose
        prev_pose = pose

        gripper = leader.last_gripper if leader.last_gripper is not None else 0.0

        action_pose_vector = action_pose.to_array(env.config.orientation_representation)
        gripper_arr = _fit_gripper_action_dim(gripper, env.config.gripper_action_dim)

        action = np.concatenate([action_pose_vector, gripper_arr])
        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def make_teleop_fn(env: ManipulatorBaseEnv, leader: TeleopRobot) -> Callable:
    """Create a teleoperation function for the leader robot.

    This function returns a Callable that can be used to control the leader robot
    in a teleoperation manner. It computes the action based on the difference
    between the current and previous end-effector pose or joint values, and
    updates the gripper value based on the leader gripper's value.

    Args:
        env (ManipulatorBaseEnv): The environment in which the leader robot operates.
        leader (TeleopRobot): The teleoperation leader robot instance.

    Returns:
        Callable: A function that, when called, performs a step in the environment
        and returns the observation and action taken.
    """
    prev_pose = leader.robot.end_effector_pose
    prev_joint = leader.robot.joint_values
    first_step = True

    def _fn() -> tuple:
        nonlocal prev_pose, prev_joint, first_step
        if first_step:
            first_step = False
            prev_pose = leader.robot.end_effector_pose
            prev_joint = leader.robot.joint_values
            return None, None

        pose = leader.robot.end_effector_pose
        joint = leader.robot.joint_values

        if env.config.use_relative_actions:
            action_pose = pose - prev_pose
            action_joint = joint - prev_joint
        else:
            action_pose = pose
            action_joint = joint

        prev_pose = pose
        prev_joint = joint

        gripper_action = _leader_gripper_to_action(
            leader_value=leader.gripper.value if leader.gripper is not None else 0.0,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )
        gripper_arr = _fit_gripper_action_dim(gripper_action, env.config.gripper_action_dim)

        action = None
        if env.ctrl_type is ControlType.CARTESIAN:
            action_pose_vector = action_pose.to_array(env.config.orientation_representation)
            action = np.concatenate([action_pose_vector, gripper_arr])
        elif env.ctrl_type is ControlType.JOINT:
            action = np.concatenate([action_joint, gripper_arr])
        else:
            raise ValueError(
                f"Unsupported control type: {env.ctrl_type}. "
                "Supported types are 'cartesian' and 'joint' for delta actions."
            )

        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def make_teleop_manus_fn(
    env: ManipulatorBaseEnv,
    glove: ManusGloveTeleop,
    arm_leader: Optional[Union[TeleopRobot, TeleopStreamedPose]] = None,
) -> Callable:
    """Create a teleoperation function that uses a Manus glove for gripper control.

    The gripper action is taken from the glove's retargeted joint targets and
    routed through the env's gripper mode (absolute / relative / binary).
    The arm action comes from ``arm_leader``:

    * ``TeleopRobot``        — Cartesian or joint delta from physical leader arm.
    * ``TeleopStreamedPose`` — Cartesian delta from phone / VR stream
                               (CARTESIAN control only).
    * ``None``               — arm stays fixed (zero action); gripper-only recording.

    Args:
        env: Follower manipulator environment.
        glove: Manus glove teleop instance exposing ``last_gripper_joints``.
        arm_leader: Optional arm pose / joint source.

    Returns:
        Callable that steps the env and returns ``(obs, action)``.
    """
    first_step = True
    prev_pose = None
    prev_joint = None
    arm_dim = env.action_space.shape[0] - env.config.gripper_action_dim

    def _fn() -> tuple:
        nonlocal first_step, prev_pose, prev_joint

        if first_step:
            first_step = False
            if arm_leader is not None:
                if hasattr(arm_leader, "robot"):  # TeleopRobot
                    prev_pose = arm_leader.robot.end_effector_pose
                    prev_joint = arm_leader.robot.joint_values
                else:  # TeleopStreamedPose
                    prev_pose = arm_leader.last_pose
            return None, None

        # --- arm ---
        if arm_leader is None:
            arm_vec = np.zeros(arm_dim, dtype=np.float32)
        elif hasattr(arm_leader, "robot"):  # TeleopRobot
            if env.ctrl_type is ControlType.CARTESIAN:
                pose = arm_leader.robot.end_effector_pose
                arm_vec = (pose - prev_pose).to_array(
                    env.config.orientation_representation
                ).astype(np.float32)
                prev_pose = pose
            else:  # JOINT
                joint = arm_leader.robot.joint_values
                arm_vec = (
                    (joint - prev_joint).astype(np.float32)
                    if env.config.use_relative_actions
                    else joint.astype(np.float32)
                )
                prev_joint = joint
        else:  # TeleopStreamedPose
            if env.ctrl_type is not ControlType.CARTESIAN:
                raise ValueError(
                    "TeleopStreamedPose arm source requires CARTESIAN control."
                )
            pose = arm_leader.last_pose
            arm_vec = (pose - prev_pose).to_array(
                env.config.orientation_representation
            ).astype(np.float32)
            prev_pose = pose

        # --- gripper ---
        gripper_raw = _leader_gripper_to_action(
            leader_value=glove.last_gripper_joints,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )
        gripper_arr = _fit_gripper_action_dim(gripper_raw, env.config.gripper_action_dim)

        action = np.concatenate([arm_vec, gripper_arr])
        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn
