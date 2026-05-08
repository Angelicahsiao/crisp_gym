#!/usr/bin/env python3
"""Connection smoke test for the ``dg3f_ur`` env.

Verifies independently that:

1. The UR robot publishes joint state and end-effector pose on the topics
   declared by :class:`crisp_py.robot.robot_config.URConfig`.
2. The Tesollo Delto DG3F driver publishes 12 joints on
   ``gripper/joint_states`` (or whatever ``joint_state_topic`` is in
   ``crisp_gym/config/grippers/gripper_dg3f.yaml``).

This script does NOT command the robot or the gripper - it only listens. Run
it after starting the UR ROS2 driver and the ``delto_3f_driver`` bringup.

Usage::

    python scripts/test_dg3f_ur_connection.py
    python scripts/test_dg3f_ur_connection.py --namespace ur5e --timeout 15
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import rclpy

from crisp_py.gripper import MultiDofGripper, MultiDofGripperConfig
from crisp_py.robot import Robot

from crisp_gym.envs.manipulator_env_config import make_env_config


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _check_robot(cfg, namespace: str, timeout: float) -> list[str]:
    failures: list[str] = []
    _section("Robot (UR)")
    print(f"robot_config:    {type(cfg.robot_config).__name__}")
    print(f"  num_joints:    {cfg.robot_config.num_joints()}")
    print(f"  joint_names:   {cfg.robot_config.joint_names}")
    print(f"  base_frame:    {cfg.robot_config.base_frame}")
    print(f"  target_frame:  {cfg.robot_config.target_frame}")
    print(f"  joint topic:   {cfg.robot_config.current_joint_topic}")
    print(f"  pose topic:    {cfg.robot_config.current_pose_topic}")

    try:
        robot = Robot(namespace=namespace, robot_config=cfg.robot_config)
        robot.wait_until_ready(timeout=timeout)
    except Exception as exc:
        failures.append(f"Robot.wait_until_ready failed: {exc}")
        traceback.print_exc()
        return failures

    try:
        joint_values = np.asarray(robot.joint_values)
        print(f"  joint_values:  shape={joint_values.shape}  {np.array2string(joint_values, precision=3)}")
        if joint_values.shape != (cfg.robot_config.num_joints(),):
            failures.append(
                f"joint_values shape {joint_values.shape} != expected {(cfg.robot_config.num_joints(),)}"
            )
    except Exception as exc:
        failures.append(f"Robot joint_values read failed: {exc}")

    try:
        pose = robot.end_effector_pose
        print(f"  ee position:   {np.array2string(np.asarray(pose.position), precision=3)}")
    except Exception as exc:
        failures.append(f"Robot end_effector_pose read failed: {exc}")

    return failures


def _check_gripper(cfg, namespace: str, timeout: float) -> list[str]:
    failures: list[str] = []
    _section("Gripper (Delto DG3F)")
    if not isinstance(cfg.gripper_config, MultiDofGripperConfig):
        failures.append(
            f"Expected MultiDofGripperConfig, got {type(cfg.gripper_config).__name__}."
        )
        return failures

    g_cfg = cfg.gripper_config
    print(f"gripper_config:  {type(g_cfg).__name__}")
    print(f"  num_joints:    {g_cfg.num_joints}")
    print(f"  state topic:   {g_cfg.joint_state_topic}")
    print(f"  cmd topic:     {g_cfg.command_topic}")
    print(f"  joint_indices: {g_cfg.joint_indices}")

    try:
        gripper = MultiDofGripper(namespace=namespace, gripper_config=g_cfg)
        gripper.wait_until_ready(timeout=timeout)
    except Exception as exc:
        failures.append(f"MultiDofGripper.wait_until_ready failed: {exc}")
        traceback.print_exc()
        return failures

    try:
        value = gripper.value
        print(f"  value:         shape={value.shape}  {np.array2string(value, precision=3)}")
        expected_shape = (g_cfg.num_joints,)
        if value.shape != expected_shape:
            failures.append(
                f"gripper.value shape {value.shape} != expected {expected_shape}"
            )
    except Exception as exc:
        failures.append(f"Gripper value read failed: {exc}")

    try:
        raw = gripper.raw_value
        if raw is not None:
            print(f"  raw_value:     {np.array2string(raw, precision=3)}")
    except Exception as exc:
        failures.append(f"Gripper raw_value read failed: {exc}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default="",
        help="ROS2 namespace prefix for the robot and gripper topics.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-component readiness timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "--env-type",
        default="dg3f_ur",
        help="Env type to load (default: dg3f_ur).",
    )
    args = parser.parse_args()

    if not rclpy.ok():
        rclpy.init()

    failures: list[str] = []
    try:
        _section(f"Loading env config: {args.env_type}")
        cfg = make_env_config(args.env_type)
        print(f"control_frequency: {cfg.control_frequency} Hz")
        print(f"gripper_action_dim: {cfg.gripper_action_dim}")

        failures.extend(_check_robot(cfg, args.namespace, args.timeout))
        failures.extend(_check_gripper(cfg, args.namespace, args.timeout))
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: UR robot and DG3F gripper are reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
