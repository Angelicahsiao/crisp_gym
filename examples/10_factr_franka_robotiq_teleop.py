"""FACTR leader arm → Franka FR3 + Robotiq 2F-85 joint teleoperation.

Franka analogue of 09_factr_ur7e_teleop.py. The only substantive difference is
the arm: the FR3 has 7 joints (the UR7e has 6), so the FACTR leader must publish
7 joint positions and the action vector is [dtheta_1..dtheta_7, gripper].

Control flow:
  FACTR arm publishes absolute joint positions → this script computes deltas
  → ManipulatorJointEnv (JIC) tracks them on the FR3.
  FACTR gripper trigger (0=open, 1=closed) → Robotiq 2F-85 (absolute_continuous).
  FR3 joint effort is on /joint_states — FACTR subscribes to that directly for
  force feedback (no crisp_gym change needed).

Prerequisites:
  1. Franka + Robotiq bringup running (arm on JIC, Robotiq 2F-85 controller on
     the /robotiq_2f85 namespace publishing /robotiq_2f85/joint_states and
     accepting GripperCommand goals on
     /robotiq_2f85/robotiq_gripper_controller/gripper_cmd).
  2. FACTR teleop node running on the same ROS network, publishing:
       /factr_teleop/{FACTR_NAME}/cmd_ur_pos      (sensor_msgs/JointState, 7 joints)
       /factr_teleop/{FACTR_NAME}/cmd_gripper_pos (sensor_msgs/JointState, position[0] 0..1)

Usage:
  python3 examples/10_factr_franka_robotiq_teleop.py
  python3 examples/10_factr_franka_robotiq_teleop.py --factr-name left --freq 30
"""

import argparse
import logging
import time

import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorJointEnv
from crisp_gym.envs.manipulator_env_config import make_env_config
from crisp_gym.teleop.teleop_factr_stream import FACTRStreamedJoints
from crisp_gym.util.setup_logger import setup_logging

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FACTR → Franka FR3 + Robotiq joint teleop")
parser.add_argument("--factr-name", type=str, default="right",
                    help="FACTR arm name used in topic prefix (default: right)")
parser.add_argument("--freq", type=float, default=30.0,
                    help="Control loop frequency in Hz (default: 30)")
parser.add_argument("--env", type=str, default="factr_franka_robotiq",
                    help="Environment config name or YAML (default: factr_franka_robotiq)")
parser.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
args = parser.parse_args()

setup_logging(level=args.log_level)
logger = logging.getLogger(__name__)

# ── FACTR stream ──────────────────────────────────────────────────────────────
logger.info("Connecting to FACTR stream...")
factr = FACTRStreamedJoints(name=args.factr_name)
factr.wait_until_ready()

joint_pos = factr.last_joint_pos
gripper = factr.last_gripper
logger.info(f"FACTR ready — joints: {np.round(joint_pos, 3)}, gripper: {gripper:.3f}")

# ── Franka environment ────────────────────────────────────────────────────────
logger.info("Setting up Franka FR3 + Robotiq 2F-85 environment (JIC)...")
env_config = make_env_config(args.env, control_frequency=args.freq)
env = ManipulatorJointEnv(namespace="", config=env_config)
env.wait_until_ready()

# The FR3 has 7 joints. Fail fast on a leader/arm DOF mismatch rather than
# stepping the env with a wrong-length action (which would misalign every joint).
n_arm_joints = env_config.robot_config.num_joints()
if joint_pos.shape[0] != n_arm_joints:
    raise ValueError(
        f"FACTR publishes {joint_pos.shape[0]} joints but the arm has "
        f"{n_arm_joints}. Use a {n_arm_joints}-DOF FACTR leader (the FR3 needs 7), "
        f"or point --env at a matching robot config."
    )

obs, _ = env.reset()
logger.info("Environment ready. Starting teleoperation — Ctrl+C to stop.")

# ── teleoperation loop ────────────────────────────────────────────────────────
prev_joint_pos = factr.last_joint_pos
dt = 1.0 / args.freq

try:
    while True:
        t_start = time.monotonic()

        current_joint_pos = factr.last_joint_pos
        current_gripper = factr.last_gripper

        # Delta joints: how much the FACTR arm moved since last cycle.
        delta_joints = current_joint_pos - prev_joint_pos
        prev_joint_pos = current_joint_pos

        # Action: [dtheta_1..dtheta_7, gripper_normalized]
        # gripper is absolute [0=open, 1=closed], mode=absolute_continuous.
        action = np.append(delta_joints, current_gripper).astype(np.float32)

        obs, _, terminated, truncated, _ = env.step(action, block=False)

        if terminated or truncated:
            logger.warning("Environment terminated/truncated. Resetting...")
            obs, _ = env.reset()
            prev_joint_pos = factr.last_joint_pos

        logger.debug(
            f"joints: {np.round(current_joint_pos, 3)}  "
            f"delta: {np.round(delta_joints, 4)}  "
            f"gripper: {current_gripper:.3f}"
        )

        elapsed = time.monotonic() - t_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    logger.info("Teleoperation stopped.")
finally:
    env.close()
    logger.info("Environment closed.")
