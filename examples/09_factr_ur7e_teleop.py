"""FACTR leader arm → UR7e + Robotiq 2F-140 joint teleoperation.

Control flow:
  FACTR arm publishes absolute joint positions → this script computes deltas
  → ManipulatorJointEnv (JIC) tracks them on the UR7e.
  FACTR gripper trigger (0=open, 1=closed) → Robotiq 2F-140 (absolute_continuous).
  UR7e joint effort is on /joint_states — FACTR subscribes to that directly for
  force feedback (no crisp_gym change needed).

Prerequisites:
  1. UR7e bringup running:
       docker compose up launch_ur_gripper
  2. FACTR teleop node running on the same ROS network, publishing:
       /factr_teleop/{FACTR_NAME}/joint_states   (sensor_msgs/JointState, 6 joints)
       /factr_teleop/{FACTR_NAME}/cmd_gripper_pos (std_msgs/Float32, 0..1)

Usage:
  python3 examples/09_factr_ur7e_teleop.py
  python3 examples/09_factr_ur7e_teleop.py --factr-name left --freq 50
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
parser = argparse.ArgumentParser(description="FACTR → UR7e + Robotiq joint teleop")
parser.add_argument("--factr-name", type=str, default="right",
                    help="FACTR arm name used in topic prefix (default: right)")
parser.add_argument("--freq", type=float, default=50.0,
                    help="Control loop frequency in Hz (default: 50)")
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

# ── UR7e environment ──────────────────────────────────────────────────────────
logger.info("Setting up UR7e + Robotiq environment (JIC)...")
env_config = make_env_config("ur7e_robotiq", control_frequency=args.freq)
env = ManipulatorJointEnv(namespace="", config=env_config)
env.wait_until_ready()

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

        # Action: [dtheta_1..dtheta_6, gripper_normalized]
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
