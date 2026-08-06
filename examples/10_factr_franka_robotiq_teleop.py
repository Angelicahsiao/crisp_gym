"""FACTR leader arm → Franka FR3 + Robotiq 2F-85 joint teleoperation.

Franka analogue of 09_factr_ur7e_teleop.py, with two differences: the FR3 has 7
joints (the UR7e has 6), and this script commands ABSOLUTE joint positions
while 09 still integrates per-step deltas.

Control flow:
  FACTR arm publishes absolute joint positions → passed straight through as the
  controller's q_ref → ManipulatorJointEnv (JIC) tracks them on the FR3.
  FACTR gripper trigger (0=open, 1=closed) → Robotiq 2F-85 (absolute_continuous).
  FR3 joint effort is on /joint_states — FACTR subscribes to that directly for
  force feedback (no crisp_gym change needed).

Why absolute instead of deltas (config/envs/factr_franka_robotiq.yaml sets
`use_relative_actions: false`): the delta scheme seeded the commanded target
from the follower exactly once and then only ever added leader deltas, so any
leader/follower offset at startup was frozen in for the whole episode, and a
blocked follower let the target integrate away unbounded and then lunge when
released. Commanding the leader's absolute pose removes both — but it makes the
leader authoritative, so the arms MUST be aligned before the first step. The
env raises if they are not (`max_startup_joint_offset`), and `max_joint_speed`
clamps the per-step target change.

This requires the FACTR leader to be a joint-for-joint replica of the FR3 (same
joint order, zeros and directions). If yours has a fixed calibration offset,
add it to the leader positions before stepping.

Prerequisites:
  1. Franka + Robotiq bringup running (arm on JIC, Robotiq 2F-85 controller on
     the /robotiq_2f85 namespace publishing /robotiq_2f85/joint_states and
     accepting GripperCommand goals on
     /robotiq_2f85/robotiq_gripper_controller/gripper_cmd).
  2. FACTR teleop node running on the same ROS network, publishing:
       /factr_teleop/{FACTR_NAME}/cmd_arm_pos     (sensor_msgs/JointState, 7 joints)
       /factr_teleop/{FACTR_NAME}/cmd_gripper_pos (sensor_msgs/JointState, position[0] 0..1)
     and subscribing to:
       /factr_teleop/{FACTR_NAME}/follow_mode     (std_msgs/Bool)

Follow mode (press ENTER to toggle, 'q' + ENTER to quit). The script STARTS
SUSPENDED (follow mode ON) so you can align the two arms before the FR3 is ever
commanded — press ENTER once they match to engage teleoperation:
  ON  — the FR3 STOPS following the leader. The loop skips env.step(), so
        robot.target_joint stays latched at its last value and keeps being
        published: the arm holds where it is. crisp_gym publishes
        follow_mode=true so the FACTR node can track the arm instead, letting
        you reposition the leader without moving the robot.
  OFF — teleoperation resumes. Because the leader may have moved independently
        while suspended, the env's startup offset check is re-armed, so a
        misalignment raises on the next step instead of the arm driving across
        the gap.

Usage:
  python3 examples/10_factr_franka_robotiq_teleop.py
  python3 examples/10_factr_franka_robotiq_teleop.py --factr-name left --freq 30
"""

import argparse
import logging
import sys
import threading
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
# The FACTR node stops publishing while it follows, and boots into follow mode,
# so ask it to leave before waiting for the stream — otherwise the wait below
# times out on a stream that never starts. The script re-enters follow mode
# once the env is up (see the suspended start further down).
if not factr.wait_for_follow_mode_subscriber():
    logger.warning(
        "No subscriber on the FACTR follow_mode topic after 5s — cannot ask "
        "the leader to leave follow mode. If it boots following, it will not "
        "publish and the readiness wait below will time out."
    )
factr.set_follow_mode(False)
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

# ── follow-mode keyboard toggle ───────────────────────────────────────────────
# follow_mode ON  → the FR3 STOPS following the leader (its target is left
#                   latched, so the arm holds) and the FACTR node is asked to
#                   track the arm instead. Use it to reposition the leader.
# follow_mode OFF → normal teleoperation resumes.
#
# The listener thread only raises a request; the main loop applies it. Keeping
# every ROS/env call on one thread avoids racing the control loop.
_toggle_requested = threading.Event()
_quit_requested = threading.Event()


def _keyboard_listener() -> None:
    """Read stdin lines: 'q' quits, anything else toggles follow mode."""
    for line in sys.stdin:
        if line.strip().lower() == "q":
            _quit_requested.set()
            return
        _toggle_requested.set()


threading.Thread(target=_keyboard_listener, daemon=True).start()

# Start SUSPENDED: the FR3 holds and the leader tracks it, so the operator can
# align the two arms before the FR3 is ever commanded. Engaging then runs the
# startup offset check at a moment the operator chose, instead of at process
# start with whatever offset happened to exist. Also symmetric with the exit
# state below, so back-to-back runs do not flip the leader's mode needlessly.
follow_mode = True
factr.set_follow_mode(follow_mode)

logger.info(
    "Environment ready — starting SUSPENDED (follow mode ON): the FR3 is "
    "holding and is NOT following the leader.\n"
    "  ENTER — toggle follow mode; press it once the arms are aligned to "
    "engage teleoperation\n"
    "  q + ENTER, or Ctrl+C — stop"
)

# ── teleoperation loop ────────────────────────────────────────────────────────
dt = 1.0 / args.freq

try:
    while not _quit_requested.is_set():
        t_start = time.monotonic()

        if _toggle_requested.is_set():
            _toggle_requested.clear()
            follow_mode = not follow_mode
            factr.set_follow_mode(follow_mode)
            if not follow_mode:
                # FACTR was silent while following, so the cached pose predates
                # the leader's tracking motion — commanding it would send the
                # arm back there. Wait for the stream to actually resume.
                if not factr.wait_for_new_data():
                    logger.warning(
                        "FACTR did not resume publishing within 5s — the next "
                        "command may use a stale leader pose."
                    )
                # Commanding resumes after a gap in which the leader moved
                # independently: re-check alignment on the next step rather
                # than driving the arm across whatever offset opened up.
                env.require_startup_check()
            logger.info(
                "Follow mode %s — FR3 %s",
                "ON" if follow_mode else "OFF",
                "holding, not following the leader" if follow_mode else "following the leader",
            )

        if follow_mode:
            # Do not step: robot.target_joint stays latched at its last value
            # and keeps being published, so the arm holds where it is.
            elapsed = time.monotonic() - t_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
            continue

        current_joint_pos = factr.last_joint_pos
        current_gripper = factr.last_gripper

        # Action: [theta_1..theta_7, gripper_normalized] — the leader's ABSOLUTE
        # joint positions, passed straight through as the controller's q_ref
        # (the env config sets use_relative_actions: false).
        # gripper is absolute [0=open, 1=closed], mode=absolute_continuous.
        action = np.append(current_joint_pos, current_gripper).astype(np.float32)

        obs, _, terminated, truncated, _ = env.step(action, block=False)

        if terminated or truncated:
            logger.warning("Environment terminated/truncated. Resetting...")
            obs, _ = env.reset()

        logger.debug(
            f"joints: {np.round(current_joint_pos, 3)}  "
            f"tracking error: "
            f"{np.round(current_joint_pos - env.robot.joint_values, 4)}  "
            f"gripper: {current_gripper:.3f}"
        )

        elapsed = time.monotonic() - t_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    logger.info("Teleoperation stopped.")
finally:
    # Leave the leader in follow mode: the arm is no longer being commanded, so
    # this is the safe resting state (same convention as the recording script).
    factr.set_follow_mode(True)
    env.close()
    logger.info("Environment closed.")
