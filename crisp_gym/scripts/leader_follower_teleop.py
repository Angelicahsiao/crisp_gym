"""Example on how to teleoperate a robot using another one."""

import argparse
import logging
import time
from pathlib import Path
import crisp_gym
import numpy as np
import yaml
from crisp_gym.envs.manipulator_env import make_env, ManipulatorCartesianEnv, ManipulatorJointEnv
from crisp_gym.envs.manipulator_env_config import FrankaEnvConfig
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.gripper import GripperConfig
from crisp_py.gripper import MultiDofGripperConfig
from crisp_py.gripper import make_gripper

import crisp_py

from crisp_gym.util.gripper_mode import GripperMode
print(crisp_py.__file__)
# Parse args:
parser = argparse.ArgumentParser(description="Teleoperation of a leader robot.")
parser.add_argument(
    "--use-force-feedback",
    action="store_true",
    help="Use force feedback from the leader robot (default: False)",
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level (default: INFO)",
)
parser.add_argument(
    "--control-frequency",
    type=float,
    default=100.0,
    help="Control frequency in Hz (default: 100.0)",
)
parser.add_argument(
    "--use-streamed-teleop",
    action="store_true",
    help="Use streamed pose instead of physical leader robot",
)

parser.add_argument(
    "--gripper-config",
    type=str,
    default=None,
    help="Path to gripper YAML config file (default: None = disable gripper)",
)

parser.add_argument(
    "--camera-config",
    type=str,
    default=None,
    help="Path to camera YAML config file (default: None = disable camera)",
)



args = parser.parse_args()

# Set up logging
setup_logging(level=args.log_level)
logger = logging.getLogger(__name__)

# %% Leader setup
logger.info("Setting up leader robot...")
if args.use_streamed_teleop:
    leader = TeleopStreamedPose()
    logger.info("Using streamed teleop as leader.")
else:
    leader = make_leader(name="left_aloha_franka", namespace="left")
    leader.wait_until_ready()
    leader.prepare_for_teleop()
    logger.info("Using physical leader robot.")

# %% Environment setup
logger.info("Setting up environment...")
# env = make_env("right_no_cam_franka", control_type="cartesian", namespace="right")
CTRL_FREQ = 50
BASE_DIR = Path(crisp_gym.__file__).parent
gripper_config_path = (
    str(BASE_DIR / args.gripper_config)
    if args.gripper_config is not None
    else None
)

camera_config = (
    str(BASE_DIR / args.camera_config)
    if args.camera_config is not None
    else None
)

# Load gripper config polymorphically: multi_dof YAMLs need MultiDofGripperConfig
if gripper_config_path is not None:
    with open(gripper_config_path, "r") as _f:
        _gripper_yaml = yaml.safe_load(_f) or {}
    if _gripper_yaml.get("type") == "multi_dof":
        loaded_gripper_config = MultiDofGripperConfig.from_yaml(gripper_config_path)
    else:
        loaded_gripper_config = GripperConfig.from_yaml(gripper_config_path)
else:
    loaded_gripper_config = None

env_config = FrankaEnvConfig(
    control_frequency=CTRL_FREQ,
    gripper_mode=GripperMode.ABSOLUTE_BINARY,
    gripper_config=loaded_gripper_config,
    camera_configs=[],
    gripper_threshold=0.1
)

print("Cartesian config:",
      env_config.cartesian_control_param_config.resolve())

print("Joint config:",
      env_config.joint_control_param_config.resolve())
print(env_config.gripper_mode)
print(env_config.gripper_config)
if env_config.gripper_config is not None:
    print(f"Gripper command action: {env_config.gripper_config.use_gripper_command_action}")
env_config.cartesian_control_param_config = str(
    BASE_DIR / "config/control/default_cartesian_impedance.yaml"
)

env_config.joint_control_param_config = str(
    BASE_DIR / "config/control/joint_control.yaml"
)
print("Cartesian config:",
      env_config.cartesian_control_param_config)

print("Joint config:",
      env_config.joint_control_param_config)

env = ManipulatorCartesianEnv(namespace="", config=env_config)
env.robot.home()
env.reset()

# %% Now run the teleoperation loop
logger.info(":rocket: Starting teleoperation...")

if not args.use_streamed_teleop:
    if args.use_force_feedback:
        leader.robot.controller_switcher_client.switch_controller("torque_feedback_controller")
    else:
        leader.robot.cartesian_controller_parameters_client.load_param_config(
            file_path=leader.config.gravity_compensation_controller
        )
        leader.robot.controller_switcher_client.switch_controller(
            "cartesian_impedance_controller"
        )


previous_pose = (
    leader.robot.end_effector_pose
    if not args.use_streamed_teleop
    else leader.last_pose
)

# Number of gripper action dims determined by config (1 for scalar, N for multi-DOF)
_n_grip = env_config.gripper_action_dim

while True:
    current_pose = (
        leader.robot.end_effector_pose
        if not args.use_streamed_teleop
        else leader.last_pose
    )

    action_pose = current_pose - previous_pose
    previous_pose = current_pose

    gripper_scalar = float(np.asarray(
        leader.gripper.value
        if not args.use_streamed_teleop and leader.gripper
        else leader.last_gripper if args.use_streamed_teleop else 0.0
    ).reshape(-1)[0])

    # Broadcast scalar leader value to all gripper joints.
    # For 1-DOF grippers this is a no-op; for multi-DOF it fans out the
    # single open/close signal to every joint uniformly.
    gripper_arr = np.full(_n_grip, gripper_scalar, dtype=np.float32)

    action = np.concatenate(
        [
            action_pose.position,
            action_pose.orientation.as_euler("xyz"),
            gripper_arr,
        ]
    )
    obs, *_ = env.step(action, block=False)
    time.sleep(1.0 / args.control_frequency)
