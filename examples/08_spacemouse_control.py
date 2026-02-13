"""Example: Control robot with SpaceMouse using streamed pose teleoperation.

This example demonstrates how to:
1. Start the SpaceMouse streamer (publishes to /phone_pose and /phone_gripper)
2. Receive streamed pose and gripper data via TeleopStreamedPose
3. Control a robot by sending relative pose commands

To use this:
1. In one terminal, start the SpaceMouse streamer:
   python3 crisp_gym/teleop/spacemouse_node.py

2. In another terminal, run this example:
   python3 examples/08_spacemouse_control.py
"""

import logging
import time

import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import FrankaEnvConfig

from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

CTRL_FREQ = 50  # control frequency in Hz
SPACEMOUSE_SCALE = 0.05  # Scale factor to amplify SpaceMouse input (try 0.01-0.1)

def control_robot_with_spacemouse(env_config: str = "default", namespace: str = ""):
    """Control a robot using SpaceMouse streamed input.
    
    Args:
        env_config: Name of the environment configuration (e.g., 'default', 'left_robot_env', etc.)
        namespace: ROS namespace for the streamed topics (should match spacemouse_node.py namespace)
    """
    # Initialize the environment
    logger.info("Initializing environment...")
    env = ManipulatorCartesianEnv(config=env_config)
    
    # Wait for the robot to be ready
    logger.info("Waiting for robot to be ready...")
    env.wait_until_ready()
    logger.info("✓ Robot ready!")
    
    # Move to a safe starting position before control
    start_position = np.array([0.4, 0.0, 0.4])
    logger.info(f"Moving to start position: {start_position}")
    env.move_to(position=start_position, speed=0.15)
    env.gripper.open()
    obs, _ = env.reset()
    logger.info("✓ Robot positioned and reset!")
    
    # Initialize the teleop streamer (subscribes to /phone_pose and /phone_gripper)
    logger.info("Initializing SpaceMouse streamer receiver...")
    teleop = TeleopStreamedPose(namespace=namespace)
    
    # Wait for the first data from the streamer
    logger.info("Waiting for SpaceMouse data... (you should have spacemouse_node.py running)")
    try:
        teleop.wait_until_ready(timeout=10.0)
    except TimeoutError:
        logger.error(
            "Timeout waiting for streamed pose. Make sure spacemouse_node.py is running:\n"
            f"  python3 crisp_gym/teleop/spacemouse_node.py --namespace={namespace}"
        )
        return
    
    logger.info("✓ SpaceMouse connected! Starting control loop...")
    
    # Debug: log which topics we're listening to
    logger.info(f"Listening to pose topic: {teleop._pose_topic}")
    logger.info(f"Listening to gripper topic: {teleop._gripper_topic}")
    logger.info(f"Environment orientation representation: {env.config.orientation_representation}")
    logger.info(f"SpaceMouse scale factor: {SPACEMOUSE_SCALE} (try 0.01-0.2 if robot doesn't move)")
    
    prev_pose = teleop.last_pose
    step_count = 0
    dt = 1.0 / CTRL_FREQ  # Control loop time step
    debug_count = 0
    pose_update_count = 0
    
    try:
        while True:
            # Get the current streamed pose
            current_pose = teleop.last_pose
            gripper_command = teleop.last_gripper
            
            # Compute incremental pose change
            delta_pose = current_pose - prev_pose
            prev_pose = current_pose
            
            # Convert pose to action (format depends on env.config.orientation_representation)
            action_pose_array = delta_pose.to_array(env.config.orientation_representation)
            
            # Scale up the SpaceMouse input to make robot movement noticeable
            action_pose_array = action_pose_array * SPACEMOUSE_SCALE
            
            # Combine pose action with gripper command
            action = np.concatenate([
                action_pose_array,
                [gripper_command],
            ])
            
            # Track if we got a new pose
            delta_pos_array = delta_pose.position
            if not np.allclose(delta_pos_array, [0, 0, 0], atol=1e-6):
                pose_update_count += 1
            
            # Execute the action in the environment (block=True ensures action is executed)
            obs, reward, terminated, truncated, info = env.step(action, block=True)
            
            step_count += 1
            debug_count += 1
            if debug_count == 50:
                debug_count = 0
                ee_pose = env.robot.end_effector_pose
                pos = ee_pose.position
                action_mag = np.linalg.norm(action[:-1])  # Magnitude excluding gripper
                logger.info(
                    f"Step {step_count:4d} | EE: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] | "
                    f"Action mag: {action_mag:.6f} | Updates: {pose_update_count}/50 | Gripper: {gripper_command:.1f}"
                )
                pose_update_count = 0
            
    except KeyboardInterrupt:
        logger.info("Stopping control loop...")
    finally:
        env.close()
        logger.info("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Control robot with SpaceMouse via TeleopStreamedPose"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="default",
        help="Environment configuration name (default: 'default')"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="ROS namespace for streamed topics (should match spacemouse_node.py --namespace)"
    )
    
    args = parser.parse_args()
    env_config = FrankaEnvConfig(control_frequency=CTRL_FREQ, gripper_config=None, camera_configs=[])
    env_config.cartesian_control_param_config = (
        "./crisp_gym/config/control/default_cartesian_impedance.yaml"
    )
    env_config.joint_control_param_config = (
        "./crisp_gym/config/control/joint_control.yaml"
    )
    try:
        control_robot_with_spacemouse(
            env_config=env_config,
            namespace=args.namespace
        )
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
