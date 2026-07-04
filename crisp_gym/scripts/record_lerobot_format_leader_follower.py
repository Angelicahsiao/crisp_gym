"""Script showcasing how to record data in Lerobot Format."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import rclpy

import crisp_gym  # noqa: F401
from crisp_gym.config.home import HomeConfig
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env, ManipulatorJointEnv
from crisp_gym.envs.manipulator_env_config import list_env_configs, FrankaEnvConfig
from crisp_gym.record.record_config import RecordConfig
from crisp_gym.record.record_functions import (
    make_record_fn,
    make_streamer_drive_fn,
    make_teleop_drive_fn,
    make_teleop_fn,
    make_teleop_streamer_fn,
)
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.teleop.teleop_robot import TeleopRobot, make_leader
from crisp_gym.teleop.teleop_robot_config import list_leader_configs
from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


def main():
    """Record data in Lerobot Format using a leader-follower teleoperation setup."""
    parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="test",
        help="Repository ID for the dataset",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["pick the lego block."],
        help="List of task descriptions to record data for, e.g. 'clean red' 'clean green'",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Type of robot being used.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for recording",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume recording of an already existing dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--recording-manager-type",
        type=str,
        default="keyboard",
        help="Type of recording manager to use. Currently only 'keyboard' and 'ros' are supported.",
    )
    parser.add_argument(
        "--leader-config",
        type=str,
        default="right_leader",
        help="Configuration name for the leader robot. You can define your own configurations, please check https://utiasdsl.github.io/crisp_controllers/misc/create_own_config/.",
    )
    parser.add_argument(
        "--follower-config",
        type=str,
        default="left_robot_env",
        help="Configuration name for the follower robot. You can define your own configurations, please check https://utiasdsl.github.io/crisp_controllers/misc/create_own_config/.",
    )
    parser.add_argument(
        "--follower-namespace",
        type=str,
        default="",
        help="Namespace for the follower robot. This is used to identify the robot in the ROS ecosystem.",
    )
    parser.add_argument(
        "--leader-namespace",
        type=str,
        default="right",
        help="Namespace for the leader robot. This is used to identify the robot in the ROS ecosystem.",
    )
    parser.add_argument(
        "--joint-control",
        action="store_true",
        help="Whether to use joint control for the robot.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logger level.",
    )

    parser.add_argument(
        "--record-config",
        type=str,
        default=None,
        help=(
            "Optional RecordConfig YAML (data contract). When given, the "
            "config-driven generic recorder is used (e.g. "
            "config/recording/umi_robot_record.yaml records UMI-style actions "
            "= measured TCP pose[t+1] while the teleop still drives the robot). "
            "When omitted, the legacy command-recording behavior is kept."
        ),
    )
    parser.add_argument(
        "--use-streamed-teleop",
        action="store_true",
        help="Whether to use streamed teleop (e.g., from a phone or VR device) for the leader robot.",
    )
    parser.add_argument(
        "--home-config-noise",
        type=float,
        default=0.0,
        help="Noise to add to the home configuration when homing the robots to randomize the position a bit.",
    )

    args = parser.parse_args()

    # Set up logger
    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg:<30}: {value}")

    # Validate arguments not passed by the user
    if args.follower_namespace is None:
        args.follower_namespace = prompt.prompt(
            "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
            default="",
        )
        logger.info(f"Using follower namespace: {args.follower_namespace}")

    if args.leader_namespace is None and not args.use_streamed_teleop:
        args.leader_namespace = prompt.prompt(
            "Please enter the leader robot namespace (e.g., 'left', 'right', ...)",
            default="left",
        )
        logger.info(f"Using leader namespace: {args.leader_namespace}")

    if args.leader_config is None and not args.use_streamed_teleop:
        leader_configs = list_leader_configs()
        args.leader_config = prompt.prompt(
            "Please enter the leader robot configuration name.",
            options=leader_configs,
            default=leader_configs[0],
        )
        logger.info(f"Using leader configuration: {args.leader_config}")

    if args.follower_config is None:
        follower_configs = list_env_configs()
        args.follower_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info(f"Using follower configuration: {args.follower_config}")

    # Add Franka-specific setup logic
    if args.follower_config == "franka":
        # Set up Franka's custom environment config
        CTRL_FREQ = 50
        BASE_DIR = Path(crisp_gym.__file__).parent
        env_config = FrankaEnvConfig(control_frequency=CTRL_FREQ, gripper_config=None, camera_configs=[])
        # env_config.cartesian_control_param_config = str(
        #     BASE_DIR / "config/control/default_cartesian_impedance.yaml"
        # )
        # env_config.joint_control_param_config = str(
        #     BASE_DIR / "config/control/joint_control.yaml"
        # )
        ctrl_type = "cartesian" if not args.joint_control else "joint"
        print("change path")
        # Choose the environment based on joint or cartesian control
        if args.joint_control:  # If joint control is enabled
            env = ManipulatorJointEnv(namespace=args.follower_namespace, config=env_config)
        else:  # Default to cartesian control
            env = ManipulatorCartesianEnv(namespace=args.follower_namespace, config=env_config)

    # If the follower config is not "franka", proceed with default logic
    else:
        ctrl_type = "cartesian" if not args.joint_control else "joint"
        env = make_env(
            env_type=args.follower_config,
            control_type=ctrl_type,
            namespace=args.follower_namespace,
        )
        print("Cartesian config:",
            env.config.cartesian_control_param_config)

        print("Joint config:",
            env.config.joint_control_param_config)
        print("Config class:", type(env.config))
        print("Your env_type:", args.follower_config)

    try:

        leader: TeleopRobot | TeleopStreamedPose | None = None
        if args.use_streamed_teleop:
            leader = TeleopStreamedPose()
            logger.info("Using streamed teleop for the leader robot.")
        else:
            leader = make_leader(args.leader_config, namespace=args.leader_namespace)
            leader.wait_until_ready()
            logger.info("Using teleop robot for the leader robot. Leader is ready.")

        record_config = None
        if args.record_config is not None:
            record_config = RecordConfig.from_yaml(args.record_config)
            logger.info(f"Using record config contract '{record_config.name}'")
            if float(args.fps) != float(record_config.rate_hz):
                raise ValueError(
                    f"--fps {args.fps} != record config rate_hz "
                    f"{record_config.rate_hz}. Align them (part of the data contract)."
                )
            features = record_config.to_features(
                joint_count=env.config.robot_config.num_joints()
            )
        else:
            keys_to_ignore = []
            features = get_features(env=env, ignore_keys=keys_to_ignore, fps=args.fps)
        logger.debug(f"Using the features: {features}")

        if args.use_streamed_teleop and ctrl_type != "cartesian":
            raise ValueError(
                "Streamed teleop is only compatible with Cartesian control. Please disable joint control."
            )

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            num_episodes=args.num_episodes,
            fps=args.fps,
            resume=args.resume,
            push_to_hub=args.push_to_hub,
        )
        recording_manager.wait_until_ready()
        logger.info("Recording manager is ready.")

        env_metadata = env.get_metadata()

        with open(recording_manager.dataset_directory / "meta" / "crisp_meta.json", "w") as f:
            json.dump(env_metadata, f, indent=4)

        if record_config is not None:
            with open(
                recording_manager.dataset_directory / "meta" / "record_config.json", "w"
            ) as f:
                json.dump(record_config.to_metadata(), f, indent=4)
            logger.info("Record contract saved to meta/record_config.json")

        logger.info(
            f"Environment metadata saved to {recording_manager.dataset_directory / 'meta' / 'crisp_meta.json'}"
        )

        logger.info("Homing both robots before starting with recording.")

        # Prepare environment and leader
        if isinstance(leader, TeleopRobot):
            leader.prepare_for_teleop()
            logger.debug("[DEBUG] leader.prepare_for_teleop() done")

        env.wait_until_ready()
        logger.debug("[DEBUG] env.wait_until_ready() done")
        env.home(home_config=HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise))
        logger.debug("[DEBUG] env.home() done")
        print("HomeConfig:", HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise))
        print("HomeConfig:", HomeConfig.OPEN_POSE.randomize(noise=args.home_config_noise))
        env.home(home_config=HomeConfig.OPEN_POSE.randomize(noise=args.home_config_noise))
        env.reset()
        logger.debug("[DEBUG] env.reset() done")

        tasks = list(args.tasks)

        def on_start():
            """Hook function to be called when starting a new episode."""
            env.robot.reset_targets()
            env.reset()

            if isinstance(leader, TeleopRobot):
                # TODO: @danielsanjosepro: allow user to change controllers based on config

                leader.robot.reset_targets()
                leader.robot.cartesian_controller_parameters_client.load_param_config(
                    leader.config.gravity_compensation_controller
                )
                leader.robot.controller_switcher_client.switch_controller(
                    "cartesian_impedance_controller"
                )

                if leader.gripper is not None:
                    leader.gripper.disable_torque()

        def on_end():
            """Hook function to be called when stopping the recording."""
            env.robot.reset_targets()
            random_home = HomeConfig.OPEN_POSE.randomize(noise=args.home_config_noise)
            env.robot.home(blocking=False, home_config=random_home)
            if isinstance(leader, TeleopRobot):
                leader.robot.reset_targets()
                # Activate incase leader should go to the same position as the follower
                # leader.robot.home(blocking=False, home_config=random_home)
                leader.robot.home(blocking=False)
            env.gripper.open()

        with recording_manager:
            while not recording_manager.done():
                logger.info(
                    f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                )

                # Create a new teleop function for each episode to reset internal variables
                teleop_fn = None
                if record_config is not None:
                    # Config-driven recorder: teleop only drives, the record
                    # config decides what is stored (e.g. UMI next_tcp_pose).
                    if isinstance(leader, TeleopRobot):
                        drive_fn = make_teleop_drive_fn(env, leader)
                    elif isinstance(leader, TeleopStreamedPose) and isinstance(
                        env, ManipulatorCartesianEnv
                    ):
                        drive_fn = make_streamer_drive_fn(env, leader)
                    else:
                        raise ValueError(
                            "Streamed teleop is only compatible with Cartesian control."
                        )
                    teleop_fn = make_record_fn(env, record_config, drive_fn=drive_fn)
                elif isinstance(leader, TeleopRobot):
                    teleop_fn = make_teleop_fn(env, leader)
                elif isinstance(leader, TeleopStreamedPose) and isinstance(
                    env, ManipulatorCartesianEnv
                ):
                    teleop_fn = make_teleop_streamer_fn(env, leader)
                else:
                    raise ValueError(
                        "Streamed teleop is only compatible with Cartesian control. Please disable joint control."
                    )

                task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
                logger.info(f"▷ Task: {task}")

                recording_manager.record_episode(
                    data_fn=teleop_fn,
                    task=task,
                    on_start=on_start,
                    on_end=on_end,
                )

        if isinstance(leader, TeleopRobot):
            logger.info("Homing leader.")
            leader.robot.home()
        logger.info("Homing follower.")
        env.home()

        logger.info("Closing the environment.")
        env.close()

        logger.info("Finished recording.")

    except TimeoutError as e:
        logger.exception(f"Timeout error occurred during recording: {e}.")
        logger.error(
            "Please check if the robot container is running and the namespace is correct."
            "\nYou can check the topics using `ros2 topic list` command."
        )

    except Exception as e:
        logger.exception(f"An error occurred during recording: {e}.")

    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
