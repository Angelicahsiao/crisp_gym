"""Script for recording UMI handheld gripper demonstrations in LeRobot format.

Uses OptiTrack (via ROS2 PoseStamped) for real-time pose and a Float32 topic
for gripper width. No robot arm is controlled — this only records handheld data.

Example:
    python record_umi_handheld.py \\
        --repo-id my_org/umi_demo \\
        --tasks "pick the lego block" \\
        --num-episodes 50 \\
        --env-config /path/to/umi_handheld.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rclpy

from crisp_gym.config.path import find_config
from crisp_gym.envs.umi_handheld_env import UmiHandheldEnv
from crisp_gym.record.record_config import RecordConfig
from crisp_gym.record.record_functions import make_record_fn
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.setup_logger import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Record UMI handheld demonstrations in LeRobot format."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="test",
        help="Repository ID for the dataset.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["finish the task."],
        help="Task descriptions to cycle through, e.g. 'pick the lego block'.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="umi_handheld",
        help="Robot type label stored in dataset metadata.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Recording frame rate.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to record.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an existing dataset.",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push the dataset to the Hugging Face Hub after recording.",
    )
    parser.add_argument(
        "--recording-manager-type",
        type=str,
        default="keyboard",
        help="Recording manager type. Supported: 'keyboard', 'ros'.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help=(
            "Path to the UMI handheld environment YAML config. "
            "Defaults to the bundled crisp_gym/config/envs/umi_handheld.yaml."
        ),
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="ROS2 namespace for the environment node.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    parser.add_argument(
        "--record-config",
        type=str,
        default=None,
        help=(
            "Path to a RecordConfig YAML (data contract: observation sources, "
            "action definition). Defaults to the bundled "
            "config/recording/umi_handheld_record.yaml."
        ),
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg:<30}: {value}")

    # Resolve environment config path
    if args.env_config is None:
        config_path = find_config("envs/umi_handheld.yaml")
        if config_path is None:
            raise FileNotFoundError(
                "Could not find bundled 'umi_handheld.yaml'. "
                "Specify --env-config explicitly."
            )
    else:
        config_path = Path(args.env_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Env config not found: {config_path}")

    logger.info(f"Loading env config from: {config_path}")
    env = UmiHandheldEnv.from_yaml(config_path, namespace=args.namespace)

    try:
        logger.info("Waiting for OptiTrack pose and gripper width topics...")
        env.wait_until_ready()
        logger.info("Environment is ready.")

        # Load the record config (the dataset's data contract)
        if args.record_config is None:
            rc_path = find_config("recording/umi_handheld_record.yaml")
            if rc_path is None:
                raise FileNotFoundError(
                    "Bundled 'recording/umi_handheld_record.yaml' not found. "
                    "Specify --record-config explicitly."
                )
        else:
            rc_path = Path(args.record_config)
            if not rc_path.exists():
                raise FileNotFoundError(f"Record config not found: {rc_path}")
        record_config = RecordConfig.from_yaml(rc_path)
        logger.info(f"Record config: {rc_path} (contract '{record_config.name}')")

        if float(args.fps) != float(record_config.rate_hz):
            raise ValueError(
                f"--fps {args.fps} != record config rate_hz {record_config.rate_hz}. "
                "The rate is part of the data contract; align them."
            )

        features = record_config.to_features()
        logger.debug(f"Dataset features: {features}")

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

        # Save env metadata alongside the dataset
        env_metadata = env.get_metadata()
        meta_dir = recording_manager.dataset_directory / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "crisp_meta.json", "w") as f:
            json.dump(env_metadata, f, indent=4)
        logger.info(f"Env metadata saved to {meta_dir / 'crisp_meta.json'}")

        # Stamp the data contract next to the dataset
        with open(meta_dir / "record_config.json", "w") as f:
            json.dump(record_config.to_metadata(), f, indent=4)
        logger.info(f"Record contract saved to {meta_dir / 'record_config.json'}")

        tasks = list(args.tasks)

        def on_start():
            env.reset()

        def on_end():
            pass

        with recording_manager:
            while not recording_manager.done():
                logger.info(
                    f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                )

                teleop_fn = make_record_fn(env, record_config)
                task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
                logger.info(f"▷ Task: {task}")

                recording_manager.record_episode(
                    data_fn=teleop_fn,
                    task=task,
                    on_start=on_start,
                    on_end=on_end,
                )

        logger.info("Finished recording.")

    except TimeoutError as e:
        logger.exception(f"Timeout error: {e}")
        logger.error(
            "Check that mocap4ros2_optitrack is publishing and the relay node is running.\n"
            "Verify with: ros2 topic echo <pose_topic>"
        )
    except Exception as e:
        logger.exception(f"Error during recording: {e}")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
