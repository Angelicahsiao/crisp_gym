"""End-to-end record test against a live ROS2 robot sim.

Prerequisites:
    1. Pixi shell with ROS2 + lerobot env active:
           pixi shell -e humble-lerobot
    2. UR sim running with the controllers from crisp_controllers, e.g.:
           ros2 control list_controllers
       must show cartesian_impedance_controller, joint_trajectory_controller,
       pose_broadcaster, twist_broadcaster, joint_state_broadcaster.

What it does:
    - Creates ManipulatorCartesianEnv (UREnvConfig, code-defined, matching
      examples/01_gym_cartesian_env_ur.py).
    - Reads one observation, builds dataset features, calls
      LeRobotDataset.create() into a tempdir.
    - Sends a zero-delta action via env.step(), grabs another obs.
    - add_frame x2, save_episode, asserts num_episodes == 1 and that the
      episode parquet was written.
    - env.close() to release ROS2 resources.

Run with:
    python tests/test_ros2_live_record.py
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# Make lerobot's HF Hub calls offline-safe in case the dataset constructor is
# ever invoked. LeRobotDataset.create() does not hit Hub, but be defensive.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _make_env():
    from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
    from crisp_gym.envs.manipulator_env_config import UREnvConfig

    cfg = UREnvConfig(control_frequency=50)
    return ManipulatorCartesianEnv(namespace="", config=cfg)


def _build_frame(obs, action, features):
    """Pack one obs/action pair into a frame dict matching `features`."""
    from crisp_gym.util.lerobot_features import concatenate_state_features

    frame = {"action": action.astype(np.float32)}
    frame["observation.state"] = concatenate_state_features(obs).astype(np.float32)
    for key in features:
        if key in ("action", "observation.state"):
            continue
        if key in obs:
            value = obs[key]
            if isinstance(value, np.ndarray):
                frame[key] = value
            else:
                frame[key] = np.array(value)
    return frame


def _add_frame_compat(dataset, frame, task):
    from inspect import signature
    if "task" in signature(dataset.add_frame).parameters:
        dataset.add_frame(frame, task=task)
    else:
        dataset.add_frame(dict(frame, task=task))


def main():
    print("\n=== ROS2 Live Record Test (UR sim) ===\n")

    print(">> Creating ManipulatorCartesianEnv...")
    try:
        env = _make_env()
    except Exception:
        print(f"  [{FAIL}] Could not create env. Is the UR sim running?")
        traceback.print_exc()
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="crisp_live_"))
    failed = False
    try:
        print(">> Reading initial observation...")
        obs = env.get_obs()
        assert isinstance(obs, dict) and len(obs) > 0, "Empty obs"
        print(f"   obs keys: {sorted(obs.keys())}")

        print(">> Building dataset features...")
        from crisp_gym.util.lerobot_features import get_features
        features = get_features(env, use_video=False)
        print(f"   feature keys: {sorted(features.keys())}")

        print(">> Creating LeRobotDataset...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        repo_id = "test_user/ur_live_smoke"
        ds_root = tmp / "ds"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(env.config.control_frequency),
            robot_type="ur",
            features=features,
            root=ds_root,
            use_videos=False,
        )

        action_dim = env.action_space.shape[0]
        zero_action = np.zeros(action_dim, dtype=np.float32)

        print(">> Frame 1: add_frame (current obs, zero action)...")
        frame1 = _build_frame(obs, zero_action, features)
        _add_frame_compat(dataset, frame1, task="ur_smoke")

        print(">> Stepping env with zero action...")
        obs2, _, _, _, _ = env.step(zero_action, block=True)
        time.sleep(0.05)

        print(">> Frame 2: add_frame (post-step obs, zero action)...")
        frame2 = _build_frame(obs2, zero_action, features)
        _add_frame_compat(dataset, frame2, task="ur_smoke")

        print(">> save_episode()...")
        dataset.save_episode()
        assert dataset.num_episodes == 1, f"Expected 1 episode, got {dataset.num_episodes}"

        # Verify episode parquet was actually written somewhere under ds_root.
        parquet_files = list(ds_root.rglob("*.parquet"))
        assert parquet_files, f"No parquet written under {ds_root}"
        print(f"   wrote {len(parquet_files)} parquet file(s): {[p.name for p in parquet_files]}")

        print(f"\n  [{PASS}] Live ROS2 record smoke test\n")
    except Exception:
        failed = True
        print(f"\n  [{FAIL}] Live ROS2 record smoke test")
        traceback.print_exc()
    finally:
        try:
            print(">> Closing env...")
            env.close()
        except Exception:
            print("   (env.close() raised; ignoring)")
            traceback.print_exc()
        shutil.rmtree(tmp, ignore_errors=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
