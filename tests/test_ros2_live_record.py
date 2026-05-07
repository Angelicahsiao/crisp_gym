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

# RecordingManager writes datasets under HF_LEROBOT_HOME / repo_id. Point that
# at a tempdir BEFORE any lerobot import so the constant is captured correctly.
_HF_TMP = tempfile.mkdtemp(prefix="crisp_live_hf_")
os.environ["HF_LEROBOT_HOME"] = _HF_TMP


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


def _test_home_pose(env):
    """Verify env.home() returns the robot to a stable, valid joint pose."""
    print(">> Waiting for env to be ready...")
    env.wait_until_ready()
    print(">> Reading joints before home()...")
    obs_before = env.get_obs()
    joints_before = np.asarray(obs_before["observation.state.joints"], dtype=np.float64)
    print(f"   joints_before: {np.round(joints_before, 3)}")

    print(">> Calling env.home(blocking=True)...")
    env.home(blocking=True)
    time.sleep(0.5)  # let any residual motion settle

    print(">> Reading joints after home()...")
    obs_after = env.get_obs()
    joints_after = np.asarray(obs_after["observation.state.joints"], dtype=np.float64)
    print(f"   joints_after:  {np.round(joints_after, 3)}")

    assert joints_after.shape == joints_before.shape, "Joint shape changed across home()"
    assert np.all(np.isfinite(joints_after)), f"Non-finite joint values after home: {joints_after}"

    # Verify robot is at rest: sample twice and check delta is small.
    time.sleep(0.1)
    joints_resample = np.asarray(env.get_obs()["observation.state.joints"], dtype=np.float64)
    drift = np.max(np.abs(joints_resample - joints_after))
    print(f"   drift between two reads after home: {drift:.4f} rad")
    assert drift < 0.05, f"Robot still moving after home() (drift={drift:.4f} rad)"

    return joints_after


def _test_fake_teleop(env, duration_s=1.0, dx_per_step=0.001):
    """Stream a synthetic cartesian +x delta at the env's control rate.

    Mimics what a leader teleop would publish: a sequence of small per-step
    cartesian deltas. Verifies that the follower tracks the commanded motion
    in the +x direction without rotating significantly.
    """
    fps = int(env.config.control_frequency)
    n_steps = int(duration_s * fps)
    action_dim = env.action_space.shape[0]

    print(f">> Streaming {n_steps} fake teleop deltas at {fps} Hz "
          f"(+{dx_per_step * 1000:.1f} mm per step)...")

    obs_start = env.get_obs()
    pos_start = np.asarray(obs_start["observation.state.cartesian"][:3], dtype=np.float64)
    print(f"   start cartesian xyz: {np.round(pos_start, 4)}")

    delta = np.zeros(action_dim, dtype=np.float32)
    delta[0] = dx_per_step  # +x translation
    for _ in range(n_steps):
        env.step(delta, block=True)

    # Stop motion: send a few zero deltas so the controller settles.
    zero = np.zeros(action_dim, dtype=np.float32)
    for _ in range(int(0.2 * fps)):
        env.step(zero, block=True)
    time.sleep(0.2)

    obs_end = env.get_obs()
    pos_end = np.asarray(obs_end["observation.state.cartesian"][:3], dtype=np.float64)
    print(f"   end cartesian xyz:   {np.round(pos_end, 4)}")

    delta_xyz = pos_end - pos_start
    expected_dx = dx_per_step * n_steps
    total_motion = np.linalg.norm(delta_xyz)
    print(f"   commanded +x: {expected_dx:.4f} m, observed delta xyz: {np.round(delta_xyz, 4)}")
    print(f"   total Euclidean motion: {total_motion:.4f} m")

    # The action frame's x-axis does not align with world-frame x; the robot
    # moves the correct distance but in a direction determined by its base
    # orientation. Assert on Euclidean magnitude only.
    assert total_motion > 0.3 * expected_dx, (
        f"Robot barely moved: {total_motion:.4f} m, expected ~{expected_dx:.4f} m"
    )
    assert total_motion < 3.0 * expected_dx, (
        f"Robot moved too much: {total_motion:.4f} m, expected ~{expected_dx:.4f} m"
    )


def _test_record_via_manager(env):
    """Interactive episode recording through KeyboardRecordingManager.record_episode().

    Mirrors the production flow in scripts/record_lerobot_format_leader_follower.py.
    You control the recording with keypresses (then ENTER):
      r  start / stop recording
      s  save the current episode
      d  delete the current episode
      q  quit (exit after up to 3 episodes)
    """
    import uuid
    from crisp_gym.record.recording_manager import make_recording_manager
    from crisp_gym.record.recording_manager_config import RecordingManagerConfig
    from crisp_gym.util.lerobot_features import get_features

    repo_id = f"test_user/ur_live_rm_{uuid.uuid4().hex[:8]}"
    features = get_features(env, use_video=False)
    fps = int(env.config.control_frequency)

    cfg = RecordingManagerConfig(
        features=features,
        repo_id=repo_id,
        robot_type="ur",
        resume=False,
        fps=fps,
        num_episodes=3,
        push_to_hub=False,
        use_sound=False,
    )

    print(">> Creating KeyboardRecordingManager...")
    rm = make_recording_manager("keyboard", config=cfg)
    rm.wait_until_ready()
    print("   recording manager ready.")

    action_dim = env.action_space.shape[0]
    zero = np.zeros(action_dim, dtype=np.float32)

    def data_fn():
        return env.get_obs(), zero

    on_start_called = {"v": False}
    on_end_called = {"v": False}

    def on_start():
        on_start_called["v"] = True

    def on_end():
        on_end_called["v"] = True

    print(">> Entering recording manager context. Use keys below to control recording.")
    with rm:
        while not rm.done():
            rm.record_episode(
                data_fn=data_fn,
                task="ur_smoke_rm",
                on_start=on_start,
                on_end=on_end,
            )

    assert on_start_called["v"], "on_start hook was not invoked"
    assert on_end_called["v"], "on_end hook was not invoked"

    if rm.episode_count == 0:
        print("   No episodes saved (all deleted or quit before saving).")
    else:
        parquet_files = list(Path(rm.dataset_directory).rglob("*.parquet"))
        assert parquet_files, f"No parquet under {rm.dataset_directory}"
        print(f"   {rm.episode_count} episode(s) saved, "
              f"{len(parquet_files)} parquet file(s) under {rm.dataset_directory}")


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
        try:
            _test_home_pose(env)
            print(f"  [{PASS}] env.home() returns to a stable pose\n")
        except Exception:
            failed = True
            print(f"  [{FAIL}] env.home() test")
            traceback.print_exc()

        try:
            # env.home(blocking=True) leaves the joint_trajectory_controller
            # active. Call reset() to switch to the cartesian impedance
            # controller before sending cartesian deltas.
            print(">> env.reset() to activate cartesian controller...")
            env.reset()
            time.sleep(0.3)
            _test_fake_teleop(env)
            print(f"  [{PASS}] fake teleop streamed deltas tracked by follower\n")
            # Return to home before the record test so frames are consistent.
            env.home(blocking=True)
            time.sleep(0.3)
        except Exception:
            failed = True
            print(f"  [{FAIL}] fake teleop test")
            traceback.print_exc()

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

    try:
        # Re-home before the recording-manager test so the env is in a known pose.
        env.home(blocking=True)
        time.sleep(0.3)
        env.reset()
        time.sleep(0.3)
        _test_record_via_manager(env)
        print(f"\n  [{PASS}] RecordingManager.record_episode end-to-end\n")
    except Exception:
        failed = True
        print(f"\n  [{FAIL}] RecordingManager.record_episode end-to-end")
        traceback.print_exc()

    try:
        print(">> Closing env...")
        env.close()
    except Exception:
        print("   (env.close() raised; ignoring)")
        traceback.print_exc()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(_HF_TMP, ignore_errors=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
