"""Test that the lerobot recording pipeline works without a real robot.

Covers the changes made for the LeRobot v0.5.1 upgrade:
  - CODEBASE_VERSION import from lerobot.datasets.dataset_metadata
  - LeRobotDataset.create() / add_frame() / save_episode() API
  - LeRobotDataset.resume() classmethod (replaces plain constructor)
  - get_features() version check accepts v3.x

Run with:
    python tests/test_lerobot_record.py
"""

import os
import sys
import tempfile
import types
import traceback
import numpy as np


# ---------------------------------------------------------------------------
# Stub out ROS2 and robot packages so the module can be imported without them
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    return mod


ROS_STUBS = [
    "rclpy", "rclpy.node", "rclpy.executors", "rclpy.qos",
    "std_msgs", "std_msgs.msg",
    "crisp_py", "crisp_py.robot", "crisp_py.robot.robot_config",
    "crisp_py.gripper", "crisp_py.gripper.gripper",
    "crisp_py.camera", "crisp_py.camera.camera_config",
    "crisp_py.sensors", "crisp_py.sensors.sensor_config",
    "crisp_py.utils", "crisp_py.utils.geometry",
    "pynput", "pynput.keyboard",
]

for _name in ROS_STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = _make_stub(_name)

# Minimal stubs for symbols actually referenced at import time
sys.modules["rclpy"].ok = lambda: False
sys.modules["rclpy"].create_node = lambda *a, **kw: None
sys.modules["rclpy.executors"].SingleThreadedExecutor = object
sys.modules["std_msgs.msg"].String = object

_geo = sys.modules["crisp_py.utils.geometry"]
_geo.OrientationRepresentation = object

_robot_cfg = sys.modules["crisp_py.robot.robot_config"]
_robot_cfg.RobotConfig = object
_robot_cfg.FrankaConfig = object
_robot_cfg.URConfig = object
_robot_cfg.make_robot_config = lambda *a, **kw: None

_gripper_cfg = sys.modules["crisp_py.gripper.gripper"]
_gripper_cfg.GripperConfig = object

_cam_cfg = sys.modules["crisp_py.camera.camera_config"]
_cam_cfg.CameraConfig = object

_sensor_cfg = sys.modules["crisp_py.sensors.sensor_config"]
_sensor_cfg.SensorConfig = object

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def run(label, fn):
    try:
        fn()
        print(f"  [{PASS}] {label}")
        results.append((label, "pass"))
    except ImportError as e:
        print(f"  [{SKIP}] {label} — lerobot not installed: {e}")
        results.append((label, "skip"))
    except Exception:
        print(f"  [{FAIL}] {label}")
        traceback.print_exc()
        results.append((label, "fail"))


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

def _make_mock_env(num_joints=7):
    """Return a mock ManipulatorBaseEnv-like object with a realistic obs space."""
    import gymnasium

    obs_space = gymnasium.spaces.Dict({
        "observation.state.cartesian": gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        ),
        "observation.state.gripper": gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        ),
        "observation.images.wrist": gymnasium.spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        ),
    })

    robot_config = types.SimpleNamespace(num_joints=lambda: num_joints)
    config = types.SimpleNamespace(
        robot_config=robot_config,
        control_frequency=15,
    )

    from crisp_gym.util.control_type import ControlType

    env = types.SimpleNamespace(
        observation_space=obs_space,
        ctrl_type=ControlType.CARTESIAN,
        config=config,
        get_metadata=lambda: {"control_frequency": 15, "ctrl_type": "cartesian"},
    )
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_codebase_version_import():
    """CODEBASE_VERSION must import from dataset_metadata in v0.5.1."""
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    assert isinstance(CODEBASE_VERSION, str), "CODEBASE_VERSION should be a string"
    assert CODEBASE_VERSION.startswith("v"), f"Unexpected format: {CODEBASE_VERSION}"


def test_get_features():
    """get_features() should return a valid feature dict without warnings."""
    from crisp_gym.util.lerobot_features import get_features
    env = _make_mock_env()
    features = get_features(env, use_video=True)

    assert "observation.state" in features
    assert "action" in features
    assert "observation.images.wrist" in features
    assert features["observation.images.wrist"]["dtype"] == "video"
    assert features["action"]["shape"] == (7,)  # 6 cartesian + 1 gripper


def test_get_features_version_check():
    """get_features() should not warn for v3.x datasets."""
    import logging
    from crisp_gym.util.lerobot_features import get_features, CODEBASE_VERSION

    # Capture warnings
    warnings = []
    handler = logging.handlers_capture = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            if record.levelno == logging.WARNING:
                warnings.append(record.getMessage())

    import logging as _logging
    logger = _logging.getLogger("crisp_gym.util.lerobot_features")
    handler = CapturingHandler()
    logger.addHandler(handler)
    try:
        env = _make_mock_env()
        get_features(env)
    finally:
        logger.removeHandler(handler)

    version_warnings = [w for w in warnings if "implemented for" in w or "Expect unexpected" in w]
    assert len(version_warnings) == 0, (
        f"get_features() issued an unexpected version warning for {CODEBASE_VERSION}: {version_warnings}"
    )


def test_dataset_create_add_save(tmp_dir):
    """LeRobotDataset.create() → add_frame() → save_episode() flow."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from crisp_gym.util.lerobot_features import get_features

    env = _make_mock_env()
    features = get_features(env, use_video=False)  # use_video=False avoids ffmpeg dependency

    repo_id = "test_user/test_dataset_create"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type="franka",
        features=features,
        root=tmp_dir,
        use_videos=False,
    )

    obs_space = env.observation_space
    frame = {
        "action": np.zeros(7, dtype=np.float32),
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.state.cartesian": np.zeros(6, dtype=np.float32),
        "observation.state.gripper": np.zeros(1, dtype=np.float32),
        "task": "pick the block",
    }
    dataset.add_frame(frame)
    dataset.add_frame(frame)
    dataset.save_episode()

    assert dataset.num_episodes == 1, f"Expected 1 episode, got {dataset.num_episodes}"


def test_dataset_resume(tmp_dir):
    """LeRobotDataset.resume() should reopen a dataset for continued writing."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from crisp_gym.util.lerobot_features import get_features

    env = _make_mock_env()
    features = get_features(env, use_video=False)
    repo_id = "test_user/test_dataset_resume"

    # First: create and save one episode
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type="franka",
        features=features,
        root=tmp_dir,
        use_videos=False,
    )
    frame = {
        "action": np.zeros(7, dtype=np.float32),
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.state.cartesian": np.zeros(6, dtype=np.float32),
        "observation.state.gripper": np.zeros(1, dtype=np.float32),
        "task": "pick the block",
    }
    dataset.add_frame(frame)
    dataset.save_episode()
    assert dataset.num_episodes == 1

    # Second: resume and add another episode
    dataset2 = LeRobotDataset.resume(repo_id=repo_id, root=tmp_dir)
    dataset2.add_frame(frame)
    dataset2.save_episode()
    assert dataset2.num_episodes == 2, f"Expected 2 episodes after resume, got {dataset2.num_episodes}"


def test_lerobot_dataset_metadata_import():
    """LeRobotDatasetMetadata must import from lerobot.datasets.dataset_metadata."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    assert LeRobotDatasetMetadata is not None


def test_recording_manager_create(tmp_dir):
    """RecordingManager._create_dataset() creates a new dataset correctly."""
    from unittest.mock import patch
    from crisp_gym.util.lerobot_features import get_features
    from crisp_gym.record.recording_manager import RecordingManager
    from crisp_gym.record.recording_manager_config import RecordingManagerConfig

    env = _make_mock_env()
    features = get_features(env, use_video=False)

    config = RecordingManagerConfig(
        features=features,
        repo_id="test_user/test_rm_create",
        robot_type="franka",
        fps=15,
        num_episodes=2,
        resume=False,
        push_to_hub=False,
        use_sound=False,
    )

    with patch("crisp_gym.record.recording_manager.HF_LEROBOT_HOME", tmp_dir):
        # Instantiate without starting the writer process
        manager = object.__new__(RecordingManager)
        manager.config = config
        dataset = manager._create_dataset()

    assert dataset is not None
    assert dataset.num_episodes == 0


def test_recording_manager_resume(tmp_dir):
    """RecordingManager._create_dataset() with resume=True uses LeRobotDataset.resume()."""
    from unittest.mock import patch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from crisp_gym.util.lerobot_features import get_features
    from crisp_gym.record.recording_manager import RecordingManager
    from crisp_gym.record.recording_manager_config import RecordingManagerConfig

    env = _make_mock_env()
    features = get_features(env, use_video=False)
    repo_id = "test_user/test_rm_resume"

    # Create dataset first
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        robot_type="franka",
        features=features,
        root=tmp_dir,
        use_videos=False,
    )
    frame = {
        "action": np.zeros(7, dtype=np.float32),
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.state.cartesian": np.zeros(6, dtype=np.float32),
        "observation.state.gripper": np.zeros(1, dtype=np.float32),
        "task": "pick the block",
    }
    ds.add_frame(frame)
    ds.save_episode()

    config = RecordingManagerConfig(
        features=features,
        repo_id=repo_id,
        robot_type="franka",
        fps=15,
        num_episodes=5,
        resume=True,
        push_to_hub=False,
        use_sound=False,
    )

    with patch("crisp_gym.record.recording_manager.HF_LEROBOT_HOME", tmp_dir):
        manager = object.__new__(RecordingManager)
        manager.config = config
        manager.episode_count_queue = __import__("multiprocessing").Queue(1)
        dataset = manager._create_dataset()

    assert dataset.num_episodes == 1, f"Expected 1 existing episode, got {dataset.num_episodes}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    print("\n=== LeRobot v0.5.1 Record Pipeline Tests ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        tmp_path = Path(tmp)

        run("CODEBASE_VERSION import from dataset_metadata", test_codebase_version_import)
        run("get_features() returns valid schema", test_get_features)
        run("get_features() no spurious version warning", test_get_features_version_check)
        run("LeRobotDataset.create → add_frame → save_episode", lambda: test_dataset_create_add_save(tmp_path))
        run("LeRobotDataset.resume() classmethod", lambda: test_dataset_resume(tmp_path))
        run("LeRobotDatasetMetadata import from dataset_metadata", test_lerobot_dataset_metadata_import)
        run("RecordingManager._create_dataset() new dataset", lambda: test_recording_manager_create(tmp_path))
        run("RecordingManager._create_dataset() resume path", lambda: test_recording_manager_resume(tmp_path))

    print("\n=== Summary ===")
    passed = sum(1 for _, s in results if s == "pass")
    failed = sum(1 for _, s in results if s == "fail")
    skipped = sum(1 for _, s in results if s == "skip")
    print(f"  {passed} passed  |  {failed} failed  |  {skipped} skipped\n")

    if failed:
        sys.exit(1)
