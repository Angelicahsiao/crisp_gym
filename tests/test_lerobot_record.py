"""Test that the lerobot recording pipeline works without a real robot.

Covers the lerobot upgrade changes; works with both v0.4.x and v0.5.1.
  - CODEBASE_VERSION import (dataset_metadata in v0.5.1, lerobot_dataset in v0.4.x)
  - LeRobotDataset.create() / add_frame() / save_episode() API
  - LeRobotDataset.resume() classmethod (v0.5.1+) with v0.4.x fallback
  - get_features() version check accepts v3.x

Run with:
    python tests/test_lerobot_record.py
"""

import os
import tempfile as _tempfile_early
# Prevent lerobot's HF Hub calls (v0.4.x's LeRobotDataset constructor calls
# get_safe_version which hits the Hub).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Redirect lerobot's default dataset cache so RecordingManager._create_dataset()
# (which calls LeRobotDataset.create() without an explicit root=) writes to a
# scratch dir instead of ~/.cache/huggingface/lerobot. Must be set before any
# lerobot import — its module-level HF_LEROBOT_HOME constant freezes this.
_LEROBOT_TEST_HOME = _tempfile_early.mkdtemp(prefix="lerobot_test_home_")
os.environ["HF_LEROBOT_HOME"] = _LEROBOT_TEST_HOME


def _import_lerobot_metadata():
    """Match the production code's version-aware imports."""
    try:
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
    except ImportError:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
    return CODEBASE_VERSION, LeRobotDatasetMetadata

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
    "crisp_py.sensors", "crisp_py.sensors.sensor", "crisp_py.sensors.sensor_config",
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

class _OrientationRepresentation:
    EULER = "euler"
    QUATERNION = "quaternion"
    ANGLE_AXIS = "angle_axis"

_geo.OrientationRepresentation = _OrientationRepresentation
_geo.Pose = object

_robot_cfg = sys.modules["crisp_py.robot.robot_config"]
_robot_cfg.RobotConfig = object
_robot_cfg.FrankaConfig = object
_robot_cfg.URConfig = object
_robot_cfg.make_robot_config = lambda *a, **kw: None

_robot_pkg = sys.modules["crisp_py.robot"]
_robot_pkg.Robot = object
_robot_pkg.RobotConfig = object
_robot_pkg.FrankaConfig = object
_robot_pkg.Pose = object
_robot_pkg.make_robot_config = lambda *a, **kw: None

_gripper_cfg = sys.modules["crisp_py.gripper.gripper"]
_gripper_cfg.GripperConfig = object

_gripper_pkg = sys.modules["crisp_py.gripper"]
_gripper_pkg.Gripper = object
_gripper_pkg.GripperConfig = object
_gripper_pkg.make_gripper = lambda *a, **kw: None

_cam_pkg = sys.modules["crisp_py.camera"]
_cam_pkg.Camera = object

_cam_cfg = sys.modules["crisp_py.camera.camera_config"]
_cam_cfg.CameraConfig = object

_sensor_pkg = sys.modules["crisp_py.sensors.sensor"]
_sensor_pkg.Sensor = object

_sensor_cfg = sys.modules["crisp_py.sensors.sensor_config"]
_sensor_cfg.SensorConfig = object

# crisp_gym.config.path calls importlib.resources.files("crisp_py") at module
# level — that call fails when crisp_py is a bare ModuleType stub. Stub the
# entire module so the import chain can proceed.
import tempfile as _tempfile
_path_stub = _make_stub("crisp_gym.config.path")
_path_stub.CRISP_CONFIG_PATH = _tempfile.mkdtemp()
_path_stub.CRISP_CONFIG_PATHS = [_path_stub.CRISP_CONFIG_PATH]
_path_stub.find_config = lambda *a, **kw: None
_path_stub.list_configs_in_folder = lambda *a, **kw: []
sys.modules["crisp_gym.config"] = _make_stub("crisp_gym.config")
sys.modules["crisp_gym.config.path"] = _path_stub

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
    """CODEBASE_VERSION must be importable (v0.5.1 path or v0.4.x fallback)."""
    CODEBASE_VERSION, _ = _import_lerobot_metadata()
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


def _add_frame_compat(dataset, frame, task):
    """Call add_frame with task as kwarg (v0.4.x) or in frame dict (v0.5.x)."""
    from inspect import signature
    if "task" in signature(dataset.add_frame).parameters:
        dataset.add_frame(frame, task=task)
    else:
        frame_with_task = dict(frame, task=task)
        dataset.add_frame(frame_with_task)


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
        root=tmp_dir / "ds_create",
        use_videos=False,
    )

    frame = {
        "action": np.zeros(7, dtype=np.float32),
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.state.cartesian": np.zeros(6, dtype=np.float32),
        "observation.state.gripper": np.zeros(1, dtype=np.float32),
        "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    _add_frame_compat(dataset, frame, task="pick the block")
    _add_frame_compat(dataset, frame, task="pick the block")
    dataset.save_episode()

    assert dataset.num_episodes == 1, f"Expected 1 episode, got {dataset.num_episodes}"


def test_lerobot_dataset_metadata_import():
    """LeRobotDatasetMetadata must be importable (v0.5.1 path or v0.4.x fallback)."""
    _, LeRobotDatasetMetadata = _import_lerobot_metadata()
    assert LeRobotDatasetMetadata is not None


class _ConcreteRecordingManager:
    """Built lazily via _make_concrete_rm() to avoid importing at module load."""


def _make_concrete_rm():
    from crisp_gym.record.recording_manager import RecordingManager

    class ConcreteRM(RecordingManager):
        def get_instructions(self):
            return ""
    return ConcreteRM


def _unique_repo_id(prefix):
    import uuid
    return f"test_user/{prefix}_{uuid.uuid4().hex[:8]}"


def test_recording_manager_create(tmp_dir):
    """RecordingManager._create_dataset() creates a new dataset correctly."""
    from crisp_gym.util.lerobot_features import get_features
    from crisp_gym.record.recording_manager_config import RecordingManagerConfig

    ConcreteRM = _make_concrete_rm()
    env = _make_mock_env()
    features = get_features(env, use_video=False)

    config = RecordingManagerConfig(
        features=features,
        repo_id=_unique_repo_id("rm_create"),
        robot_type="franka",
        fps=15,
        num_episodes=2,
        resume=False,
        push_to_hub=False,
        use_sound=False,
    )

    manager = object.__new__(ConcreteRM)
    manager.config = config
    dataset = manager._create_dataset()

    assert dataset is not None
    assert dataset.num_episodes == 0




# ---------------------------------------------------------------------------
# Deploy / inference tests
# ---------------------------------------------------------------------------

def test_concatenate_state_features():
    """concatenate_state_features should join all observation.state.* into one vector."""
    from crisp_gym.util.lerobot_features import concatenate_state_features

    obs = {
        "observation.state.cartesian": np.arange(6, dtype=np.float32),
        "observation.state.gripper": np.array([0.5], dtype=np.float32),
        "observation.images.wrist": np.zeros((4, 4, 3), dtype=np.uint8),  # ignored
    }
    out = concatenate_state_features(obs)
    assert out.shape == (7,), f"Expected (7,), got {out.shape}"
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:6], np.arange(6, dtype=np.float32))
    assert out[6] == 0.5


def test_numpy_obs_to_torch():
    """numpy_obs_to_torch should produce torch tensors with correct shape and batch dim."""
    import torch
    from crisp_gym.util.lerobot_features import numpy_obs_to_torch

    obs = {
        "observation.state": np.arange(7, dtype=np.float32),
        "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "task": "pick the block",
    }
    torch_obs = numpy_obs_to_torch(obs)
    assert isinstance(torch_obs["observation.state"], torch.Tensor)
    assert torch_obs["observation.state"].shape == (1, 7)
    assert torch_obs["observation.state"].dtype == torch.float32
    assert torch_obs["task"] == "pick the block"


def test_make_pre_post_processors_detection():
    """Verify the USE_LEROBOT_PROCESSORS flag matches actual import availability."""
    from crisp_gym.policy import lerobot_policy

    try:
        from lerobot.policies.factory import make_pre_post_processors  # noqa: F401
        expected = True
    except ImportError:
        expected = False

    assert lerobot_policy.USE_LEROBOT_PROCESSORS == expected, (
        f"USE_LEROBOT_PROCESSORS={lerobot_policy.USE_LEROBOT_PROCESSORS} but actual import={expected}"
    )


def _make_tiny_act_policy(state_dim=7, action_dim=7, image_shape=(3, 96, 96)):
    """Build a minimal ACTPolicy for round-trip save/load testing."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    # PolicyFeature/FeatureType lives at different places across versions.
    try:
        from lerobot.configs.types import PolicyFeature, FeatureType
    except ImportError:
        from lerobot.configs.policies import PolicyFeature, FeatureType  # fallback

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=4,
        n_action_steps=1,
        # Tiny model to keep the test fast
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
    )

    # Dataset stats are needed for normalization layers
    import torch
    stats = {
        "observation.state": {
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
            "min": -torch.ones(state_dim),
            "max": torch.ones(state_dim),
        },
        "observation.images.wrist": {
            "mean": torch.zeros(image_shape).reshape(3, 1, 1),
            "std": torch.ones(image_shape).reshape(3, 1, 1),
        },
        "action": {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
            "min": -torch.ones(action_dim),
            "max": torch.ones(action_dim),
        },
    }

    return ACTPolicy(cfg, dataset_stats=stats)


def test_policy_save_load_inference(tmp_dir):
    """Create a tiny ACT policy, save_pretrained → from_pretrained → select_action."""
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    state_dim, action_dim = 7, 7
    image_shape = (3, 96, 96)

    policy = _make_tiny_act_policy(state_dim, action_dim, image_shape)
    save_dir = tmp_dir / "tiny_act"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(save_dir))

    loaded = ACTPolicy.from_pretrained(str(save_dir))
    loaded.eval()

    obs = {
        "observation.state": torch.zeros(1, state_dim),
        "observation.images.wrist": torch.zeros(1, *image_shape),
    }
    with torch.inference_mode():
        action = loaded.select_action(obs)

    assert isinstance(action, torch.Tensor)
    # Shape may be (1, action_dim) or (action_dim,) depending on version
    assert action.shape[-1] == action_dim, f"Unexpected action shape: {action.shape}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    print("\n=== LeRobot Record Pipeline Tests (v0.4.x / v0.5.1) ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        tmp_path = Path(tmp)

        run("CODEBASE_VERSION import from dataset_metadata", test_codebase_version_import)
        run("get_features() returns valid schema", test_get_features)
        run("get_features() no spurious version warning", test_get_features_version_check)
        run("LeRobotDataset.create → add_frame → save_episode", lambda: test_dataset_create_add_save(tmp_path))
        run("LeRobotDatasetMetadata import from dataset_metadata", test_lerobot_dataset_metadata_import)
        run("RecordingManager._create_dataset() new dataset", lambda: test_recording_manager_create(tmp_path))

        print("\n--- Deploy / inference tests ---\n")
        run("concatenate_state_features", test_concatenate_state_features)
        run("numpy_obs_to_torch", test_numpy_obs_to_torch)
        run("USE_LEROBOT_PROCESSORS detection", test_make_pre_post_processors_detection)
        run("ACT policy save_pretrained → from_pretrained → select_action", lambda: test_policy_save_load_inference(tmp_path))

    print("\n=== Summary ===")
    passed = sum(1 for _, s in results if s == "pass")
    failed = sum(1 for _, s in results if s == "fail")
    skipped = sum(1 for _, s in results if s == "skip")
    print(f"  {passed} passed  |  {failed} failed  |  {skipped} skipped\n")

    if failed:
        sys.exit(1)
