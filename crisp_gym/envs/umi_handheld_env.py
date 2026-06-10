"""UMI Handheld Environment for recording demonstrations with an OptiTrack-tracked gripper.

This environment has no robot arm — it subscribes to an external pose source (OptiTrack via
ROS2 PoseStamped) and a gripper width topic (std_msgs/Float32), and records observations in
the same LeRobot format as the standard manipulator environments.

Usage:
    env = UmiHandheldEnv.from_yaml("config/envs/umi_handheld.yaml")
    env.wait_until_ready()
    obs, _ = env.reset()
    while recording:
        obs, _, _, _, _ = env.step(None)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import rclpy
import rclpy.executors
import yaml
from crisp_py.camera import Camera
from crisp_py.camera.camera_config import CameraConfig
from crisp_py.utils.geometry import OrientationRepresentation
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32

from crisp_gym.config.path import find_config
from crisp_gym.envs.manipulator_env_config import ObservationKeys
from crisp_gym.util.control_type import ControlType

logger = logging.getLogger(__name__)


class _HandheldRobotStub:
    """Minimal stub satisfying get_features()'s robot_config.num_joints() call."""

    def num_joints(self) -> int:
        return 6


@dataclass
class UmiHandheldEnvConfig:
    """Configuration for the UMI Handheld environment.

    Attributes:
        control_frequency: Recording frequency in Hz (used for video metadata).
        pose_topic: ROS2 topic publishing geometry_msgs/PoseStamped from OptiTrack.
        gripper_width_topic: ROS2 topic publishing std_msgs/Float32 gripper width in meters.
        max_gripper_width: Maximum gripper opening in meters; used to normalize to [0, 1].
        camera_configs: List of camera configurations (e.g. GoPro via UVC).
        tx_world_correction: 4×4 homogeneous matrix aligning OptiTrack world frame to Z-up.
            Default is identity — configure once OptiTrack axes are known.
        tx_body_tcp: 4×4 homogeneous matrix from OptiTrack rigid body frame to TCP frame.
            Default is identity — define the rigid body axes in Motive to match TCP convention.
        orientation_representation: Rotation format stored in observations (Euler default).
        observations_to_include_to_state: Which state keys to include in the dataset.
        log_status_every_n_frames: Log current pose every N frames (~1 Hz at 15 fps default).
    """

    control_frequency: float = 15.0
    pose_topic: str = "/optitrack/umi_gripper/pose"
    gripper_width_topic: str = "/umi/gripper_width"
    max_gripper_width: float = 0.09
    camera_configs: List[CameraConfig] = field(default_factory=list)
    tx_world_correction: List[List[float]] = field(
        default_factory=lambda: np.eye(4).tolist()
    )
    tx_body_tcp: List[List[float]] = field(
        default_factory=lambda: np.eye(4).tolist()
    )
    orientation_representation: OrientationRepresentation = OrientationRepresentation.EULER
    observations_to_include_to_state: List[str] = field(
        default_factory=lambda: [
            ObservationKeys.CARTESIAN_OBS,
            ObservationKeys.GRIPPER_OBS,
        ]
    )
    log_status_every_n_frames: int = 15

    # Stub satisfying get_features() which calls env.config.robot_config.num_joints()
    robot_config: _HandheldRobotStub = field(
        default_factory=_HandheldRobotStub, repr=False
    )

    def __post_init__(self):
        if isinstance(self.orientation_representation, str):
            self.orientation_representation = OrientationRepresentation(
                self.orientation_representation
            )

    def get_metadata(self) -> dict:
        return {
            "env_type": "umi_handheld",
            "pose_topic": self.pose_topic,
            "gripper_width_topic": self.gripper_width_topic,
            "max_gripper_width": self.max_gripper_width,
            "orientation_representation": str(self.orientation_representation),
            "camera_configs": [c.__dict__ for c in self.camera_configs],
            "tx_world_correction": self.tx_world_correction,
            "tx_body_tcp": self.tx_body_tcp,
        }

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "UmiHandheldEnvConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if "camera_configs" in data and isinstance(data["camera_configs"], list):
            parsed = []
            for cam in data["camera_configs"]:
                if "from_yaml" in cam:
                    cam_path = find_config(cam["from_yaml"])
                    if cam_path is None:
                        raise FileNotFoundError(f"Camera config '{cam['from_yaml']}' not found.")
                    parsed.append(CameraConfig.from_yaml(yaml_path=cam_path.resolve()))
                else:
                    parsed.append(CameraConfig(**cam))
            data["camera_configs"] = parsed

        # robot_config is a stub — never loaded from YAML
        data.pop("robot_config", None)

        return cls(**data)


class UmiHandheldEnv(gym.Env):
    """Gymnasium environment for recording UMI handheld gripper demonstrations.

    Subscribes to OptiTrack pose and gripper width ROS2 topics. No robot arm is
    controlled — step() is a passthrough that returns the latest sensor observation.
    """

    def __init__(
        self,
        config: UmiHandheldEnvConfig,
        namespace: str = "",
        task: str = "Finish the task.",
    ):
        super().__init__()
        self.config = config
        self.task = task
        self.timestep = 0
        self.ctrl_type = ControlType.CARTESIAN

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("umi_handheld_env", namespace=namespace)
        self._last_pose_msg: PoseStamped | None = None
        self._last_gripper_width: float | None = None
        self._lock = threading.Lock()

        self._node.create_subscription(
            PoseStamped,
            config.pose_topic,
            self._cb_pose,
            qos_profile=qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self._node.create_subscription(
            Float32,
            config.gripper_width_topic,
            self._cb_gripper,
            qos_profile=qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        self.cameras = [
            Camera(namespace=namespace, config=cam_cfg)
            for cam_cfg in config.camera_configs
        ]

        # Pre-compute transform matrices
        self._tx_world_correction = np.array(config.tx_world_correction, dtype=np.float64)
        self._tx_body_tcp = np.array(config.tx_body_tcp, dtype=np.float64)

        rot_dim = self._rotation_dim()
        cartesian_dim = 3 + rot_dim

        state_spaces = {}
        if ObservationKeys.CARTESIAN_OBS in config.observations_to_include_to_state:
            state_spaces[ObservationKeys.CARTESIAN_OBS] = gym.spaces.Box(
                low=-np.inf * np.ones((cartesian_dim,), dtype=np.float32),
                high=np.inf * np.ones((cartesian_dim,), dtype=np.float32),
                dtype=np.float32,
            )
        if ObservationKeys.GRIPPER_OBS in config.observations_to_include_to_state:
            state_spaces[ObservationKeys.GRIPPER_OBS] = gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )

        image_spaces = {
            f"{ObservationKeys.IMAGE_OBS}.{cam.config.camera_name}": gym.spaces.Box(
                low=np.zeros((*cam.config.resolution, 3), dtype=np.uint8),
                high=255 * np.ones((*cam.config.resolution, 3), dtype=np.uint8),
                dtype=np.uint8,
            )
            for cam in self.cameras
            if cam.config.resolution is not None
        }

        self.observation_space = gym.spaces.Dict(
            {**state_spaces, **image_spaces, "task": gym.spaces.Text(max_length=256)}
        )

        self.action_space = gym.spaces.Box(
            low=-np.inf * np.ones((cartesian_dim + 1,), dtype=np.float32),
            high=np.inf * np.ones((cartesian_dim + 1,), dtype=np.float32),
            dtype=np.float32,
        )

    # ── ROS2 callbacks ────────────────────────────────────────────────────────

    def _cb_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_pose_msg = msg

    def _cb_gripper(self, msg: Float32) -> None:
        with self._lock:
            self._last_gripper_width = float(msg.data)

    def _spin(self) -> None:
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self._node)
        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)

    # ── Pose conversion ───────────────────────────────────────────────────────

    def _rotation_dim(self) -> int:
        if self.config.orientation_representation == OrientationRepresentation.QUATERNION:
            return 4
        return 3  # EULER and ANGLE_AXIS are both 3D

    def _pose_msg_to_array(self, msg: PoseStamped) -> np.ndarray:
        """Convert PoseStamped → apply frame transforms → return [pos, rot] array."""
        p = msg.pose.position
        q = msg.pose.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])

        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]

        T_tcp = self._tx_world_correction @ T @ self._tx_body_tcp

        pos = T_tcp[:3, 3].astype(np.float32)
        rot_tcp = Rotation.from_matrix(T_tcp[:3, :3])

        if self.config.orientation_representation == OrientationRepresentation.EULER:
            rot_array = rot_tcp.as_euler("xyz").astype(np.float32)
        elif self.config.orientation_representation == OrientationRepresentation.QUATERNION:
            rot_array = rot_tcp.as_quat().astype(np.float32)
        else:  # ANGLE_AXIS
            rot_array = rot_tcp.as_rotvec().astype(np.float32)

        return np.concatenate([pos, rot_array])

    @property
    def _gripper_normalized(self) -> float:
        """Gripper width in meters normalized to [0, 1]."""
        if self._last_gripper_width is None:
            return 0.0
        return float(
            np.clip(self._last_gripper_width / self.config.max_gripper_width, 0.0, 1.0)
        )

    # ── Gym interface ─────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        obs: dict[str, Any] = {"task": self.task}

        with self._lock:
            pose_msg = self._last_pose_msg

        if pose_msg is not None:
            pose_array = self._pose_msg_to_array(pose_msg)
            if ObservationKeys.CARTESIAN_OBS in self.config.observations_to_include_to_state:
                obs[ObservationKeys.CARTESIAN_OBS] = pose_array
        else:
            if ObservationKeys.CARTESIAN_OBS in self.config.observations_to_include_to_state:
                obs[ObservationKeys.CARTESIAN_OBS] = np.zeros(
                    self.observation_space[ObservationKeys.CARTESIAN_OBS].shape,
                    dtype=np.float32,
                )

        if ObservationKeys.GRIPPER_OBS in self.config.observations_to_include_to_state:
            obs[ObservationKeys.GRIPPER_OBS] = np.array(
                [self._gripper_normalized], dtype=np.float32
            )

        for camera in self.cameras:
            obs[f"{ObservationKeys.IMAGE_OBS}.{camera.config.camera_name}"] = (
                camera.current_image
            )

        return obs

    def current_pose_as_action(self) -> np.ndarray:
        """Return [pos, rot, gripper_normalized] — used as action[t] = pose[t+1]."""
        with self._lock:
            pose_msg = self._last_pose_msg
        if pose_msg is None:
            raise RuntimeError(
                f"No pose received yet. Check topic: {self.config.pose_topic}"
            )
        pose_array = self._pose_msg_to_array(pose_msg)
        gripper = np.array([self._gripper_normalized], dtype=np.float32)
        return np.concatenate([pose_array, gripper])

    def step(
        self, action: np.ndarray | None, block: bool = False
    ) -> Tuple[dict, float, bool, bool, dict]:
        """Return current observation. action is ignored (no robot to command).

        The action parameter exists only to satisfy the gym interface — the recording
        function (make_umi_handheld_fn) builds and stores the action independently.
        """
        self.timestep += 1

        if self.timestep % self.config.log_status_every_n_frames == 0:
            with self._lock:
                pose_msg = self._last_pose_msg
            if pose_msg is not None:
                p = self._pose_msg_to_array(pose_msg)
                logger.info(
                    f"[UMI] pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
                    f"rot=[{p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f}] "
                    f"gripper={self._gripper_normalized:.2f}"
                )

        return self._get_obs(), 0.0, False, False, {}

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[dict, dict]:
        super().reset(seed=seed, options=options)
        self.timestep = 0
        return self._get_obs(), {}

    def close(self) -> None:
        super().close()

    def home(self, **kwargs) -> None:
        """No-op — handheld env has no robot to home."""
        pass

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        """Wait until first pose and gripper messages have been received."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                pose_ok = self._last_pose_msg is not None
                grip_ok = self._last_gripper_width is not None
            if pose_ok and grip_ok:
                for cam in self.cameras:
                    cam.wait_until_ready(timeout=max(1.0, timeout - (time.time() - start)))
                logger.info("UmiHandheldEnv is ready.")
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"Timed out waiting for UmiHandheldEnv. "
            f"Check topics: pose={self.config.pose_topic}, "
            f"gripper={self.config.gripper_width_topic}"
        )

    def get_metadata(self) -> dict:
        from importlib.metadata import version

        return {
            "crisp_gym_version": version("crisp_gym"),
            "env_type": "umi_handheld",
            "env_config": self.config.get_metadata(),
        }

    @classmethod
    def from_yaml(cls, yaml_path: Path | str, **kwargs) -> "UmiHandheldEnv":
        config = UmiHandheldEnvConfig.from_yaml(yaml_path)
        return cls(config=config, **kwargs)
