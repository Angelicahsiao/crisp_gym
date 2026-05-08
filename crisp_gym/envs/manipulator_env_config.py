"""General manipulator environment configs."""

import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import yaml
from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_py.gripper.multi_dof_gripper import MultiDofGripperConfig
from crisp_py.robot.robot_config import FrankaConfig, RobotConfig, URConfig, make_robot_config
from crisp_py.sensors.sensor_config import SensorConfig
from crisp_py.utils.geometry import OrientationRepresentation

from crisp_gym.config.path import CRISP_CONFIG_PATH, find_config, list_configs_in_folder
from crisp_gym.util.gripper_mode import GripperMode


class ObservationKeys:
    """Standardized keys for observations in manipulator environments."""

    STATE_OBS = "observation.state"

    GRIPPER_OBS = STATE_OBS + ".gripper"
    JOINT_OBS = STATE_OBS + ".joints"
    CARTESIAN_OBS = STATE_OBS + ".cartesian"
    TARGET_OBS = STATE_OBS + ".target"
    SENSOR_OBS = STATE_OBS + ".sensors"

    IMAGE_OBS = "observation.images"


ALLOWED_STATE_OBS_KEYS = {
    ObservationKeys.GRIPPER_OBS,
    ObservationKeys.JOINT_OBS,
    ObservationKeys.CARTESIAN_OBS,
    ObservationKeys.TARGET_OBS,
    ObservationKeys.SENSOR_OBS,
}


def _load_gripper_config_from_dict(gripper_cfg: dict) -> GripperConfig:
    """Build a GripperConfig (or MultiDofGripperConfig) from a dict.

    Dispatches based on a ``type`` key (currently ``"multi_dof"``).
    """
    cfg = dict(gripper_cfg)
    cfg_type = cfg.pop("type", None)
    if cfg_type == "multi_dof":
        return MultiDofGripperConfig(**cfg)
    return GripperConfig(**cfg)


def _load_gripper_config_from_yaml(yaml_path: Path) -> GripperConfig:
    """Load a GripperConfig (or MultiDofGripperConfig) from a YAML path."""
    with open(yaml_path, "r") as f:
        peek = yaml.safe_load(f) or {}
    if peek.get("type") == "multi_dof":
        return MultiDofGripperConfig.from_yaml(yaml_path)
    return GripperConfig.from_yaml(path=yaml_path.resolve())


@dataclass(kw_only=True)
class ManipulatorEnvConfig(ABC):
    """Manipulator Gym Environment Configuration.

    This class serves as a base configuration for manipulator environments.
    It includes parameters for control frequency, robot configuration,
    gripper configuration, camera configurations, and control parameters.
    """

    control_frequency: float

    # === Core Configurations ===
    robot_config: RobotConfig
    gripper_config: GripperConfig | None
    camera_configs: List[CameraConfig]

    cartesian_control_param_config: Path | None
    joint_control_param_config: Path | None

    sensor_configs: List[SensorConfig] = field(default_factory=lambda: [])

    # Gripper specific configurations
    gripper_mode: GripperMode | str = GripperMode.ABSOLUTE_CONTINUOUS
    gripper_threshold: float = 0.1

    gripper_enabled: bool | None = None  # Deprecated, use gripper_mode instead
    gripper_continuous_control: bool | None = None  # Deprecated, use gripper_mode instead

    # Orientation representation
    orientation_representation: OrientationRepresentation = OrientationRepresentation.EULER

    use_relative_actions: bool = True

    # Safety limits
    min_x: None | float = None
    min_y: None | float = None
    min_z: None | float = 0.0  # e.g., table height
    max_x: None | float = None
    max_y: None | float = None
    max_z: None | float = None

    max_episode_steps: int | None = None

    observations_to_include_to_state: List[str] = field(
        default_factory=lambda: [
            ObservationKeys.CARTESIAN_OBS,
            ObservationKeys.JOINT_OBS,
            ObservationKeys.GRIPPER_OBS,
            ObservationKeys.TARGET_OBS,
        ]
    )

    @property
    def gripper_action_dim(self) -> int:
        """Number of action dimensions occupied by the gripper."""
        if isinstance(self.gripper_config, MultiDofGripperConfig):
            return int(self.gripper_config.num_joints)
        return 1

    def __post_init__(self):
        """Post-initialization checks."""
        if self.gripper_enabled is not None:
            logging.warning(
                "Deprecated: 'gripper_enabled' is deprecated, use 'gripper_mode' instead."
            )
        if self.gripper_continuous_control is not None:
            logging.warning(
                "Deprecated: 'gripper_continuous_control' is deprecated, use 'gripper_mode' instead."
            )

        if isinstance(self.gripper_mode, str):
            self.gripper_mode = GripperMode(self.gripper_mode)

        if isinstance(self.cartesian_control_param_config, str):
            self.cartesian_control_param_config = (
                find_config(self.cartesian_control_param_config)
                or Path(self.cartesian_control_param_config)
            )

        if isinstance(self.joint_control_param_config, str):
            self.joint_control_param_config = (
                find_config(self.joint_control_param_config)
                or Path(self.joint_control_param_config)
            )

        if (
            self.cartesian_control_param_config is not None
            and not self.cartesian_control_param_config.exists()
        ):
            raise FileNotFoundError(
                f"Cartesian control param config file not found: {self.cartesian_control_param_config}"
            )

        if (
            self.joint_control_param_config is not None
            and not self.joint_control_param_config.exists()
        ):
            raise FileNotFoundError(
                f"Joint control param config file not found: {self.joint_control_param_config}"
            )

        if isinstance(self.orientation_representation, str):
            self.orientation_representation = OrientationRepresentation(
                self.orientation_representation
            )

        supported_representations = [
            OrientationRepresentation.EULER,
            OrientationRepresentation.QUATERNION,
            OrientationRepresentation.ANGLE_AXIS,
        ]
        if self.orientation_representation not in supported_representations:
            raise ValueError(
                f"Unsupported orientation representation: {self.orientation_representation}. "
                f"Supported: {supported_representations}"
            )

        self.safety_box = {
            "lower": np.array(
                [
                    self.min_x or -float("inf"),
                    self.min_y or -float("inf"),
                    self.min_z or -float("inf"),
                ]
            ),
            "upper": np.array(
                [
                    self.max_x or float("inf"),
                    self.max_y or float("inf"),
                    self.max_z or float("inf"),
                ]
            ),
        }

    def get_metadata(self) -> dict:
        return {
            "robot_config": self.robot_config.__dict__,
            "gripper_config": self.gripper_config.__dict__ if self.gripper_config else "None",
            "camera_config": [camera.__dict__ for camera in self.camera_configs],
            "sensor_config": [sensor.__dict__ for sensor in self.sensor_configs],
            "gripper_mode": str(self.gripper_mode),
            "gripper_threshold": self.gripper_threshold,
            "cartesian_control_param_config": str(self.cartesian_control_param_config),
            "joint_control_param_config": str(self.joint_control_param_config),
            "orientation_representation": str(self.orientation_representation),
            "use_relative_actions": self.use_relative_actions,
        }

    @classmethod
    def from_yaml(cls, yaml_path: Path, **overrides) -> "ManipulatorEnvConfig":  # noqa: ANN003
        with open(yaml_path, "r") as f:
            original_data = yaml.safe_load(f) or {}

        original_data.update(overrides)

        data = dict(original_data)

        if "robot_config" in data:
            if not isinstance(data["robot_config"], dict):
                raise ValueError("robot_config must be a dictionary in the YAML file.")

            if "from_yaml" in data["robot_config"]:
                robot_yaml_path = find_config(data["robot_config"]["from_yaml"])
                if robot_yaml_path is None:
                    raise FileNotFoundError(
                        f"Robot config file '{data['robot_config']['from_yaml']}' not found in any CRISP config paths"
                    )
                data["robot_config"] = RobotConfig.from_yaml(yaml_path=robot_yaml_path.resolve())
            else:
                data["robot_config"] = make_robot_config(**data["robot_config"])

        if "gripper_config" in data and data["gripper_config"] is not None:
            gripper_cfg = data["gripper_config"]
            if not isinstance(gripper_cfg, dict):
                raise ValueError("gripper_config must be a dictionary in the YAML file.")
            if "from_yaml" in gripper_cfg:
                gripper_yaml_path = find_config(gripper_cfg["from_yaml"])
                if gripper_yaml_path is None:
                    raise FileNotFoundError(
                        f"Gripper config file '{gripper_cfg['from_yaml']}' not found in any CRISP config paths"
                    )
                data["gripper_config"] = _load_gripper_config_from_yaml(gripper_yaml_path)
            else:
                data["gripper_config"] = _load_gripper_config_from_dict(gripper_cfg)

        if "camera_configs" in data and isinstance(data["camera_configs"], list):
            data["camera_configs"] = []
            for camera_cfg in original_data["camera_configs"]:
                if "from_yaml" in camera_cfg:
                    camera_yaml_path = find_config(camera_cfg["from_yaml"])
                    if camera_yaml_path is None:
                        raise FileNotFoundError(
                            f"Camera config file '{camera_cfg['from_yaml']}' not found in any CRISP config paths"
                        )
                    cam_config = CameraConfig.from_yaml(yaml_path=camera_yaml_path.resolve())
                    data["camera_configs"].append(cam_config)
                else:
                    data["camera_configs"].append(
                        CameraConfig(**camera_cfg) if isinstance(camera_cfg, dict) else camera_cfg
                    )

        if "sensor_configs" in data and isinstance(data["sensor_configs"], list):
            data["sensor_configs"] = []
            for sensor_config in original_data["sensor_configs"]:
                if "from_yaml" in sensor_config:
                    sensor_yaml_path = find_config(sensor_config["from_yaml"])
                    if sensor_yaml_path is None:
                        raise FileNotFoundError(
                            f"Sensor config file '{sensor_config['from_yaml']}' not found in any CRISP config paths"
                        )
                    sensor_cfg = SensorConfig.from_yaml(yaml_path=sensor_yaml_path.resolve())
                    data["sensor_configs"].append(sensor_cfg)
                else:
                    data["sensor_configs"].append(
                        SensorConfig(**sensor_config)
                        if isinstance(sensor_config, dict)
                        else sensor_config
                    )

        return cls(**data)


# === Franka Robotics FR3 Environment Configurations ===
@dataclass(kw_only=True)
class FrankaEnvConfig(ManipulatorEnvConfig, ABC):
    """Franka Gym Environment Configuration."""
    control_frequency: float = 30.0

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())

    cartesian_control_param_config: Path | None = field(
        default_factory=lambda: find_config("control/default_cartesian_impedance.yaml")
        or CRISP_CONFIG_PATH / "control" / "default_cartesian_impedance.yaml"
    )
    joint_control_param_config: Path | None = field(
        default_factory=lambda: find_config("control/joint_control.yaml")
        or CRISP_CONFIG_PATH / "control" / "joint_control.yaml"
    )


@dataclass
class NoCamFrankaEnvConfig(FrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])


@dataclass
class LeftNoCamFrankaEnvConfig(NoCamFrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_left.yaml") or CRISP_CONFIG_PATH / "gripper_left.yaml"
            ).resolve()
        )
    )


@dataclass
class RightNoCamFrankaEnvConfig(NoCamFrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_right.yaml") or CRISP_CONFIG_PATH / "gripper_right.yaml"
            ).resolve()
        )
    )


@dataclass
class OnlyWristCamFrankaEnvConfig(FrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(
            min_value=0,
            max_value=1,
            command_topic="gripper/gripper_position_controller/commands",
            joint_state_topic="gripper/joint_states",
            reboot_service="gripper/reboot_gripper",
            enable_torque_service="gripper/dynamixel_hardware_interface/set_dxl_torque",
        )
    )
    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="camera",
                camera_frame="wrist_link",
                resolution=[256, 256],
                camera_color_image_topic="camera/wrist_camera/color/image_rect_raw",
                camera_color_info_topic="camera/wrist_camera/color/camera_info",
            ),
        ]
    )


@dataclass
class RobotiqFrankaEnvConfig(FrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(
            min_value=0,
            max_value=0.7,
            command_topic="robotiq_2f85/robotiq_gripper_controller/gripper_cmd",
            joint_state_topic="robotiq_2f85/joint_states",
            reboot_service="robotiq_2f85/reboot_gripper",
            enable_torque_service="robotiq_2f85/dynamixel_hardware_interface/set_dxl_torque",
        )
    )
    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="camera",
                camera_frame="wrist_link",
                resolution=[256, 256],
                camera_color_image_topic="camera/wrist_camera/color/image_rect_raw",
                camera_color_info_topic="camera/wrist_camera/color/camera_info",
            ),
        ]
    )


@dataclass
class DG3FFrankaEnvConfig(FrankaEnvConfig):
    """Franka + Tesollo Delto DG3F 3-finger gripper."""

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: _load_gripper_config_from_yaml(
            (
                find_config("grippers/gripper_dg3f.yaml")
                or CRISP_CONFIG_PATH / "grippers" / "gripper_dg3f.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])


@dataclass
class AlohaFrankaEnvConfig(FrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_aloha.yaml") or CRISP_CONFIG_PATH / "gripper_aloha.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])
    max_episode_steps: int | None = 1000


@dataclass
class LeftAlohaFrankaEnvConfig(AlohaFrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_left.yaml") or CRISP_CONFIG_PATH / "gripper_left.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="primary",
                camera_frame="primary_link",
                resolution=[256, 256],
                camera_color_image_topic="left_third_person_camera/color/image_raw",
                camera_color_info_topic="left_third_person_camera/color/camera_info",
            ),
            CameraConfig(
                camera_name="wrist",
                camera_frame="wrist_link",
                resolution=[256, 256],
                camera_color_image_topic="left_wrist_camera/color/image_rect_raw",
                camera_color_info_topic="left_wrist_camera/color/camera_info",
            ),
        ]
    )


@dataclass
class RightAlohaFrankaEnvConfig(AlohaFrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_right.yaml") or CRISP_CONFIG_PATH / "gripper_right.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="primary",
                camera_frame="primary_link",
                resolution=[256, 256],
                camera_color_image_topic="right_third_person_camera/color/image_raw",
                camera_color_info_topic="right_third_person_camera/color/camera_info",
            ),
            CameraConfig(
                camera_name="wrist",
                camera_frame="wrist_link",
                resolution=[256, 256],
                camera_color_image_topic="right_wrist_camera/color/image_rect_raw",
                camera_color_info_topic="right_wrist_camera/color/camera_info",
            ),
        ]
    )


@dataclass
class NoCamNoGripperFrankaEnvConfig(FrankaEnvConfig):
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])
    gripper_mode: GripperMode | str = GripperMode.NONE


@dataclass(kw_only=True)
class UREnvConfig(ManipulatorEnvConfig):
    """Universal Robots (UR) Gym Environment Configuration (generic 6-DOF)."""

    control_frequency: float = 50.0

    robot_config: RobotConfig = field(default_factory=lambda: URConfig())
    gripper_config: GripperConfig | None = None
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    gripper_mode: GripperMode | str = GripperMode.NONE

    cartesian_control_param_config: Path | None = field(
        default_factory=lambda: find_config("control/ur_cartesian_impedance.yaml")
        or CRISP_CONFIG_PATH / "control" / "ur_cartesian_impedance.yaml"
    )
    joint_control_param_config: Path | None = field(
        default_factory=lambda: find_config("control/ur_joint_control.yaml")
        or CRISP_CONFIG_PATH / "control" / "ur_joint_control.yaml"
    )


@dataclass
class DG3FUREnvConfig(UREnvConfig):
    """Universal Robots e-series arm + Tesollo Delto DG3F 3-finger gripper.

    The crisp_py ``URConfig`` is generic across the UR e-series
    (UR3e/UR5e/UR10e/UR16e) - all have the same 6 joint names and
    cartesian frame conventions, so a single env config covers them all.
    Note: "UR7e" is not a standard Universal Robots model; pick a real
    UR variant in your launch / URDF stack.
    """

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: _load_gripper_config_from_yaml(
            (
                find_config("grippers/gripper_dg3f.yaml")
                or CRISP_CONFIG_PATH / "grippers" / "gripper_dg3f.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])
    gripper_mode: GripperMode | str = GripperMode.ABSOLUTE_CONTINUOUS


def make_env_config(
    env_type: str,
    config_path: Path | str | None = None,
    **overrides,  # noqa: ANN003
) -> ManipulatorEnvConfig:
    """Factory function to create an environment configuration based on the type."""
    config_class = STRING_TO_CONFIG.get(env_type.lower())
    if config_class is None:
        config_path = find_config("envs/" + env_type.lower() + ".yaml")
        if config_path is None:
            raise ValueError(
                f"Unsupported environment type: {env_type}. The list of supported types are: {list_env_configs()}"
            )
        config_class = ManipulatorEnvConfig

    if config_path:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        return config_class.from_yaml(config_path, **overrides)

    return config_class(**overrides)


def list_env_configs() -> list[str]:
    predefined = list(STRING_TO_CONFIG.keys())
    other = list_configs_in_folder("envs")
    yaml_configs = [file.stem for file in other if file.suffix == ".yaml"]
    return predefined + yaml_configs


STRING_TO_CONFIG = {
    "right_aloha_franka": RightAlohaFrankaEnvConfig,
    "left_aloha_franka": LeftAlohaFrankaEnvConfig,
    "franka": FrankaEnvConfig,
    "robotiq_franka": RobotiqFrankaEnvConfig,
    "dg3f_franka": DG3FFrankaEnvConfig,
    "no_cam_franka": NoCamFrankaEnvConfig,
    "left_no_cam_franka": LeftNoCamFrankaEnvConfig,
    "right_no_cam_franka": RightNoCamFrankaEnvConfig,
    "only_wrist_cam_franka": OnlyWristCamFrankaEnvConfig,
    "no_cam_no_gripper_franka": NoCamNoGripperFrankaEnvConfig,
    "ur": UREnvConfig,
    "dg3f_ur": DG3FUREnvConfig,
}
