"""Config-driven recording: decouple recorded data from the recording method.

A RecordConfig declares WHAT goes into the dataset (observation fields, action
definition, rates, normalization references) independently of HOW the robot is
driven (leader arm, FACTR joints, phone stream, OptiTrack handheld, policy).

The resolved config is the dataset's data contract: it is stamped into
meta/record_config.json and checked at training time before datasets are mixed.

This module is pure Python (numpy + yaml only) — no ROS, no torch — so it can
be imported on any machine. Source providers access the env by duck typing.

See config/recording/record_config_example.yaml for all parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── Pose representation dims ──────────────────────────────────────────────────

POSE_DIMS = {
    "euler": 6,        # x y z rx ry rz
    "angle_axis": 6,   # x y z ax ay az
    "quaternion": 7,   # x y z qx qy qz qw
    "rotation_6d": 9,  # x y z + first two rows of R (UMI/pytorch3d convention)
}

POSE_NAMES = {
    "euler": ["x", "y", "z", "roll", "pitch", "yaw"],
    "angle_axis": ["x", "y", "z", "ax", "ay", "az"],
    "quaternion": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    "rotation_6d": ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)],
}

# Action definitions — a small CLOSED set on purpose. Each has distinct,
# documented semantics; arbitrary user-composed action pipelines are a footgun.
ACTION_DEFINITIONS = (
    # UMI convention: action[t] = measured TCP pose at t+lookahead (absolute).
    # Works for handheld (OptiTrack) and robot (measured EEF) identically.
    "next_tcp_pose",
    # Classic crisp_gym teleop: action[t] = the command vector sent to the
    # robot at t (whatever semantics the driver used — delta or absolute;
    # stamped via `command_semantics` metadata).
    "command",
    # action[t] = measured joint positions at t+lookahead.
    "next_joint_positions",
)


# ── Source provider registry ──────────────────────────────────────────────────
# A provider is a callable (env) -> np.ndarray | image, resolved by name.
# Register new sources with @register_source("my.source").

SOURCE_REGISTRY: Dict[str, Callable] = {}


def register_source(name: str):
    """Decorator registering an observation source provider by name."""

    def _wrap(fn: Callable):
        SOURCE_REGISTRY[name] = fn
        return fn

    return _wrap


def _pose_to_array(pose, representation: str) -> np.ndarray:
    """crisp_py Pose -> array in the requested representation."""
    from crisp_py.utils.geometry import OrientationRepresentation

    return np.asarray(
        pose.to_array(OrientationRepresentation(representation)), dtype=np.float32
    )


@register_source("robot.tcp_pose")
def _src_tcp_pose(env, representation: str = "rotation_6d", **_) -> np.ndarray:
    """Measured TCP pose. Robot envs: robot.end_effector_pose. Handheld env:
    the tracked pose (UmiHandheldEnv exposes it via current_pose_as_action)."""
    if hasattr(env, "robot"):
        return _pose_to_array(env.robot.end_effector_pose, representation)
    if hasattr(env, "current_pose_as_action"):
        # UmiHandheldEnv: [pose..., gripper] with env-config representation.
        # The env's orientation_representation must match `representation`.
        rep_obj = getattr(env.config, "orientation_representation", "")
        env_rep = getattr(rep_obj, "value", str(rep_obj))
        if representation != env_rep:
            raise ValueError(
                f"robot.tcp_pose wants '{representation}' but handheld env is "
                f"configured with '{env_rep}'. Align the env YAML."
            )
        return np.asarray(env.current_pose_as_action()[:-1], dtype=np.float32)
    raise AttributeError("env exposes neither .robot nor a tracked pose")


@register_source("robot.joint_positions")
def _src_joint_positions(env, **_) -> np.ndarray:
    return np.asarray(env.robot.joint_values, dtype=np.float32)


@register_source("robot.joint_velocities")
def _src_joint_velocities(env, **_) -> np.ndarray:
    return np.asarray(env.robot.joint_velocities, dtype=np.float32)


@register_source("robot.joint_efforts")
def _src_joint_efforts(env, **_) -> np.ndarray:
    """Measured joint efforts. Requires has_effort_feedback on the robot
    config and effort values in the JointState messages (else zeros)."""
    return np.asarray(env.robot.current_joint_effort, dtype=np.float32)


@register_source("robot.external_effort")
def _src_external_effort(env, calibration: str | None = None, **_) -> np.ndarray:
    """Gravity-free (external) joint effort: tau_ext = tau_measured - g(q).

    Uses crisp_gym.util.external_effort.ExternalEffortEstimator (Pinocchio),
    built once from the robot's live /robot_description and cached on the env.
    pinocchio is imported LAZILY here so the training/GPU import paths stay
    free of it. Requires has_effort_feedback and a spinning robot node.

    Params (record config):
        calibration: optional path to a JSON file with per-joint
            {"scale": [...], "offset": [...]} from a contact-free fit
            (ExternalEffortEstimator.fit_calibration). Resolved via
            find_config (crisp_gym.config.path), then treated as a literal path.
    """
    est = getattr(env, "_external_effort_estimator", None)
    if est is None:
        from crisp_gym.util.external_effort import ExternalEffortEstimator

        scale = offset = None
        if calibration:
            import json

            from crisp_gym.config.path import find_config

            cal_path = find_config(calibration) or Path(calibration)
            with open(cal_path) as f:
                cal = json.load(f)
            scale = cal.get("scale")
            offset = cal.get("offset")
        est = ExternalEffortEstimator.from_robot(env.robot, scale=scale, offset=offset)
        env._external_effort_estimator = est  # cache: build the model only once
    q = np.asarray(env.robot.joint_values, dtype=float)
    tau = np.asarray(env.robot.current_joint_effort, dtype=float)
    return est.external_effort(q, tau).astype(np.float32)


@register_source("robot.target_pose")
def _src_target_pose(env, representation: str = "rotation_6d", **_) -> np.ndarray:
    """Commanded (target) TCP pose — useful to analyze controller tracking."""
    return _pose_to_array(env.robot.target_pose, representation)


@register_source("gripper.raw_value")
def _src_gripper_raw(env, **_) -> np.ndarray:
    """Uncalibrated device gripper value [0,1] — for debugging width scaling."""
    if hasattr(env, "_gripper_normalized"):
        return np.array([float(env._gripper_normalized)], dtype=np.float32)
    if getattr(env, "gripper", None) is not None:
        return np.array([float(env.gripper.value)], dtype=np.float32)
    return np.zeros(1, dtype=np.float32)


@register_source("gripper.width_normalized")
def _src_gripper(env, reference_width: float | None = None,
                 device_max_width: float | None = None, **_) -> np.ndarray:
    """Gripper opening normalized to [0,1] against a SHARED reference width.

    value_normalized = (device_value * device_max_width) / reference_width

    - Handheld env: device_value is already width/max_gripper_width, so
      device_max_width defaults to env.config.max_gripper_width.
    - Robot envs: Gripper.value is calibrated [0,1] per GripperConfig;
      device_max_width MUST be given (physical width in meters at value=1)
      unless reference scaling is disabled (reference_width null -> raw value).
    """
    if hasattr(env, "_gripper_normalized"):  # UmiHandheldEnv
        raw = float(env._gripper_normalized)
        env_max = float(env.config.max_gripper_width)
        if device_max_width is not None and abs(device_max_width - env_max) > 1e-9:
            raise ValueError(
                f"record config device_max_width ({device_max_width}) != env "
                f"max_gripper_width ({env_max}). The handheld raw value is "
                "normalized by the env's max width, so reconstructing meters "
                "requires these to match — otherwise the gripper channel is "
                "silently mis-scaled."
            )
        dev_max = device_max_width or env_max
    elif getattr(env, "gripper", None) is not None:
        raw = float(env.gripper.value)
        if reference_width is not None and device_max_width is None:
            raise ValueError(
                "gripper.width_normalized on a robot env needs device_max_width "
                "(meters at gripper value 1.0) to unify widths across devices."
            )
        dev_max = device_max_width or 1.0
    else:
        return np.zeros(1, dtype=np.float32)

    if reference_width is None:
        return np.array([raw], dtype=np.float32)
    width_m = raw * dev_max
    return np.array([np.clip(width_m / reference_width, 0.0, 1.0)], dtype=np.float32)


@register_source("camera.image")
def _src_camera(env, camera: str = "", **_):
    cams = getattr(env, "cameras", [])
    for cam in cams:
        if cam.config.camera_name == camera:
            return cam.current_image
    available = [c.config.camera_name for c in cams]
    raise KeyError(
        f"Camera '{camera}' (from the record config) not found in the env's "
        f"cameras {available}. The record config's `camera:` field must match a "
        "`camera_name` in the env config's camera_configs."
    )


# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass
class ObsFieldConfig:
    """One observation entry: dataset key + source provider + its params."""

    key: str                        # e.g. "observation.state.cartesian"
    source: str                     # provider name in SOURCE_REGISTRY
    params: Dict[str, Any] = field(default_factory=dict)
    # For state features: explicit dim (else derived); for images: (H, W, C).
    shape: List[int] | None = None
    include_in_state: bool = True   # concatenated into observation.state

    def resolved_shape(self) -> tuple:
        if self.shape is not None:
            return tuple(self.shape)
        if self.source in ("robot.tcp_pose", "robot.target_pose"):
            return (POSE_DIMS[self.params.get("representation", "rotation_6d")],)
        if self.source in ("gripper.width_normalized", "gripper.raw_value"):
            return (1,)
        raise ValueError(f"shape required for source '{self.source}' (key {self.key})")

    def names(self) -> List[str]:
        if self.source in ("robot.tcp_pose", "robot.target_pose"):
            names = POSE_NAMES[self.params.get("representation", "rotation_6d")]
            if self.source == "robot.target_pose":
                return [f"target_{n}" for n in names]
            return names
        if self.source in ("gripper.width_normalized", "gripper.raw_value"):
            return ["gripper"]
        n = int(np.prod(self.resolved_shape()))
        stem = self.key.split(".")[-1]
        return [f"{stem}_{i}" for i in range(n)]


@dataclass
class ActionConfig:
    """What the `action` column means."""

    definition: str = "next_tcp_pose"     # one of ACTION_DEFINITIONS
    lookahead: int = 1                    # for next_*: action[t] = value[t+lookahead]
    representation: str = "rotation_6d"   # pose representation for next_tcp_pose
    include_gripper: bool = True
    gripper_params: Dict[str, Any] = field(default_factory=dict)  # same as source params
    # for definition == "command": dim of the command vector and a free-text
    # semantics tag stamped into metadata ("delta_pose_euler", "joint_delta", ...)
    command_dim: int | None = None
    command_semantics: str = ""

    def __post_init__(self):
        if self.definition not in ACTION_DEFINITIONS:
            raise ValueError(
                f"action.definition '{self.definition}' not in {ACTION_DEFINITIONS}"
            )
        if self.lookahead < 1 and self.definition.startswith("next_"):
            raise ValueError("lookahead must be >= 1 for next_* action definitions")

    def dim(self, joint_count: int | None = None) -> int:
        if self.definition == "next_tcp_pose":
            base = POSE_DIMS[self.representation]
        elif self.definition == "next_joint_positions":
            if joint_count is None:
                raise ValueError("joint_count required for next_joint_positions")
            base = joint_count
        else:  # command
            if self.command_dim is None:
                raise ValueError("command_dim required for definition 'command'")
            return self.command_dim  # command vector already includes gripper
        return base + (1 if self.include_gripper else 0)

    def names(self, joint_count: int | None = None) -> List[str]:
        if self.definition == "next_tcp_pose":
            names = list(POSE_NAMES[self.representation])
        elif self.definition == "next_joint_positions":
            names = [f"joint_{i}" for i in range(joint_count or 0)]
        else:
            return [f"cmd_{i}" for i in range(self.command_dim or 0)]
        if self.include_gripper:
            names.append("gripper")
        return names


@dataclass
class RecordConfig:
    """Full data contract for a recording session."""

    observations: List[ObsFieldConfig] = field(default_factory=list)
    action: ActionConfig = field(default_factory=ActionConfig)
    rate_hz: float = 15.0
    name: str = "unnamed"

    # ── loading ──
    @classmethod
    def from_yaml(cls, path: Path | str) -> "RecordConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        obs = [
            ObsFieldConfig(
                key=o["key"],
                source=o["source"],
                params={k: v for k, v in o.items()
                        if k not in ("key", "source", "shape", "include_in_state")},
                shape=o.get("shape"),
                include_in_state=o.get("include_in_state", True),
            )
            for o in data.get("observations", [])
        ]
        act = ActionConfig(**data.get("action", {}))
        cfg = cls(
            observations=obs,
            action=act,
            rate_hz=float(data.get("rate_hz", 15.0)),
            name=str(data.get("name", Path(path).stem)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for o in self.observations:
            if o.source not in SOURCE_REGISTRY:
                raise ValueError(
                    f"Unknown source '{o.source}' for key '{o.key}'. "
                    f"Available: {sorted(SOURCE_REGISTRY)}"
                )
        keys = [o.key for o in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate observation keys: {keys}")
        for o in self.observations:
            # LeRobot's dataset_to_policy_features() treats EVERY observation.*
            # key as a policy input, ignoring our include_in_state flag. Extras
            # must live outside that prefix (e.g. "extra.joints") or they leak
            # into the policy and break cross-device mixability.
            if (
                not o.include_in_state
                and o.key.startswith("observation.")
                and not o.key.startswith("observation.images")
            ):
                raise ValueError(
                    f"'{o.key}' has include_in_state: false but uses the "
                    "'observation.' prefix — LeRobot would still feed it to the "
                    "policy as a STATE input. Rename it to an 'extra.' key "
                    "(e.g. 'extra.joints')."
                )

    # ── features (LeRobot schema) ──
    def to_features(self, joint_count: int | None = None,
                    use_video: bool = True) -> Dict[str, Dict]:
        features: Dict[str, Dict] = {}
        state_len, state_names = 0, []
        for o in self.observations:
            if o.key.startswith("observation.images"):
                shape = o.resolved_shape()
                features[o.key] = {
                    "dtype": "video" if use_video else "image",
                    "shape": shape,
                    "names": ["height", "width", "channels"],
                    **({"video_info": {
                        "video.fps": self.rate_hz,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    }} if use_video else {}),
                }
            else:
                shape = o.resolved_shape()
                names = o.names()
                features[o.key] = {"dtype": "float32", "shape": shape, "names": names}
                if o.include_in_state:
                    state_len += int(np.prod(shape))
                    state_names += names
        features["observation.state"] = {
            "dtype": "float32", "shape": (state_len,), "names": state_names,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (self.action.dim(joint_count),),
            "names": self.action.names(joint_count),
        }
        return features

    # ── contract stamping / compatibility ──
    def to_metadata(self) -> dict:
        return {
            "record_config_name": self.name,
            "rate_hz": self.rate_hz,
            "observations": [
                {
                    "key": o.key,
                    "source": o.source,
                    "include_in_state": o.include_in_state,
                    **o.params,
                }
                for o in self.observations
            ],
            "action": {
                "definition": self.action.definition,
                "lookahead": self.action.lookahead,
                "representation": self.action.representation,
                "include_gripper": self.action.include_gripper,
                "command_semantics": self.action.command_semantics,
            },
        }

    # Fields that must match for two datasets to be trained together.
    CONTRACT_FIELDS = ("rate_hz", "action")

    @staticmethod
    def contracts_compatible(meta_a: dict, meta_b: dict) -> bool:
        """Check two stamped record_config metadata dicts for train-mixability."""
        for f in RecordConfig.CONTRACT_FIELDS:
            if meta_a.get(f) != meta_b.get(f):
                logger.error(
                    f"Record contracts differ on '{f}': {meta_a.get(f)} vs {meta_b.get(f)}"
                )
                return False

        # Only POLICY-RELEVANT observations (include_in_state) must match —
        # debug/analysis extras (include_in_state: false, e.g. joint efforts
        # on a robot dataset) do not affect train-mixability; align schemas
        # with scripts/postprocess_align_datasets.py before concatenating.
        # device_max_width is device-specific by design (0.140 Robotiq vs the
        # handheld's own max width) — what must match is reference_width, the
        # shared physical scale. Exclude it from the comparison.
        _DEVICE_SPECIFIC = ("include_in_state", "device_max_width")

        def _state_obs(meta):
            return {
                o["key"]: {k: v for k, v in o.items() if k not in _DEVICE_SPECIFIC}
                for o in meta.get("observations", [])
                if o.get("include_in_state", True)
            }

        a_obs, b_obs = _state_obs(meta_a), _state_obs(meta_b)
        if a_obs.keys() != b_obs.keys():
            logger.error(f"State observation keys differ: {a_obs.keys()} vs {b_obs.keys()}")
            return False
        for k in a_obs:
            if a_obs[k] != b_obs[k]:
                logger.error(f"Observation '{k}' params differ: {a_obs[k]} vs {b_obs[k]}")
                return False
        return True
