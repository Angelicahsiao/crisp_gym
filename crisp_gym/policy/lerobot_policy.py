"""Interface for a Policy interacting in CRISP."""

import json
import logging
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class

try:
    # v0.5.1+
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
except ImportError:
    try:
        # v0.4.x: defined alongside LeRobotDataset
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError:
        # v0.3.x and earlier: re-exported from policies.factory
        from lerobot.policies.factory import LeRobotDatasetMetadata
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import (
    concatenate_state_features,
    get_features,
    numpy_obs_to_torch,
)
from crisp_gym.util.setup_logger import setup_logging

try:
    from lerobot.policies.factory import make_pre_post_processors
    USE_LEROBOT_PROCESSORS = True
    logging.info("Found lerobot pre/post processor support.")
except ImportError:
    USE_LEROBOT_PROCESSORS = False
    logging.warning("No lerobot pre/post processor support found.")


logger = logging.getLogger(__name__)


def _resolve_train_config_path(pretrained_path: str) -> str:
    """Resolve a local checkpoint directory to the train config file path.

    TrainPipelineConfig.from_pretrained() only handles a local path when it
    points directly to a file.  When given a directory it falls through to
    hf_hub_download, which rejects absolute paths (multiple slashes) as
    invalid repo-ids.  This helper detects that case and returns the path to
    the config file inside the directory.

    For HuggingFace repo-ids (e.g. "user/repo") the input is returned as-is.
    """
    p = Path(pretrained_path)
    if not p.is_dir():
        return pretrained_path
    for candidate in ("train_config.json", "config.json"):
        candidate_path = p / candidate
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(
        f"Could not find train_config.json or config.json in {pretrained_path}. "
        "Ensure the checkpoint directory contains a valid training config."
    )


def _apply_umilike_state(obs: dict) -> dict:
    """Overwrite observation.state with concat(cartesian, gripper) in-place.

    Called inside the inference worker when the loaded model was trained with
    umilike state (observation.state = cartesian + gripper).  The individual
    sub-keys remain untouched so the recording side is unaffected.
    """
    cart = np.asarray(obs["observation.state.cartesian"], dtype=np.float32).ravel()
    grip = np.asarray(obs["observation.state.gripper"], dtype=np.float32).ravel()
    obs["observation.state"] = np.concatenate([cart, grip])
    return obs


def _umilike_names_and_dim(env: ManipulatorBaseEnv) -> tuple[list[str], int] | None:
    """Return (names, dim) of the umilike state for this env, or None if unavailable.

    The umilike state is concat(observation.state.cartesian, observation.state.gripper).
    Names and shapes are derived exactly the way the recording pipeline / the
    add_umilike_state_observation.py script produce them, so they can be compared
    against the training dataset's stored observation.state names.
    """
    try:
        env_features = get_features(env, use_video=True)
    except Exception:
        return None

    cart = env_features.get("observation.state.cartesian")
    grip = env_features.get("observation.state.gripper")
    if cart is None or grip is None:
        return None

    names = list(cart["names"]) + list(grip["names"])
    dim = int(np.prod(cart["shape"])) + int(np.prod(grip["shape"]))
    return names, dim


def _detect_umilike(
    policy,
    env: ManipulatorBaseEnv,
    train_config: TrainPipelineConfig,
    logger: logging.Logger,
) -> bool:
    """Return True if the loaded policy expects umilike state.

    Detection order:
      1. Name-based (most reliable): compare the training dataset's stored
         observation.state names against the env's umilike names
         (cartesian + gripper).  Requires the training dataset to be reachable.
      2. Dim-based fallback: compare the model's expected observation.state dim
         against the umilike dim.  Works offline with only the model + env.

    Falls back to the full concatenated state when neither check confirms umilike.
    """
    feat = getattr(policy.config, "robot_state_feature", None)
    if feat is None:
        logger.info("[Inference] Policy has no robot_state_feature; using full state.")
        return False
    expected_dim = int(feat.shape[0])

    umilike = _umilike_names_and_dim(env)
    if umilike is None:
        logger.info(
            "[Inference] cartesian/gripper features unavailable in env; using full state."
        )
        return False
    umilike_names, umilike_dim = umilike

    # 1) Name-based check against the training dataset metadata.
    try:
        meta = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        state_feature = meta.info["features"].get("observation.state", {})
        state_names = state_feature.get("names")
        if state_names is not None:
            if list(state_names) == umilike_names:
                logger.info(
                    f"[Inference] Training observation.state names {list(state_names)} "
                    f"match umilike. Using umilike state."
                )
                return True
            logger.info(
                f"[Inference] Training observation.state names {list(state_names)} "
                f"do not match umilike names {umilike_names}. Using full state."
            )
            return False
        logger.info(
            "[Inference] Training metadata has no observation.state names; "
            "falling back to dim comparison."
        )
    except Exception as e:
        logger.info(
            f"[Inference] Could not read training dataset metadata ({e}); "
            "falling back to dim comparison."
        )

    # 2) Dim-based fallback.
    if expected_dim == umilike_dim:
        logger.info(
            f"[Inference] Model expects {expected_dim}D state = umilike dim "
            f"({umilike_names}). Using umilike state."
        )
        return True

    logger.info(
        f"[Inference] Model expects {expected_dim}D state (umilike would be "
        f"{umilike_dim}D). Using full concatenated state."
    )
    return False


@register_policy("lerobot_policy")
class LerobotPolicy(Policy):
    """A policy implementation that wraps a LeRobot policy for use in CRISP environments.

    This class runs LeRobot policy inference in a separate process and communicates with the
    environment to generate actions based on observations. It is intended for direct use in
    CRISP-based manipulation environments.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict | None = None,
    ):
        """Initialize the policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
        """
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.overrides = overrides if overrides is not None else {}

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env": env,
                "overrides": self.overrides,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:
        """Generate observation and action by communicating with the inference worker."""

        def _fn() -> tuple:
            logger.debug("Requesting action from policy...")
            obs_raw: Observation = self.env.get_obs()

            # Build the full flat state; the inference worker will overwrite
            # observation.state with the umilike vector if the model needs it.
            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
    overrides: dict | None = None,
):
    """Policy inference process.

    Loads the policy on GPU, receives observations via conn, returns actions,
    and exits on None.

    For models trained with umilike state (observation.state = cartesian + gripper),
    the worker automatically rewrites observation.state before calling select_action.
    Models trained with the full concatenated state work unchanged.

    Args:
        conn (Connection): The connection to the parent process.
        pretrained_path (str): Path to the pretrained policy model.
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
        overrides (dict | None): Optional overrides for the policy configuration.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Inference] Could not import third-party plugins for LeRobot. Continuing without them."
        )
    logger.info("[Inference] Starting inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        logger.info(f"[Inference] Loading training config from {pretrained_path}...")
        # from_pretrained only handles a local *file* path, not a directory.
        # Resolve the checkpoint directory to the actual config file.
        train_config_path = _resolve_train_config_path(pretrained_path)
        train_config = TrainPipelineConfig.from_pretrained(train_config_path)
        _check_dataset_metadata(train_config, env, logger)
        logger.info("[Inference] Loaded training config.")
        logger.debug(f"[Inference] Train config: {train_config}")

        if train_config.policy is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                "Please ensure the policy is correctly configured."
            )

        logger.info("[Inference] Loading policy...")
        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)

        for override_key, override_value in (overrides or {}).items():
            logger.warning(
                f"[Inference] Overriding policy config: {override_key} = "
                f"{getattr(policy.config, override_key)} -> {override_value}"
            )
            setattr(policy.config, override_key, override_value)

        logger.info(
            f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
        )
        policy.reset()
        policy.to(device).eval()

        if USE_LEROBOT_PROCESSORS:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config, pretrained_path=pretrained_path
            )

        # Detect whether the model was trained with umilike state.
        # If so, the worker rewrites observation.state before calling select_action.
        use_umilike = _detect_umilike(policy, env, train_config, logger)

        # ── Warm-up ──────────────────────────────────────────────────────────────────────────
        warmup_obs_raw = env.observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        if use_umilike:
            warmup_obs_raw = _apply_umilike_state(warmup_obs_raw)
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
        if USE_LEROBOT_PROCESSORS:
            warmup_obs = preprocessor(warmup_obs)

        logger.info("[Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            import time

            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed_list.append(end - start)

            torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        logger.info(
            f"[Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, "
            f"max={max(elapsed_list) * 1000:.2f}ms, min={min(elapsed_list) * 1000:.2f}ms"
        )
        logger.info("[Inference] Warm-up complete")

        # ── Inference loop ──────────────────────────────────────────────────────────────
        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                logger.info("[Inference] Resetting policy")
                policy.reset()
                if USE_LEROBOT_PROCESSORS:
                    preprocessor.reset()
                    postprocessor.reset()
                continue

            if use_umilike:
                obs_raw = _apply_umilike_state(obs_raw)

            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                if USE_LEROBOT_PROCESSORS:
                    obs = preprocessor(obs)
                action = policy.select_action(obs)
                if USE_LEROBOT_PROCESSORS:
                    action = postprocessor(action)

            logger.debug(f"[Inference] Computed action: {action}")
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Inference] Worker shutting down")


def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env: ManipulatorBaseEnv,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env (ManipulatorBaseEnv): The environment to compare against.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            env_metadata = env.get_metadata()
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Inference] Could not load dataset metadata: {e}")
        logger.info("[Inference] Skipping metadata comparison.")
