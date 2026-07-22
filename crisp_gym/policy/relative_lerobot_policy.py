"""Local (in-process) deployment of a relative-pose rot6d LeRobot checkpoint.

Deployment counterpart of `scripts/lerobot_relative_pose.py` for the case where
the robot machine's lerobot version MATCHES the training version (0.4.4). Use
this to VERIFY a trained model without standing up the websocket server;
`RemotePolicy` (REMOTE_INFERENCE.md) is the canonical long-term path and this
class deliberately mirrors its division of labor, so switching later changes
only the transport:

  client role (this class, main process):
    - collects an n_obs_steps history of ABSOLUTE observations / raw units,
    - snapshots the TCP pose at OBSERVATION time for each chunk,
    - composes every returned action: T_cmd = T_base(obs time) @ T_rel
      (HANDOFF §1.1 DEPLOY INVARIANT — never the live execution-time pose),
    - converts gripper units, steps the env with ABSOLUTE commands.
  server role (worker subprocess):
    - loads the checkpoint + lerobot pre/post processors (normalization with
      the RECOMPUTED relative stats saved at training time),
    - feeds the obs window through the policy queues, returns one raw-unit
      action chunk per request via `predict_action_chunk`.

Training parity (lerobot 0.4.4 — verified against its modeling_diffusion.py):
  - The 0.4.4 policy consumes the CONCATENATED `observation.state` (OBS_STATE
    queue). Whether that tensor was ABSOLUTE or RELATIVE at training depends
    on the wrapper version that trained the checkpoint:
      * old wrapper (before observation.state conversion): ABSOLUTE state —
        no pose_repr.json next to the checkpoint, or one saying "absolute";
      * fixed wrapper: RELATIVE state (current frame -> identity) — stamped
        "relative_to_last_obs_frame" in pose_repr.json.
    The worker auto-detects this from pose_repr.json (missing => absolute,
    with a loud warning) and, for relative checkpoints, converts the obs
    window server-side (convert_window_state_to_relative — the same role the
    remote policy server owns in REMOTE_INFERENCE.md). Override with the
    `state_input` config param if needed.
  - `observation.state.cartesian` / `.gripper` sub-keys exist in the dataset
    (RecordConfig.to_features) and therefore in the checkpoint's
    input_features/normalizer, but the 0.4.4 diffusion model never reads them.
    They are sent along (converted in relative mode) to satisfy the
    normalizer.
  - `observation.state.rot_wrt_start` is wrapper-generated and NOT in the
    dataset features, hence not in the checkpoint's input_features — it is not
    sent (sending it would hit a normalizer with no stats for the key).
  - Gripper on disk: width_normalized = clip(width_m / reference_width, 0, 1)
    (record_config `gripper.width_normalized`); the env's Gripper.value is
    device-normalized [0,1] — both directions converted here.

Env requirements (see config/envs/*_deploy_umi.yaml):
  - orientation_representation: rotation_6d  (obs cartesian must be 9D)
  - use_relative_actions: false              (this class outputs ABSOLUTE
    commands; the env's own "relative" branch is a decoupled world-frame add,
    NOT the UMI body-frame composition)
  - a camera named to match the trained image key (e.g. "primary").

The module top stays torch/lerobot-free (imports are lazy inside the worker),
matching remote_policy.py, so the pure-numpy helpers are unit-testable
anywhere (tests/test_relative_deploy.py).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

from crisp_gym.policy.policy import Action, Observation, Policy, register_policy

if TYPE_CHECKING:
    from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv

logger = logging.getLogger(__name__)

# Trained/recorded gripper reference scale (UMI convention, HANDOFF §1.5).
DEFAULT_REFERENCE_WIDTH = 0.09


# ── pure-numpy helpers (unit-tested without torch/ROS) ────────────────────────

def gripper_device_to_ref(
    value: float, reference_width: float, device_max_width: float
) -> float:
    """Device-normalized [0,1] -> reference-normalized (recording semantics)."""
    return float(np.clip(value * device_max_width / reference_width, 0.0, 1.0))


def gripper_ref_to_device(
    value: float, reference_width: float, device_max_width: float
) -> float:
    """Reference-normalized model output -> device-normalized [0,1] command."""
    return float(np.clip(value, 0.0, 1.0) * reference_width / device_max_width)


def compose_relative_pose(action9: np.ndarray, T_base: np.ndarray) -> np.ndarray:
    """T_cmd = T_base @ T_rel for a [pos3, rot6d6] relative action -> 4x4.

    T_base MUST be the TCP pose captured when the chunk's observation was
    taken, never the live execution-time pose (HANDOFF §1.1).
    """
    from crisp_gym.util.rot6d import pose9d_to_mat

    return T_base @ pose9d_to_mat(np.asarray(action9[:9], dtype=np.float64))


def convert_window_state_to_relative(frames: List[dict]) -> List[dict]:
    """Server-side obs conversion for RELATIVE-state checkpoints.

    Mirrors RelativePoseDataset.convert_item on a deploy window: the pose dims
    of `observation.state` (and `.cartesian`) of EVERY frame are re-expressed
    relative to the LAST frame — the current observation becomes the identity
    pose [0,0,0, 1,0,0, 0,1,0]. Gripper / extra dims pass through untouched.
    Input frames are not mutated.
    """
    from crisp_gym.util.rot6d import mat_to_pose9d, pose9d_to_mat

    base = np.asarray(frames[-1]["observation.state"][:9], dtype=np.float64)
    T_base_inv = np.linalg.inv(pose9d_to_mat(base))
    converted = []
    for frame in frames:
        out = dict(frame)
        for key in ("observation.state", "observation.state.cartesian"):
            if key in out:
                v = np.asarray(out[key], dtype=np.float64)
                rel = mat_to_pose9d(T_base_inv @ pose9d_to_mat(v[:9]))
                out[key] = np.concatenate([rel, v[9:]]).astype(np.float32)
        converted.append(out)
    return converted


def append_wrt_start_to_window(frames: List[dict], start_pose9: np.ndarray) -> List[dict]:
    """Append the wrt-start rot6d to each frame's observation.state (UMI parity).

    Mirrors the training wrapper: wrt-start is computed from the ABSOLUTE
    frame poses against the episode-start pose (rotation-only, NOISE OFF at
    inference — UMI real_inference_util.py), and appended AFTER all existing
    state dims. Must run BEFORE convert_window_state_to_relative (which only
    rewrites the first 9 dims and passes appended dims through). Input frames
    are not mutated.
    """
    from crisp_gym.util.rot6d import mat_to_pose9d, pose9d_to_mat

    T_start_inv = np.linalg.inv(pose9d_to_mat(np.asarray(start_pose9[:9], np.float64)))
    out_frames = []
    for frame in frames:
        out = dict(frame)
        v = np.asarray(out["observation.state"], dtype=np.float64)
        wrt6 = mat_to_pose9d(T_start_inv @ pose9d_to_mat(v[:9]))[3:9]
        out["observation.state"] = np.concatenate([v, wrt6]).astype(np.float32)
        out_frames.append(out)
    return out_frames


def find_pose_repr(pretrained_path: str) -> dict | None:
    """Load pose_repr.json stamped by lerobot_relative_pose.py, if present.

    The stamp lives at the training output root; pretrained_path is usually
    <output_dir>/checkpoints/<step>/pretrained_model — walk up a few levels.
    """
    import json
    from pathlib import Path

    p = Path(pretrained_path).resolve()
    for candidate in (p, *list(p.parents)[:4]):
        f = candidate / "pose_repr.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
    return None


def target_pose9d_to_euler6d(target9: np.ndarray) -> np.ndarray:
    """9-D [pos3, rot6d] target -> 6-D [pos3, euler_xyz] (legacy state layout).

    Datasets migrated from the old Euler recorder keep the observation.state
    TARGET sub-key in Euler (the migration converts only the cartesian obs);
    a rot6d deploy env emits the target as 9-D, so it must be converted back
    to match the trained layout.
    """
    from scipy.spatial.transform import Rotation

    from crisp_gym.util.rot6d import rot6d_to_mat

    t = np.asarray(target9, dtype=np.float64).reshape(-1)
    euler = Rotation.from_matrix(rot6d_to_mat(t[3:9])).as_euler("xyz")
    return np.concatenate([t[:3], euler]).astype(np.float32)


def build_obs_frame(
    obs_raw: dict,
    reference_width: float,
    device_max_width: float,
    image_keys: List[str] | None = None,
    target_to_euler: bool = False,
) -> dict:
    """One history frame from a raw env observation, in TRAINING units.

    Layout of `observation.state` matches the recorded dataset (UMI robot
    record config): [cartesian9 absolute, gripper reference-normalized].
    """
    cart = np.asarray(obs_raw["observation.state.cartesian"], dtype=np.float32)
    if cart.shape[-1] != 9:
        raise ValueError(
            f"observation.state.cartesian has dim {cart.shape[-1]}, expected 9 "
            "(pos3 + rot6d). Configure the deploy env with "
            "orientation_representation: rotation_6d."
        )
    g_dev = float(np.asarray(obs_raw["observation.state.gripper"]).reshape(-1)[0])
    g_ref = gripper_device_to_ref(g_dev, reference_width, device_max_width)

    # observation.state must reproduce the DATASET's concatenation: every
    # observation.state.* sub-key the env produces, in the env's insertion
    # order — the same order the recording side concatenated them (a UMI
    # record config yields [cartesian9, gripper1]; legacy/promoted-state
    # datasets add joints/target/... — the env config controls which sub-keys
    # exist, and it must match the recording env). The gripper sub-key is
    # unit-converted; everything else passes through.
    frame: dict = {}
    state_parts = []
    for key, value in obs_raw.items():
        if not key.startswith("observation.state.") or key == "observation.state":
            continue
        if key == "observation.state.gripper":
            part = np.array([g_ref], dtype=np.float32)
        elif key == "observation.state.cartesian":
            part = cart
        elif key == "observation.state.target" and target_to_euler:
            part = target_pose9d_to_euler6d(np.asarray(value))
        else:
            part = np.asarray(value, dtype=np.float32).reshape(-1)
        frame[key] = part
        state_parts.append(part.reshape(-1))
    frame["observation.state"] = np.concatenate(state_parts)
    for key, value in obs_raw.items():
        if key.startswith("observation.images"):
            if image_keys is None or key in image_keys:
                if value is None:
                    raise ValueError(
                        f"camera image '{key}' is None — the camera has not "
                        "published a frame yet. Check the topic is streaming "
                        "(ros2 topic hz) and the deploy env camera name/topic "
                        "match what the checkpoint was trained on."
                    )
                frame[key] = np.asarray(value)
    return frame


# ── the policy ────────────────────────────────────────────────────────────────

@register_policy("relative_lerobot_policy")
class RelativeLerobotPolicy(Policy):
    """In-process inference for a checkpoint trained via lerobot_relative_pose.py.

    Args:
        env: Manipulator env (rotation_6d obs, use_relative_actions=false).
        pretrained_path: Path to the checkpoint's `pretrained_model` directory.
        device_max_width: Physical gripper width in meters at Gripper.value=1.0
            (0.140 Robotiq 2F-140, 0.085 2F-85, 0.08 Franka Hand). REQUIRED —
            wrong value silently mis-scales the gripper channel both ways.
        reference_width: Shared UMI reference width used at recording (0.09).
        n_action_steps: Execute at most this many steps of each chunk before
            requesting a new one. None -> the policy config's own value
            (the full returned chunk).
        state_input: What the checkpoint's observation.state input was at
            training. "auto" (default) reads pose_repr.json next to the
            checkpoint (missing => absolute, with a warning — all checkpoints
            trained before the wrapper converted observation.state).
            "absolute" / "relative" / "relative_wrt_start" (16-D UMI parity:
            wrt-start rot6d appended server-side, noise off) force it.
        overrides: Optional lerobot policy-config overrides (as LerobotPolicy).
    """

    def __init__(
        self,
        env: "ManipulatorBaseEnv",
        pretrained_path: str,
        device_max_width: float | None = None,
        reference_width: float = DEFAULT_REFERENCE_WIDTH,
        n_action_steps: int | None = None,
        state_input: str = "auto",
        target_to_euler: bool = False,
        compose_mode: str = "coupled",
        invert_gripper: bool = False,
        log_actions: int = 0,
        overrides: dict | None = None,
    ):
        if compose_mode not in ("coupled", "decoupled"):
            raise ValueError(
                f"compose_mode must be 'coupled' (UMI 'relative', T_base@T_rel) "
                f"or 'decoupled' (UMI 'rel', base-frame position delta), got "
                f"{compose_mode!r}"
            )
        if state_input not in ("auto", "absolute", "relative", "relative_wrt_start"):
            raise ValueError(
                "state_input must be 'auto', 'absolute', 'relative' or "
                f"'relative_wrt_start', got {state_input!r}"
            )
        if device_max_width is None:
            raise ValueError(
                "device_max_width is required (meters at gripper value 1.0): "
                "0.140 for Robotiq 2F-140, 0.085 for 2F-85, 0.08 for Franka "
                "Hand. It must match the value used in the record config."
            )
        use_rel = getattr(env.config, "use_relative_actions", True)
        if use_rel:
            raise ValueError(
                "The deploy env must set use_relative_actions: false — this "
                "policy composes ABSOLUTE commands (T_cmd = T_base @ T_rel); "
                "the env's relative branch is a decoupled world-frame add, not "
                "the UMI body-frame composition. See config/envs/"
                "ur7e_robotiq_deploy_umi.yaml."
            )

        self.env = env
        self.reference_width = float(reference_width)
        self.device_max_width = float(device_max_width)
        self.target_to_euler = bool(target_to_euler)
        self.compose_mode = compose_mode
        self.invert_gripper = bool(invert_gripper)
        self._log_actions = int(log_actions) > 0
        self._action_log_left = int(log_actions)
        self._requested_n_action_steps = n_action_steps

        from multiprocessing import Pipe, Process

        self.parent_conn, child_conn = Pipe()
        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": child_conn,
                "pretrained_path": pretrained_path,
                "state_input": state_input,
                "overrides": overrides or {},
            },
            daemon=True,
        )
        self.inf_proc.start()

        meta = self.parent_conn.recv()
        if isinstance(meta, tuple) and meta[0] == "error":
            raise RuntimeError(f"Inference worker failed to load: {meta[1]}")
        self.meta = meta
        self.n_obs_steps = int(meta["n_obs_steps"])
        chunk_len = int(meta["n_action_steps"])
        self.n_action_steps = (
            min(int(n_action_steps), chunk_len) if n_action_steps else chunk_len
        )
        logger.info(
            f"Relative policy ready: n_obs_steps={self.n_obs_steps}, chunk="
            f"{chunk_len}, executing {self.n_action_steps}/chunk, "
            f"image_keys={meta.get('image_keys')}, "
            f"state_input={meta.get('state_input')}"
        )

        from collections import deque

        self._history: deque = deque(maxlen=self.n_obs_steps)
        self._state_dim_checked = False
        self._chunk: np.ndarray | None = None
        self._chunk_idx = 0
        # Base pose for the CURRENT chunk, captured at observation time
        # (HANDOFF §1.1 DEPLOY INVARIANT; same as RemotePolicy._chunk_base).
        self._chunk_base: np.ndarray | None = None

    # ── client-side geometry ──────────────────────────────────────────────────

    def _current_pose_mat(self) -> np.ndarray:
        """Measured TCP pose as a 4x4 homogeneous matrix."""
        pose = self.env.robot.end_effector_pose
        T = np.eye(4)
        T[:3, :3] = pose.orientation.as_matrix()
        T[:3, 3] = pose.position
        return T

    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        """Relative model action -> absolute env command in the env's units."""
        from scipy.spatial.transform import Rotation

        from crisp_gym.util.rot6d import rot6d_to_mat

        if self.compose_mode == "decoupled":
            # UMI 'rel' (legacy/decoupled): position is a BASE-frame delta
            # (NOT rotated by the current EE orientation), rotation composes
            # in the body frame. Use only if the checkpoint was trained with
            # action_pose_repr: rel rather than relative.
            T_cmd = np.eye(4)
            T_cmd[:3, :3] = (
                rot6d_to_mat(np.asarray(action[3:9])) @ self._chunk_base[:3, :3]
            )
            T_cmd[:3, 3] = np.asarray(action[:3]) + self._chunk_base[:3, 3]
        else:  # coupled (UMI 'relative'): T_cmd = T_base @ T_rel
            T_cmd = compose_relative_pose(action[:9], self._chunk_base)

        pos = T_cmd[:3, 3]
        rot = Rotation.from_matrix(T_cmd[:3, :3])
        rep = self.env.config.orientation_representation
        rep_value = getattr(rep, "value", rep)
        if rep_value == "quaternion":
            rot_arr = rot.as_quat()
        elif rep_value == "angle_axis":
            rot_arr = rot.as_rotvec()
        elif rep_value == "rotation_6d":
            from crisp_gym.util.rot6d import mat_to_rot6d

            rot_arr = mat_to_rot6d(rot.as_matrix())
        else:  # euler
            rot_arr = rot.as_euler("xyz")

        # gripper is the LAST action dim (robust to any pose-dim count).
        # invert_gripper: the env OBSERVATION is 1 - gripper.value but the
        # COMMAND (_set_gripper_action -> set_target) uses value directly. If
        # the model's action-gripper is in the OBSERVATION convention (e.g.
        # datasets where the action gripper == obs gripper at t+1, such as the
        # migrated legacy demos), it must be inverted back before commanding,
        # otherwise the gripper oscillates (obs=0 -> cmd close -> obs=1 ->
        # cmd open -> ...).
        a_grip = float(action[-1])
        if self.invert_gripper:
            a_grip = 1.0 - a_grip
        gripper = gripper_ref_to_device(
            a_grip, self.reference_width, self.device_max_width
        )

        if self._log_actions and self._action_log_left > 0:
            self._action_log_left -= 1
            # (a) MODEL OUTPUT — relative action, EE/body frame (what the
            #     policy predicts): position + rot6d relative to the obs-time
            #     TCP; current frame is identity, so pure translation reads as
            #     motion along the EE axes.
            rel_euler = Rotation.from_matrix(
                rot6d_to_mat(np.asarray(action[3:9]))
            ).as_euler("xyz")
            # (b) CIC INPUT — absolute command, ROBOT-BASE frame (what
            #     env.step feeds robot.set_target and the Cartesian impedance
            #     controller tracks): composed T_cmd = base ∘ rel.
            cmd_euler = rot.as_euler("xyz")
            logger.info(
                "[action] MODEL rel(EE)  pos=%s rot_euler=%s grip=%.4f "
                "-> cmd_grip=%.4f (invert=%s)  (dim=%d)"
                % (np.round(action[:3], 4).tolist(),
                   np.round(rel_euler, 4).tolist(),
                   float(action[-1]), float(gripper), self.invert_gripper,
                   len(action))
            )
            logger.info(
                "[action] CIC   abs(base) pos=%s rot_euler=%s grip=%.4f  "
                "(mode=%s, Δpos_base=%s, |Δ|=%.1fmm)"
                % (np.round(pos, 4).tolist(),
                   np.round(cmd_euler, 4).tolist(),
                   float(gripper), self.compose_mode,
                   np.round(pos - self._chunk_base[:3, 3], 4).tolist(),
                   float(np.linalg.norm(pos - self._chunk_base[:3, 3]) * 1000))
            )

        return np.concatenate([pos, rot_arr, [gripper]]).astype(np.float32)

    def _verify_state_dim(self, frame: dict) -> None:
        """Fail BEFORE inference if the built observation.state cannot match
        the checkpoint's normalizer — a mismatch inside lerobot's normalize
        step is a cryptic 'size of tensor a must match tensor b' error.
        """
        if self._state_dim_checked:
            return
        self._state_dim_checked = True
        expected = self.meta.get("state_dim")
        if not expected:
            return
        wrt_extra = 6 if self.meta.get("state_input") == "relative_wrt_start" else 0
        built = int(frame["observation.state"].shape[-1])
        if built + wrt_extra != int(expected):
            parts = {
                k: int(np.prod(np.asarray(v).shape))
                for k, v in frame.items()
                if k.startswith("observation.state.")
            }
            raise ValueError(
                f"observation.state dim mismatch: checkpoint expects {expected}, "
                f"client built {built}"
                + (f" (+{wrt_extra} wrt-start appended in the worker)" if wrt_extra else "")
                + f". Sub-keys from the env (insertion order): {parts}. "
                "Fix by making the deploy env config produce the SAME state "
                "components (observations_to_include_to_state / sensors) and "
                "representations as the RECORDING env of the training dataset "
                "— compare with the dataset's info.json "
                "features['observation.state']['names']. For datasets migrated "
                "from the Euler recorder, the state's target sub-key stayed "
                "6-D Euler — set target_to_euler: true in the policy config. "
                "If the wrt-start +6 is wrong for this checkpoint, a stale "
                "pose_repr.json from another run may sit in a parent directory "
                "of --path; override with state_input: 'absolute' or 'relative'."
            )

    # ── Policy interface ──────────────────────────────────────────────────────

    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:
        """Observation/action generator driving env.step (recording-loop hook)."""

        def _fn() -> tuple:
            obs_raw: Observation = self.env.get_obs()
            frame = build_obs_frame(
                obs_raw,
                self.reference_width,
                self.device_max_width,
                image_keys=self.meta.get("image_keys"),
                target_to_euler=self.target_to_euler,
            )
            self._verify_state_dim(frame)
            self._history.append(frame)

            if self._chunk is None or self._chunk_idx >= min(
                self.n_action_steps, len(self._chunk)
            ):
                # Snapshot the base pose NOW — the same tick as the newest obs
                # of the window being sent (training base = last obs frame).
                chunk_base = self._current_pose_mat()
                t0 = time.monotonic()
                self.parent_conn.send(("infer", list(self._history)))
                result = self.parent_conn.recv()
                if isinstance(result, tuple) and result[0] == "error":
                    logger.error(f"Inference failed: {result[1]} — holding pose.")
                    return obs_raw, None
                self._chunk = np.asarray(result)
                self._chunk_idx = 0
                self._chunk_base = chunk_base
                logger.debug(
                    f"Chunk {self._chunk.shape} in "
                    f"{(time.monotonic() - t0) * 1e3:.1f} ms"
                )

            action = self._chunk[self._chunk_idx]
            self._chunk_idx += 1
            env_action = self._to_env_action(action)

            try:
                self.env.step(env_action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, env_action

        return _fn

    def reset(self):
        """Clear obs history / chunk state here and policy queues in the worker."""
        self._history.clear()
        self._chunk = None
        self._chunk_idx = 0
        self._chunk_base = None
        self.parent_conn.send(("reset", None))

    def shutdown(self):
        """Stop the inference worker."""
        try:
            self.parent_conn.send(None)
        except Exception:
            pass
        self.inf_proc.join(timeout=10.0)


# ── worker (subprocess: torch + lerobot live only here) ───────────────────────

def inference_worker(conn, pretrained_path: str, overrides: dict,
                     state_input: str = "auto"):  # noqa: ANN001
    """Load the checkpoint; per request, run the obs window through the policy
    queues and return one raw-unit action chunk (numpy (n, dim)).

    Mirrors lerobot_policy.inference_worker's loading path (lerobot 0.4.4:
    processors carry normalization; the stats are the RELATIVE ones recomputed
    at training time and saved with the checkpoint).
    """
    from crisp_gym.util.setup_logger import setup_logging

    setup_logging()
    wlog = logging.getLogger(__name__ + ".worker")

    try:
        import torch
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.utils import populate_queues

        try:
            from lerobot.constants import OBS_IMAGES
        except ImportError:  # older/newer layout
            OBS_IMAGES = "observation.images"
        try:
            from lerobot.constants import ACTION
        except ImportError:
            ACTION = "action"

        from crisp_gym.util.lerobot_features import numpy_obs_to_torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wlog.info(f"[RelInference] Loading {pretrained_path} on {device} ...")

        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
        if train_config.policy is None:
            raise ValueError(f"No policy config in {pretrained_path}")
        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)
        for k, v in (overrides or {}).items():
            wlog.warning(f"[RelInference] Override {k}: "
                         f"{getattr(policy.config, k, None)} -> {v}")
            setattr(policy.config, k, v)
        policy.to(device).eval()
        policy.reset()

        if not hasattr(policy, "predict_action_chunk"):
            raise RuntimeError(
                f"{type(policy).__name__} has no predict_action_chunk (needs "
                "lerobot >= 0.4). Chunk-wise inference is required to keep the "
                "obs-time composition base."
            )

        preprocessor = postprocessor = None
        try:
            from lerobot.policies.factory import make_pre_post_processors

            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config, pretrained_path=pretrained_path
            )
            wlog.info("[RelInference] Using lerobot pre/post processors.")
        except ImportError:
            wlog.info("[RelInference] No processor support; assuming the "
                      "policy normalizes internally.")

        # ── what did this checkpoint train on? (pose_repr.json provenance) ──
        # Three generations:
        #   absolute           — pre-fix checkpoints (no stamp / no fields)
        #   relative           — state converted, 10-D
        #   relative_wrt_start — state converted + wrt-start appended (16-D,
        #                        UMI parity)
        pose_repr = find_pose_repr(pretrained_path)
        if state_input == "auto":
            obs_stamp = (pose_repr or {}).get("observation", {})
            stamped = obs_stamp.get("observation.state", "")
            if str(stamped).startswith("relative"):
                state_mode = (
                    "relative_wrt_start"
                    if obs_stamp.get("state_includes_wrt_start")
                    else "relative"
                )
            else:
                state_mode = "absolute"
                if pose_repr is None:
                    wlog.warning(
                        "[RelInference] No pose_repr.json found next to the "
                        "checkpoint — assuming ABSOLUTE observation.state "
                        "(all checkpoints trained before the wrapper converted "
                        "observation.state). Force with state_input if wrong."
                    )
        else:
            state_mode = state_input
        wlog.info(f"[RelInference] observation.state input mode: {state_mode}")
        wlog.info(
            "[RelInference] checkpoint input_features: "
            f"{list(getattr(policy.config, 'input_features', {}) or {})}"
        )

        image_features = list(getattr(policy.config, "image_features", []) or [])
        state_dim = None
        try:
            in_feats = getattr(policy.config, "input_features", {}) or {}
            if "observation.state" in in_feats:
                state_dim = int(np.prod(in_feats["observation.state"].shape))
        except Exception:
            pass
        meta = {
            "n_obs_steps": int(getattr(policy.config, "n_obs_steps", 1)),
            "n_action_steps": int(getattr(policy.config, "n_action_steps", 1)),
            "image_keys": image_features,
            "policy_type": train_config.policy.type,
            "state_input": state_mode,
            # Total observation.state dim the checkpoint's normalizer expects —
            # the client verifies its built state against this BEFORE the first
            # inference (a mismatch inside the normalizer is cryptic).
            "state_dim": state_dim,
        }
        conn.send(meta)
        wlog.info(f"[RelInference] Ready: {meta}")

        first_infer = True
        episode_start_pose = None  # captured on the first infer after reset

        def _prepare(frame: dict) -> dict:
            batch = numpy_obs_to_torch(frame)
            if preprocessor is not None:
                batch = preprocessor(batch)
            if image_features:
                batch = dict(batch)
                batch[OBS_IMAGES] = torch.stack(
                    [batch[key] for key in image_features], dim=-4
                )
            return batch

        while True:
            msg = conn.recv()
            if msg is None:
                break
            kind, payload = msg
            if kind == "reset":
                policy.reset()
                episode_start_pose = None  # next infer re-captures it
                if preprocessor is not None:
                    preprocessor.reset()
                    postprocessor.reset()
                continue
            if kind != "infer":
                wlog.warning(f"[RelInference] Unknown message {kind!r}")
                continue
            try:
                # Server-side obs conversion (mirrors the training wrapper /
                # the remote server's role). Order matters and mirrors
                # training: wrt-start is computed from the ABSOLUTE poses
                # first, then the window is re-expressed against its LAST
                # frame (which only rewrites dims 0-8, keeping the appended
                # wrt dims intact).
                window = payload
                if state_mode == "relative_wrt_start":
                    if episode_start_pose is None:
                        # Episode start = first obs after reset = the oldest
                        # frame of the first post-reset window (noise OFF).
                        episode_start_pose = np.asarray(
                            window[0]["observation.state"][:9], np.float64
                        ).copy()
                        wlog.info(
                            "[RelInference] Captured episode-start pose for "
                            f"wrt_start: {np.round(episode_start_pose, 4).tolist()}"
                        )
                    window = append_wrt_start_to_window(window, episode_start_pose)
                if state_mode in ("relative", "relative_wrt_start"):
                    window = convert_window_state_to_relative(window)
                if first_infer:
                    # One-shot sanity print: what the policy actually eats.
                    # Relative state => |pos| ~ 0 and rot6d ~ [1,0,0,0,1,0];
                    # absolute state => workspace-scale positions (~0.3-0.8 m).
                    s = np.asarray(window[-1]["observation.state"], np.float64)
                    wlog.info(
                        "[RelInference] CHECK first observation.state fed to "
                        f"the policy ({state_mode} mode): "
                        f"{np.round(s, 4).tolist()}"
                    )
                    wlog.info(
                        f"[RelInference] CHECK |pos| = {np.linalg.norm(s[:3]):.4f} m "
                        f"-> looks {'RELATIVE (near zero)' if np.linalg.norm(s[:3]) < 0.05 else 'ABSOLUTE (workspace scale)'}"
                    )
                    first_infer = False
                with torch.inference_mode():
                    # Deterministic windowing: rebuild the obs queues from the
                    # client's window every request (deques cap at n_obs_steps;
                    # populate_queues pads a short window by repeating the
                    # first frame, matching lerobot's own reset behavior).
                    policy.reset()
                    batch = None
                    for frame in window:
                        batch = _prepare(frame)
                        # The lerobot preprocessor injects an `action` key into
                        # the observation batch; left in, populate_queues pushes
                        # it into the policy's ACTION queue and
                        # predict_action_chunk then torch.stacks a NoneType.
                        # select_action drops it for the same reason — mirror it.
                        batch.pop(ACTION, None)
                        policy._queues = populate_queues(policy._queues, batch)
                    actions = policy.predict_action_chunk(batch)
                    if postprocessor is not None:
                        actions = postprocessor(actions)
                chunk = actions.squeeze(0).to("cpu").numpy()
                conn.send(chunk)
            except Exception as e:  # keep serving; client holds pose
                wlog.exception(f"[RelInference] infer failed: {e}")
                conn.send(("error", str(e)))
    except Exception as e:
        wlog.exception(f"[RelInference] worker died: {e}")
        try:
            conn.send(("error", str(e)))
        except Exception:
            pass
    finally:
        conn.close()
        wlog.info("[RelInference] Worker shut down.")
