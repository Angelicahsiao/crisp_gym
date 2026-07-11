"""Remote policy client: sends observations to a policy server over websocket.

Decouples model inference from the ROS2 machine: the policy (torch, lerobot,
CUDA — all bound to their own Python version) runs on a separate server; this
client only needs `websockets`, `msgpack`, and numpy, which work on any Python
version a ROS2 distro pins.

Protocol (openpi-compatible, msgpack-numpy over websocket):
    connect  -> server sends msgpack metadata dict (e.g. {"action_dim": 10})
    request  -> client sends msgpack {"type": "infer"|"reset", "obs": {...}}
    response -> server sends msgpack {"actions": ndarray (chunk, dim)} on
                success, or a plain string on error.

Actions are returned in CHUNKS (the policy's full action horizon). The client
executes `n_action_steps` of the chunk locally before requesting a new one,
hiding network latency. If the policy was trained with relative poses (UMI
style), set `relative_actions=True`; each executed action is then composed
with the robot's current TCP pose: T_cmd = T_current @ T_rel.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np

from crisp_gym.policy.policy import Action, Observation, Policy, register_policy

if TYPE_CHECKING:
    from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv

logger = logging.getLogger(__name__)


def _pack(obj) -> bytes:
    import msgpack

    def _default(o):
        if isinstance(o, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": o.tobytes(),
                b"dtype": str(o.dtype),
                b"shape": list(o.shape),
            }
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"Cannot serialize type {type(o)}")

    return msgpack.packb(obj, default=_default, use_bin_type=True)


def _unpack(data: bytes):
    import msgpack

    def _object_hook(o):
        if isinstance(o, dict) and (b"__ndarray__" in o or "__ndarray__" in o):
            get = lambda k: o.get(k.encode()) if o.get(k.encode()) is not None else o.get(k)  # noqa: E731
            return np.frombuffer(get("data"), dtype=np.dtype(get("dtype"))).reshape(
                get("shape")
            )
        return o

    return msgpack.unpackb(data, object_hook=_object_hook, raw=False, strict_map_key=False)


class WebsocketPolicyClient:
    """Thin synchronous websocket client for a policy server."""

    def __init__(self, uri: str, connect_timeout: float = 10.0, infer_timeout: float = 5.0):
        self.uri = uri
        self.connect_timeout = connect_timeout
        self.infer_timeout = infer_timeout
        self._conn = None
        self.server_metadata: dict = {}

    def connect(self) -> None:
        from websockets.sync.client import connect

        logger.info(f"Connecting to policy server at {self.uri} ...")
        self._conn = connect(
            self.uri, open_timeout=self.connect_timeout, max_size=None
        )
        # Server greets with metadata
        self.server_metadata = _unpack(self._conn.recv(timeout=self.connect_timeout))
        logger.info(f"Connected. Server metadata: {self.server_metadata}")

    def _ensure_connected(self) -> None:
        if self._conn is None:
            self.connect()

    def infer(self, obs: dict) -> np.ndarray:
        """Send one observation, receive an action chunk (n_steps, action_dim)."""
        self._ensure_connected()
        self._conn.send(_pack({"type": "infer", "obs": obs}))
        response = self._conn.recv(timeout=self.infer_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error: {response}")
        result = _unpack(response)
        if isinstance(result, str):
            raise RuntimeError(f"Policy server error: {result}")
        actions = np.asarray(result["actions"])
        if actions.ndim == 1:
            actions = actions[None]
        return actions

    def reset(self) -> None:
        if self._conn is None:
            return
        self._conn.send(_pack({"type": "reset"}))

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


@register_policy("remote_policy")
class RemotePolicy(Policy):
    """Policy that delegates inference to a websocket policy server.

    No torch/lerobot imports — safe to run in any ROS2-pinned Python.

    Args:
        env: The manipulator environment.
        uri: Websocket URI of the policy server, e.g. "ws://192.168.1.10:8000".
        n_action_steps: How many steps of each received action chunk to execute
            before requesting a new chunk.
        relative_actions: If True, treat the pose part of each action as a pose
            relative to the robot's current TCP frame (UMI convention) and
            compose it to an absolute command before env.step. Requires the
            action layout [x, y, z, rot6d(6), gripper] (10D) and an env exposing
            robot.end_effector_pose.
        infer_timeout: Seconds to wait for the server per request.
    """

    def __init__(
        self,
        env: "ManipulatorBaseEnv",
        uri: str = "ws://localhost:8000",
        n_action_steps: int = 8,
        relative_actions: bool = False,
        infer_timeout: float = 5.0,
    ):
        self.env = env
        self.client = WebsocketPolicyClient(uri, infer_timeout=infer_timeout)
        self.n_action_steps = n_action_steps
        self.relative_actions = relative_actions
        self._chunk: np.ndarray | None = None
        self._chunk_idx = 0
        # Base pose for the CURRENT chunk, captured at observation time. Every
        # T_rel in a chunk is relative to the SAME obs frame (training base =
        # last obs timestep; UMI composes the whole chunk against the obs-time
        # ActualTCPPose). Composing per-step against the live pose instead
        # re-adds already-executed motion and compounds within the chunk
        # (~doubles velocity, tens of mm / tens of degrees by step 8).
        self._chunk_base: np.ndarray | None = None

    def _current_pose_mat(self) -> np.ndarray:
        """Measured TCP pose as a 4x4 homogeneous matrix."""
        pose = self.env.robot.end_effector_pose
        T = np.eye(4)
        T[:3, :3] = pose.orientation.as_matrix()
        T[:3, 3] = pose.position
        return T

    def _compose_relative(self, action: np.ndarray, T_base: np.ndarray) -> np.ndarray:
        """T_cmd = T_base @ T_rel for a [pos(3), rot6d(6), grip...] action.

        T_base must be the pose captured when the chunk's observation was
        taken — NOT the live pose at execution time.
        """
        from scipy.spatial.transform import Rotation

        T_cur = T_base

        # rot6d (first two rows) -> matrix via Gram-Schmidt
        a1, a2 = action[3:6], action[6:9]
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        T_rel = np.eye(4)
        T_rel[:3, :3] = np.stack([b1, b2, b3], axis=0)
        T_rel[:3, 3] = action[:3]

        T_cmd = T_cur @ T_rel
        pos = T_cmd[:3, 3]
        rot = Rotation.from_matrix(T_cmd[:3, :3])
        rep = self.env.config.orientation_representation
        rep_value = getattr(rep, "value", rep)
        if rep_value == "quaternion":
            rot_arr = rot.as_quat()
        elif rep_value == "angle_axis":
            rot_arr = rot.as_rotvec()
        elif rep_value == "rotation_6d":
            # First two ROWS of R flattened (UMI/pytorch3d convention) — must
            # match Pose.to_pos_rotation_6d_array / env.action_to_rotation.
            rot_arr = rot.as_matrix()[:2, :].flatten()
        else:  # euler default for cartesian control commands
            rot_arr = rot.as_euler("xyz")
        return np.concatenate([pos, rot_arr, action[9:]])

    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:
        """Generate observation and action by querying the remote policy server."""

        def _fn() -> tuple:
            obs_raw: Observation = self.env.get_obs()

            # Lazy import to keep this module importable without lerobot installed;
            # concatenation itself is pure numpy.
            from crisp_gym.util.lerobot_features import concatenate_state_features

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            # Request a fresh chunk when the current one is exhausted
            if self._chunk is None or self._chunk_idx >= min(
                self.n_action_steps, len(self._chunk)
            ):
                # Snapshot the base pose NOW — the same tick as the observation
                # being sent — before inference latency lets the robot drift.
                # The whole chunk is composed against this one base.
                chunk_base = (
                    self._current_pose_mat() if self.relative_actions else None
                )
                t0 = time.monotonic()
                try:
                    self._chunk = self.client.infer(obs_raw)
                except Exception as e:
                    logger.error(f"Remote inference failed: {e} — holding pose.")
                    return obs_raw, None
                self._chunk_idx = 0
                self._chunk_base = chunk_base
                logger.debug(
                    f"Received chunk {self._chunk.shape} in "
                    f"{(time.monotonic() - t0) * 1e3:.1f} ms"
                )

            action = self._chunk[self._chunk_idx]
            self._chunk_idx += 1

            env_action = (
                self._compose_relative(action, self._chunk_base)
                if self.relative_actions
                else action
            )

            try:
                self.env.step(env_action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, env_action

        return _fn

    def reset(self):
        """Reset the policy state (clears the local chunk and the server-side queue)."""
        self._chunk = None
        self._chunk_idx = 0
        self._chunk_base = None
        try:
            self.client.reset()
        except Exception as e:
            logger.warning(f"Failed to reset remote policy: {e}")

    def shutdown(self):
        """Close the websocket connection."""
        self.client.close()
