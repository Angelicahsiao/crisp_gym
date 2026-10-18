"""Remote policy client: sends observations to a policy server over websocket.

Decouples model inference from the ROS2 machine: the policy (torch, lerobot,
CUDA — all bound to their own Python version) runs on a separate server; this
client only needs `websockets`, `msgpack`, and numpy, which work on any Python
version a ROS2 distro pins.

Speaks protocol v1 — see `remote_protocol.py` for the frame format and
REMOTE_INFERENCE.md for the contract. Every frame carries (v, t, id) and
the client keeps exactly one outstanding id, so a late reply after a
timeout is discarded instead of being composed against the wrong base
pose. v0 (untagged {"type": ...} frames) is not accepted.
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


from crisp_gym.policy import remote_protocol as proto


class WebsocketPolicyClient:
    """Protocol-v1 websocket client. Synchronous; one outstanding request.

    Args:
        uri: e.g. "ws://192.168.1.10:8000".
        connect_timeout: seconds to wait for the socket AND the handshake.
        infer_timeout: seconds to wait for the matching `actions` frame.
        contract: the client's view of the contract (from the policy YAML),
            sent in `hello` and compared against the server's answer.
        verify_handshake: refuse to run when the contracts disagree. Leave it
            on — a wrong rotation layout or gripper scale does not crash, it
            moves the robot wrongly.
        reconnect_attempts: socket reopen attempts, with exponential backoff.
    """

    def __init__(
        self,
        uri: str,
        connect_timeout: float = 10.0,
        infer_timeout: float = 5.0,
        contract: dict | None = None,
        verify_handshake: bool = True,
        reconnect_attempts: int = 3,
    ):
        self.uri = uri
        self.connect_timeout = connect_timeout
        self.infer_timeout = infer_timeout
        self.contract = proto.contract_fields(contract or {})
        self.verify_handshake = verify_handshake
        self.reconnect_attempts = reconnect_attempts

        self._conn = None
        self._next_id = 1
        self.server_meta: dict | None = None
        self.image_spec: dict = {"encoding": "raw"}
        #: Set once `reset` has been acknowledged. `infer` refuses before that,
        #: so a chunk computed from the previous episode's state can never be
        #: executed in this one.
        self._reset_acked = False

    # ── framing ──────────────────────────────────────────────────────────
    def _id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _send(self, frame: dict) -> None:
        self._conn.send(proto.pack(frame))

    def _await(self, want_id: int, want_type: str, timeout: float) -> dict:
        """Read until the reply to `want_id` arrives; drop anything staler.

        The deadline is for the whole call, not per-recv, so a backlog of late
        frames cannot extend it indefinitely.
        """
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(
                    f"no {want_type} for request {want_id} within {timeout:.2f}s"
                )
            frame = proto.validate(
                proto.unpack(self._conn.recv(timeout=left)),
                expect_from=proto.SERVER_FRAMES,
            )
            if frame["t"] == proto.T_ERROR:
                if frame.get("fatal"):
                    raise RuntimeError(
                        f"policy server [{frame.get('code')}]: {frame.get('message')}"
                    )
                if frame["id"] == want_id:
                    raise RuntimeError(
                        f"policy server [{frame.get('code')}]: {frame.get('message')}"
                    )
                logger.warning("dropping stale server error: %s", frame.get("message"))
                continue
            if frame["id"] != want_id:
                # The whole point of v1: a reply to a request we already gave
                # up on. Executing it would compose the PREVIOUS observation's
                # chunk against the CURRENT base pose.
                logger.warning(
                    "dropping stale %s for request %s (waiting for %s)",
                    frame["t"], frame["id"], want_id,
                )
                continue
            if frame["t"] != want_type:
                raise proto.ProtocolError(
                    f"expected {want_type} for request {want_id}, got {frame['t']}"
                )
            return frame

    # ── lifecycle ────────────────────────────────────────────────────────
    def connect(self) -> None:
        from websockets.sync.client import connect

        self._conn = connect(
            self.uri, open_timeout=self.connect_timeout, max_size=None
        )
        hid = self._id()
        self._send(proto.hello(hid, self.contract))
        meta = self._await(hid, proto.T_HELLO_OK, self.connect_timeout)
        self.server_meta = meta
        self.image_spec = meta.get("image") or {"encoding": "raw"}
        self._reset_acked = False

        server_contract = meta.get("contract") or {}
        diffs = proto.compare_contracts(self.contract, server_contract)
        if diffs:
            detail = "\n  ".join(diffs)
            if self.verify_handshake:
                self.close()
                raise RuntimeError(
                    "CONTRACT MISMATCH — refusing to run.\n"
                    f"  client digest {proto.digest(self.contract)}\n"
                    f"  server digest {proto.digest(server_contract)}\n"
                    f"  {detail}\n"
                    "A mismatched layout or scale does not crash; it moves the "
                    "robot wrongly. Fix the policy config or the checkpoint."
                )
            logger.warning("contract mismatch (verify_handshake off):\n  %s", detail)
        logger.info(
            "Connected to %s | model=%s chunk=%s images=%s",
            self.uri, meta.get("model_id"), meta.get("chunk_size"),
            self.image_spec.get("encoding"),
        )

    def _ensure_connected(self) -> None:
        if self._conn is not None:
            return
        delay = 0.5
        last: Exception | None = None
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                self.connect()
                return
            except RuntimeError:
                raise  # a contract mismatch is not worth retrying
            except Exception as exc:  # noqa: BLE001
                last = exc
                logger.warning(
                    "connect to %s failed (%s/%s): %s",
                    self.uri, attempt, self.reconnect_attempts, exc,
                )
                self._conn = None
                if attempt < self.reconnect_attempts:
                    time.sleep(delay)
                    delay *= 2
        raise ConnectionError(f"could not reach {self.uri}: {last}")

    def _drop_connection(self) -> None:
        """Forget a broken socket so the next call reconnects and re-handshakes."""
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001, S110
            pass
        self._conn = None
        self._reset_acked = False

    # ── requests ─────────────────────────────────────────────────────────
    def infer(self, obs: dict, task: str | None = None,
              obs_t: float | None = None) -> np.ndarray:
        """Send one observation, return its action chunk (steps, action_dim)."""
        self._ensure_connected()
        if not self._reset_acked:
            # A server that has not been reset still holds the previous
            # episode's queues; its chunk would be meaningless here.
            self.reset()
        payload = {}
        for key, value in obs.items():
            if key == "task":
                continue  # travels as its own field, not as an obs array
            if key.startswith("observation.images") and value is not None:
                payload[key] = proto.encode_image(np.asarray(value), self.image_spec)
            else:
                payload[key] = value
        rid = self._id()
        try:
            self._send(proto.infer(rid, payload, task,
                                   obs_t if obs_t is not None else time.monotonic()))
            frame = self._await(rid, proto.T_ACTIONS, self.infer_timeout)
        except (TimeoutError, RuntimeError, proto.ProtocolError):
            raise
        except Exception:
            self._drop_connection()
            raise
        actions = np.asarray(frame["actions"])
        return actions[None] if actions.ndim == 1 else actions

    def reset(self, episode: int | None = None) -> None:
        """Clear server-side state and WAIT for the acknowledgement."""
        self._ensure_connected()
        self._reset_acked = False
        rid = self._id()
        self._send(proto.reset(rid, episode))
        self._await(rid, proto.T_RESET_OK, self.infer_timeout)
        self._reset_acked = True

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._send(proto.bye(self._id()))
            except Exception:  # noqa: BLE001, S110
                pass
            self._drop_connection()


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
        uri: str | None = None,
        n_action_steps: int = 8,
        relative_actions: bool = False,
        infer_timeout: float = 5.0,
        connect_timeout: float = 10.0,
        # ── contract blocks, straight from the policy YAML ────────────────
        server: dict | None = None,
        observation: dict | None = None,
        action: dict | None = None,
        normalization: str | None = None,
        reset_on_episode_start: bool = True,
        verify_handshake: bool = True,
        reconnect: dict | None = None,
    ):
        self.env = env
        # The contract configs describe the server as host/port; `uri` is the
        # flat override. Before v1 these nested blocks were rejected outright
        # (__init__ took five scalars), so remote_umi_*.yaml raised TypeError
        # and the documented contract had never actually run.
        server = server or {}
        if uri is None:
            host = server.get("host", "localhost")
            port = server.get("port", 8000)
            scheme = server.get("scheme", "ws")
            uri = f"{scheme}://{host}:{port}"
        reconnect = reconnect or {}
        contract = {
            "observation": observation or {},
            "action": action or {},
            "normalization": normalization,
        }
        self.contract = contract
        self.reset_on_episode_start = bool(reset_on_episode_start)
        # The contract's own action.pose_repr decides composition when it is
        # given; the flat flag stays for remote_policy.yaml, which has no
        # contract block.
        pose_repr = (action or {}).get("pose_repr")
        if pose_repr is not None:
            relative_actions = pose_repr == "relative"
        self.client = WebsocketPolicyClient(
            uri,
            connect_timeout=float(server.get("connect_timeout_s", connect_timeout)),
            infer_timeout=float(server.get("infer_timeout_s", infer_timeout)),
            contract=contract,
            verify_handshake=bool(verify_handshake),
            reconnect_attempts=int(reconnect.get("attempts", 3)),
        )
        self.n_action_steps = int((action or {}).get("n_action_steps", n_action_steps))
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

        from crisp_gym.util.rot6d import pose9d_to_mat

        T_cur = T_base
        T_rel = pose9d_to_mat(np.asarray(action[:9], dtype=np.float64))

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
            from crisp_gym.util.rot6d import mat_to_rot6d

            rot_arr = mat_to_rot6d(rot.as_matrix())
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
                    # obs_t is the capture instant, so the server can drop an
                    # observation that queued too long instead of spending a
                    # GPU slot on it.
                    self._chunk = self.client.infer(
                        obs_raw, task=obs_raw.get("task"), obs_t=t0
                    )
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
        if not self.reset_on_episode_start:
            return
        try:
            self.client.reset()
        except Exception as e:
            # Not fatal here: infer() re-sends the reset and waits for the ack
            # before it will send an observation, so a missed reset delays the
            # first chunk rather than running the episode un-reset.
            logger.warning(f"Failed to reset remote policy: {e}")

    def shutdown(self):
        """Close the websocket connection."""
        self.client.close()
