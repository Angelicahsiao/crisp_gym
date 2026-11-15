"""The remote `WorkerConnection`: an inference worker on another machine.

`RelativeLerobotPolicy` talks to its inference worker through a
`WorkerConnection` (worker_connection.py — the interface and the message
vocabulary are defined there, once):

    recv()                      -> the worker's metadata (once, after loading)
    send(("infer", window))     -> later recv() gives a chunk, or ("error", msg)
    send(("reset", None))       -> no reply
    send(None)                  -> stop
    poll(timeout)               -> is a reply waiting?  (used by the prefetch)

Locally that is a multiprocessing pipe. `RemoteWorkerConnection` answers the
same calls, but the worker runs behind `scripts/policy_server.py` on another
machine. The window that goes into
`send` is the window the worker receives; the chunk it produces is the chunk
`recv` returns. Nothing about the observation, the relative conversion or the
command composition lives here — this is transport only, so a policy that
works over the pipe behaves the same over the network, apart from latency.

Needs numpy, msgpack and websockets. No torch, no lerobot, no ROS.

What a pipe cannot do and a network can:

* A request can time out or the socket can drop. Either becomes
  ``("error", message)`` for that request — the caller already holds the
  robot's pose on a worker error — and the late answer is discarded by the
  protocol client (it matches replies to requests by id).
* After a dropped socket the client reconnects and RESETS the server, which
  has no memory of the episode. For a wrt-start checkpoint that re-captures
  the episode-start pose from the next window, mid-episode. It is logged as a
  warning; treat a reconnect as the start of a new episode.
"""

from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

from crisp_gym.policy.remote_policy import WebsocketPolicyClient

logger = logging.getLogger(__name__)


class RemoteWorkerConnection:
    """Pipe-like connection to an inference worker behind a policy server.

    Args:
        uri: e.g. "ws://192.168.1.50:8765".
        connect_timeout: seconds for the socket and the handshake.
        infer_timeout: seconds to wait for one chunk before reporting an error.
        reconnect_attempts: socket reopen attempts (exponential backoff).

    Raises:
        RuntimeError: the server answered the handshake without worker
            metadata, i.e. it is not `scripts/policy_server.py`.
    """

    def __init__(self, uri: str, *, connect_timeout: float = 10.0,
                 infer_timeout: float = 5.0, reconnect_attempts: int = 3):
        """Connect, shake hands and start the thread that owns the socket."""
        self.uri = uri
        # contract=None: this side declares none. The caller verifies the
        # worker's metadata against the robot, as it does for a local worker.
        self.client = WebsocketPolicyClient(
            uri, connect_timeout=connect_timeout, infer_timeout=infer_timeout,
            contract=None, reconnect_attempts=reconnect_attempts,
        )
        self.client.connect()
        meta = (self.client.server_meta or {}).get("meta")
        if not isinstance(meta, dict):
            self.client.close()
            raise RuntimeError(
                f"{uri} did not send inference-worker metadata in its handshake. "
                "RemoteWorkerConnection needs scripts/policy_server.py on the other end."
            )
        meta = dict(meta)
        # One fact the robot side needs that only the machine holding the
        # checkpoint can read: whether the model outputs an absolute or a
        # relative pose. A local deployment reads action_repr.json from disk;
        # here the server has read it and says so in its handshake.
        pose_repr = (self.client.server_meta.get("action") or {}).get("pose_repr")
        if pose_repr in ("absolute", "relative"):
            meta.setdefault("action_pose_repr", pose_repr)
        #: The checkpoint the server says it is serving (its path over there).
        self.model_id = self.client.server_meta.get("model_id")
        self._replies: collections.deque = collections.deque([meta])
        self._requests: collections.deque = collections.deque()
        self._cond = threading.Condition()
        self._closed = False
        #: Server-side timing of the last answered request (ms), if reported.
        self.last_timing: dict | None = None
        self._thread = threading.Thread(target=self._run, name="remote-worker", daemon=True)
        self._thread.start()

    # ── the pipe's interface ─────────────────────────────────────────────
    def send(self, message: tuple | None) -> None:
        """Queue a request. Returns at once, like a pipe write.

        `None` is the worker's stop message: here it says goodbye to the server
        (which is then free for the next robot process) and closes the socket.
        The server and its model keep running.
        """
        with self._cond:
            if self._closed:
                raise OSError("the connection to the policy server is closed")
            if message is None:
                self._closed = True
            # The capture instant, for the server's out-of-order check: the
            # caller snapshots its base pose in the same tick it calls send().
            self._requests.append((message, time.monotonic()))
            self._cond.notify_all()

    def recv(self) -> dict | np.ndarray | tuple:
        """Block until a reply is available and return it."""
        with self._cond:
            while not self._replies:
                self._cond.wait()
            return self._replies.popleft()

    def poll(self, timeout: float | None = 0.0) -> bool:
        """Whether recv() would return without blocking (waits up to `timeout`)."""
        with self._cond:
            if not self._replies and timeout:
                self._cond.wait(timeout)
            return bool(self._replies)

    def close(self) -> None:
        """Stop, and wait for the goodbye to be sent. Safe to call twice."""
        try:
            self.send(None)
        except OSError:
            pass  # already stopping
        self._thread.join(timeout=5.0)

    # ── the thread that owns the socket ──────────────────────────────────
    def _reply(self, value: dict | np.ndarray | tuple) -> None:
        with self._cond:
            self._replies.append(value)
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._requests:
                    self._cond.wait()
                message, sent_at = self._requests.popleft()
            if message is None:
                self.client.close()
                return
            try:
                kind, payload = message
            except (TypeError, ValueError):
                logger.warning("ignoring malformed worker message %r", message)
                continue
            if kind == "reset":
                try:
                    self.client.reset()
                except Exception as exc:  # noqa: BLE001
                    # Not lost: infer() re-sends the reset and waits for the
                    # ack before it will send an observation.
                    logger.warning("reset of the policy server failed: %s", exc)
                continue
            if kind != "infer":
                logger.warning("ignoring unknown worker message %r", kind)
                continue
            reconnecting = self.client._conn is None
            try:
                chunk = self.client.infer(payload, obs_t=sent_at)
            except Exception as exc:  # noqa: BLE001
                self._reply(("error", f"{type(exc).__name__}: {exc}"))
                continue
            if reconnecting:
                logger.warning(
                    "reconnected to %s during an episode: the server was reset, so a "
                    "wrt-start checkpoint now measures rotation from THIS observation, "
                    "not from the episode start. Restart the episode.", self.uri,
                )
            self.last_timing = self.client.last_timing
            # A copy: arrays off the wire are read-only, a pipe's are not.
            self._reply(np.array(chunk))
