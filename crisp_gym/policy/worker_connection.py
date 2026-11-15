"""The connection between a policy's robot-side half and its inference worker.

A lerobot deployment has two halves:

    robot side  (RelativeLerobotPolicy)        inference worker (inference_worker)
      observation history, gripper units,        checkpoint + processors,
      state / image checks, command              relative / wrt-start conversion,
      composition, prefetch                      normalization, the model

They only ever talk through a `WorkerConnection`. WHERE the worker runs is the
connection's business and nobody else's:

    local   a `multiprocessing` pipe to a spawned subprocess on this machine.
            The pipe's own `Connection` object already is a WorkerConnection.
    remote  `RemoteWorkerConnection` (remote_worker.py): the same four calls,
            carried over a websocket to `scripts/policy_server.py`, which
            hands them to the same worker over a pipe on the GPU machine.

So there is one robot-side implementation and one worker implementation, and
the choice between local and remote changes neither.

MESSAGES (this is the whole vocabulary)

    worker -> robot side, once, when the checkpoint has loaded
        meta: dict      n_obs_steps, n_action_steps, image_keys, image_shapes,
                        policy_type, state_input, state_dim, rename_map
                        (remote only: action_pose_repr, see RemoteWorkerConnection)
        ("error", str)  instead of meta: loading failed

    robot side -> worker
        ("infer", window)   window = list of observation frames, oldest first,
                            n_obs_steps long, one control tick apart. Poses
                            ABSOLUTE, raw units, images uint8 HWC.
                            Answered by exactly one reply, in order:
                              np.ndarray (steps, action_dim), raw units, or
                              ("error", str) — the caller holds the robot's pose
        ("reset", None)     episode boundary. No reply. Applies before any
                            later "infer" (the connection keeps order).
        None                stop. No reply.

RULES

    * Every "infer" gets exactly one reply, so `recv()` after it must return —
      with ("error", ...) if need be — and never hang. A remote connection
      turns a timeout or a dropped socket into ("error", ...).
    * Replies come back in the order the requests were sent.
    * `poll(0)` never blocks: the prefetch calls it every control tick.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkerConnection(Protocol):
    """What a policy needs from the link to its inference worker.

    `multiprocessing.connection.Connection` and `RemoteWorkerConnection` both
    satisfy it. A new way to reach a worker implements these four calls and
    the message rules in this module's docstring; nothing else has to change.
    """

    def send(self, message: tuple | None) -> None:
        """Queue one message for the worker and return at once."""
        ...

    def recv(self) -> Any:  # noqa: ANN401
        """Block until the worker's next message (meta, a chunk or an error) is here."""
        ...

    def poll(self, timeout: float | None = 0.0) -> bool:
        """Whether recv() would return without blocking; waits up to `timeout`."""
        ...

    def close(self) -> None:
        """Release the connection. The worker itself is stopped with send(None)."""
        ...


def is_error(reply: Any) -> bool:  # noqa: ANN401
    """Whether a reply is the worker's ``("error", message)``."""
    return isinstance(reply, tuple) and len(reply) == 2 and reply[0] == "error"
