"""Wire format for the remote policy server — protocol v1.

ONE module, imported by BOTH the client (`remote_policy.py`, robot PC) and the
server (`scripts/policy_server.py`, GPU PC), so the two cannot drift. Depends
on numpy + msgpack only: no torch, no lerobot, no ROS, so it imports anywhere.

WHY v1 BREAKS v0
    The previous format was an untagged request/reply: the client sent
    {"type": "infer", "obs": ...} and read result["actions"] off whatever came
    back next. Nothing identified WHICH request a reply belonged to, and that
    is not cosmetic. After an infer timeout the late reply is still queued, so
    the next recv() returns the PREVIOUS observation's chunk — which the client
    then composes against the base pose it captured for the NEW observation.
    That is exactly the error RemotePolicy caches `_chunk_base` to avoid; its
    own comment puts the cost at "~doubles velocity, tens of mm / tens of
    degrees by step 8". One dropped deadline was enough to cause it silently.

    v1 therefore puts three fields on every frame:

        v    protocol version (int). Unknown -> refuse, both directions.
        t    frame type (str).
        id   uint64, client-monotonic, ECHOED by the server on every reply.

    The client keeps exactly one outstanding id and discards anything else.

THE FRAMES

    client -> server                     server -> client
    ----------------------------------   ----------------------------------
    hello   contract, contract_digest    hello_ok  model_id, observation,
    reset   episode                                action, chunk_size,
    infer   obs, task, obs_t                       normalization, image,
    bye                                            contract_digest
                                         reset_ok
                                         actions   actions, server_t
                                         error     code, message, fatal

HANDSHAKE
    The client sends its contract (from the policy YAML) and a digest of it.
    The server answers with its own, derived from the checkpoint. The client
    compares and REFUSES TO RUN on mismatch — a wrong rotation layout or
    gripper scale does not crash, it moves the robot wrongly. This is the
    `verify_handshake` the configs have always declared and no code performed.

IMAGES
    The server owns the checkpoint, so the server decides: `hello_ok.image`
    carries {"encoding": "raw"|"jpeg", "quality": int, "shape": [H, W, C]} and
    the client obeys. Lossy encoding changes what the model sees relative to
    training, so it is never chosen silently by the client. Raw 800x1280x3 is
    3.07 MB a frame, 46 MB/s at 15 Hz, which saturates 1 GbE.

STALENESS
    `infer` carries `obs_t`, the client's monotonic clock at capture. The two
    clocks are NOT comparable, so the server measures staleness against the
    previous obs_t it saw on this connection, not against its own clock.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1

# client -> server
T_HELLO = "hello"
T_RESET = "reset"
T_INFER = "infer"
T_BYE = "bye"
# server -> client
T_HELLO_OK = "hello_ok"
T_RESET_OK = "reset_ok"
T_ACTIONS = "actions"
T_ERROR = "error"

CLIENT_FRAMES = frozenset({T_HELLO, T_RESET, T_INFER, T_BYE})
SERVER_FRAMES = frozenset({T_HELLO_OK, T_RESET_OK, T_ACTIONS, T_ERROR})

# error codes. `fatal` means the client must stop the episode rather than retry.
E_BAD_VERSION = "bad_version"        # fatal
E_BAD_FRAME = "bad_frame"            # fatal
E_CONTRACT_MISMATCH = "contract_mismatch"  # fatal
E_NOT_READY = "not_ready"            # retryable: reset not acked yet
E_STALE_OBS = "stale_obs"            # retryable: obs older than the budget
E_INFER_FAILED = "infer_failed"      # retryable: the model raised
FATAL_CODES = frozenset({E_BAD_VERSION, E_BAD_FRAME, E_CONTRACT_MISMATCH})


class ProtocolError(RuntimeError):
    """A frame violated the wire format. Never retried — the peer is wrong."""


# ── msgpack codec ────────────────────────────────────────────────────────────
# Arrays travel as a tagged map carrying dtype+shape, so the far side gets the
# array back bit-exact rather than a nested list of Python floats.

_ND = "__ndarray__"


def _default(o: Any):
    if isinstance(o, np.ndarray):
        arr = np.ascontiguousarray(o)
        return {_ND: True, "data": arr.tobytes(), "dtype": str(arr.dtype),
                "shape": list(arr.shape)}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"cannot serialize {type(o)!r} over the wire")


def _object_hook(o: dict):
    if _ND in o:
        return np.frombuffer(o["data"], dtype=np.dtype(o["dtype"])).reshape(o["shape"])
    return o


def pack(frame: dict) -> bytes:
    """Serialize one frame to a binary websocket payload."""
    import msgpack

    return msgpack.packb(frame, default=_default, use_bin_type=True)


def unpack(payload: bytes) -> dict:
    """Deserialize one frame. Raises ProtocolError on anything unusable."""
    import msgpack

    if isinstance(payload, str):
        # A text frame is never valid in v1 — v0 servers sent bare error
        # strings this way, so say so rather than failing obscurely.
        raise ProtocolError(
            f"expected a binary msgpack frame, got text: {payload[:200]!r}. "
            "A v0 server cannot talk to a v1 client."
        )
    try:
        frame = msgpack.unpackb(payload, object_hook=_object_hook, raw=False,
                                strict_map_key=False)
    except Exception as exc:  # noqa: BLE001
        raise ProtocolError(f"undecodable frame: {exc}") from exc
    if not isinstance(frame, dict):
        raise ProtocolError(f"frame is {type(frame).__name__}, expected a map")
    return frame


# ── frame validation ─────────────────────────────────────────────────────────

def validate(frame: dict, *, expect_from: frozenset) -> dict:
    """Check the three mandatory fields and the type's direction.

    Args:
        frame: a decoded frame.
        expect_from: CLIENT_FRAMES on the server, SERVER_FRAMES on the client.

    Raises:
        ProtocolError: wrong version, missing field, or a frame travelling the
            wrong way (which means the peer is confused about its own role).
    """
    for field in ("v", "t", "id"):
        if field not in frame:
            raise ProtocolError(f"frame is missing required field {field!r}: {frame!r}")
    if frame["v"] != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version {frame['v']!r} != {PROTOCOL_VERSION} — refusing. "
            "Both sides must run the same crisp_gym remote_protocol."
        )
    if frame["t"] not in expect_from:
        raise ProtocolError(
            f"frame type {frame['t']!r} cannot arrive from this direction "
            f"(expected one of {sorted(expect_from)})"
        )
    if not isinstance(frame["id"], int) or frame["id"] < 0:
        raise ProtocolError(f"id must be a non-negative int, got {frame['id']!r}")
    return frame


def _frame(t: str, id_: int, **fields) -> dict:
    return {"v": PROTOCOL_VERSION, "t": t, "id": int(id_), **fields}


# client -> server
def hello(id_: int, contract: dict) -> dict:
    """Opening frame: what the CLIENT believes the contract to be."""
    return _frame(T_HELLO, id_, contract=contract, contract_digest=digest(contract))


def reset(id_: int, episode: int | None = None) -> dict:
    return _frame(T_RESET, id_, episode=episode)


def infer(id_: int, obs: dict, task: str | None, obs_t: float) -> dict:
    return _frame(T_INFER, id_, obs=obs, task=task, obs_t=float(obs_t))


def bye(id_: int) -> dict:
    return _frame(T_BYE, id_)


# server -> client
def hello_ok(id_: int, *, model_id: str, observation: dict, action: dict,
             chunk_size: int, normalization: str, image: dict,
             contract: dict) -> dict:
    """Answer to hello: what the SERVER's checkpoint actually requires."""
    return _frame(
        T_HELLO_OK, id_, model_id=model_id, observation=observation,
        action=action, chunk_size=int(chunk_size), normalization=normalization,
        image=image, contract=contract, contract_digest=digest(contract),
    )


def reset_ok(id_: int) -> dict:
    return _frame(T_RESET_OK, id_)


def actions(id_: int, chunk: np.ndarray, server_t: float) -> dict:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2:
        raise ProtocolError(f"an action chunk must be 2-D (steps, dim), got {arr.shape}")
    return _frame(T_ACTIONS, id_, actions=arr, server_t=float(server_t))


def error(id_: int | None, code: str, message: str) -> dict:
    """`fatal` is derived from the code, so the two can never disagree."""
    return _frame(T_ERROR, id_ if id_ is not None else 0, code=code,
                  message=message, fatal=code in FATAL_CODES)


# ── contract digest ──────────────────────────────────────────────────────────

def digest(contract: dict) -> str:
    """Stable sha256 over the contract, for a one-glance handshake compare.

    Key order must not matter, so the mapping is dumped sorted. Only the fields
    that change what the robot DOES are included by the caller — see
    contract_fields().
    """
    blob = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                      default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:32]


def contract_fields(spec: dict) -> dict:
    """The subset of a policy contract that both sides must agree on.

    Deliberately narrow. Timeouts, hostnames and reconnect policy are local
    choices and differing on them is harmless; a rotation layout, a state
    width or a gripper scale is a silent mis-command.
    """
    obs = spec.get("observation") or {}
    act = spec.get("action") or {}
    return {
        "observation": {
            "n_obs_steps": obs.get("n_obs_steps"),
            "keys": [
                {"name": k.get("name"), "shape": list(k.get("shape") or []),
                 "layout": k.get("layout")}
                for k in (obs.get("keys") or [])
            ],
        },
        "action": {
            "dim": act.get("dim"),
            "layout": act.get("layout"),
            "pose_repr": act.get("pose_repr"),
            "chunk_size": act.get("chunk_size"),
            "gripper": act.get("gripper"),
        },
        "normalization": spec.get("normalization"),
    }


def compare_contracts(client: dict, server: dict) -> list[str]:
    """Field-by-field differences, deepest path first. Empty list == agreement."""
    diffs: list[str] = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, "<missing>"), b.get(key, "<missing>"),
                     f"{path}.{key}" if path else key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{path}: client has {len(a)} entries, server has {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif a != b:
            diffs.append(f"{path}: client={a!r}  server={b!r}")

    walk(client, server, "")
    return diffs


# ── image transport ──────────────────────────────────────────────────────────

def encode_image(img: np.ndarray, spec: dict) -> np.ndarray | dict:
    """Encode one HWC uint8 frame per the server's declared image spec.

    Returns the array unchanged for "raw", or a tagged map for "jpeg" so the
    far side knows to decode it. The server decides; see the module docstring.
    """
    encoding = (spec or {}).get("encoding", "raw")
    if encoding == "raw":
        return img
    if encoding == "jpeg":
        import cv2

        quality = int((spec or {}).get("quality", 90))
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ProtocolError("cv2.imencode failed on a camera frame")
        return {"__jpeg__": True, "data": buf.tobytes(), "shape": list(img.shape)}
    raise ProtocolError(f"unknown image encoding {encoding!r} (raw|jpeg)")


def decode_image(payload: np.ndarray | dict) -> np.ndarray:
    """Inverse of encode_image. Raw arrays pass straight through."""
    if isinstance(payload, dict) and payload.get("__jpeg__"):
        import cv2

        arr = np.frombuffer(payload["data"], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ProtocolError("cv2.imdecode failed on a received frame")
        return img[:, :, ::-1]  # BGR -> RGB
    return np.asarray(payload)
