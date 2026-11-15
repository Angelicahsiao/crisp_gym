#!/usr/bin/env python3
r"""Remote policy server (GPU machine) — protocol v1.

Serves a lerobot checkpoint to a crisp_gym robot machine over a websocket, so
the robot machine needs neither a GPU nor torch/lerobot, and the GPU machine
needs no ROS.

    robot machine                               GPU machine (this script)
    client half of the policy  ── websocket ──► protocol front (this process)
      observation window, command               │  multiprocessing pipe
      composition, gripper units                ▼
                                                inference_worker (subprocess)
                                                  checkpoint + processors,
                                                  relative conversion,
                                                  predict_action_chunk

The inference worker is `crisp_gym.policy.relative_lerobot_policy.
inference_worker` — the SAME function, unmodified, that `deploy_policy.py`
runs in a local subprocess. This script does not re-implement any of what it
does (which checkpoint generation, the relative / wrt-start conversion, the
uint8 -> [0, 1] image scaling, the queue handling per policy family); it only
puts the wire protocol in front of the pipe. A checkpoint that deploys
in-process deploys through this server with the same numbers.

Any lerobot policy with `predict_action_chunk` is served (diffusion,
multi_task_dit, act, vqbet, groot, ...). Serve with the lerobot version the
checkpoint was trained with.

Needs, on the GPU machine: this repository (a plain clone is enough — the
script puts it on sys.path), torch + lerobot, and `pip install websockets
msgpack rich` (plus opencv for --image-encoding jpeg; lerobot already brings
it). The script checks this at start and names what is missing.

    python scripts/policy_server.py \\
        --path outputs/train/<run>/checkpoints/<step>/pretrained_model \\
        --host 0.0.0.0 --port 8765

Protocol and rules: crisp_gym/policy/remote_protocol.py, REMOTE_INFERENCE.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# A plain clone is enough: no `pip install` of crisp_gym (which would pull the
# ROS-side dependencies) is needed on the GPU machine.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crisp_gym.policy import relative_lerobot_policy as rlp  # noqa: E402
from crisp_gym.policy import remote_protocol as proto  # noqa: E402

logger = logging.getLogger("policy_server")

DEFAULT_PORT = 8765


class InferenceFailed(RuntimeError):
    """The worker raised on this window. The next request may well succeed."""


class WorkerDied(RuntimeError):
    """The worker process is gone. Nothing can be served any more."""


class Worker:
    """The inference worker and the pipe to it.

    Speaks exactly what RelativeLerobotPolicy speaks to its local worker:
    the worker sends its `meta` once it has loaded, then `("infer", window)`
    is answered with a chunk or `("error", message)`, `("reset", None)` has no
    reply, and `None` stops it.

    Args:
        target: the worker function, called as ``target(conn=..., **kwargs)``.
        kwargs: its other arguments (pretrained_path, overrides, ...).
        in_thread: run it in a thread instead of a spawned process. For tests
            with a stand-in worker; the real one initializes CUDA and belongs
            in its own interpreter.
    """

    def __init__(self, target: Callable, kwargs: dict, *, in_thread: bool = False):
        """Start the worker and wait for its metadata (i.e. for the checkpoint to load)."""
        import multiprocessing as mp

        if in_thread:
            self.conn, child = mp.Pipe()
            self.proc = threading.Thread(
                target=target, kwargs={"conn": child, **kwargs}, daemon=True
            )
        else:
            # spawn, not fork: the worker initializes CUDA (see the same note
            # in RelativeLerobotPolicy.__init__).
            ctx = mp.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(
                target=target, kwargs={"conn": child, **kwargs}, daemon=True
            )
        self.proc.start()
        self._lock = threading.Lock()
        self.dead = False
        try:
            meta = self._recv()
        except WorkerDied as exc:
            raise RuntimeError(
                "the inference worker stopped before it finished loading — its own "
                f"error is printed above this line ({exc})"
            ) from exc
        if _is_error(meta):
            raise RuntimeError(f"inference worker failed to load: {meta[1]}")
        self.meta: dict = meta

    def _recv(self) -> Any:  # noqa: ANN401
        """Return the worker's next message.

        Raises WorkerDied instead of waiting forever on a worker that exited
        without answering (an import that fails before its own error handling,
        a crash, an OOM kill).
        """
        while True:
            try:
                if self.conn.poll(0.25):
                    return self.conn.recv()
                if not self.proc.is_alive():
                    if self.conn.poll(0):   # it answered just before exiting
                        return self.conn.recv()
                    raise EOFError("it exited without answering")
            except (EOFError, OSError) as exc:
                self.dead = True
                raise WorkerDied(f"inference worker is gone: {exc}") from exc

    def _send(self, message: tuple | None) -> None:
        try:
            self.conn.send(message)
        except (BrokenPipeError, OSError) as exc:
            self.dead = True
            raise WorkerDied(f"inference worker is gone: {exc!r}") from exc

    def reset(self) -> None:
        """Queue a reset. The worker does not answer it; the pipe keeps it in order."""
        with self._lock:
            self._send(("reset", None))

    def infer(self, window: list) -> np.ndarray:
        """One chunk for one observation window. Raises InferenceFailed / WorkerDied."""
        with self._lock:
            self._send(("infer", window))
            result = self._recv()
        if _is_error(result):
            raise InferenceFailed(str(result[1]))
        return np.asarray(result)

    def stop(self) -> None:
        """Ask the worker to exit and wait for it."""
        try:
            self.conn.send(None)
        except Exception:  # noqa: BLE001, S110
            pass
        self.proc.join(timeout=10.0)


def _is_error(message: Any) -> bool:  # noqa: ANN401
    return isinstance(message, tuple) and len(message) == 2 and message[0] == "error"


# ── what the checkpoint requires, for the handshake ──────────────────────────

def client_state_dim(meta: dict) -> int | None:
    """Width of the `observation.state` the CLIENT sends.

    The worker reports what the policy's normalizer expects. For a
    relative_wrt_start checkpoint that includes the 6 wrt-start dims the worker
    appends itself, so the client sends 6 fewer.
    """
    state_dim = meta.get("state_dim")
    if not state_dim:
        return None
    return int(state_dim) - 6 if meta.get("state_input") == "relative_wrt_start" else int(state_dim)


def checkpoint_spec(pretrained_path: str, meta: dict) -> dict:
    """Observation / action spec derived from the checkpoint and its stamps.

    Never hardcoded: `meta` comes from the loaded policy, the pose layout and
    representation from pose_repr.json / action_repr.json next to the
    checkpoint, the action width from the checkpoint's config.json.
    """
    path = Path(pretrained_path)
    try:
        config = json.loads((path / "config.json").read_text())
    except Exception:  # noqa: BLE001
        config = {}
    pose_repr = rlp.find_pose_repr(pretrained_path) or {}
    action_repr = rlp.find_action_repr(pretrained_path) or {}

    keys = []
    state_dim = client_state_dim(meta)
    if state_dim:
        keys.append({"name": "observation.state", "shape": [state_dim],
                     "dtype": "float32", "layout": pose_repr.get("pose_layout")})
    for key, shape in (meta.get("image_shapes") or {}).items():
        chw = [int(d) for d in shape]
        hwc = [chw[1], chw[2], chw[0]] if len(chw) == 3 else chw
        keys.append({"name": key, "shape": hwc, "dtype": "uint8", "layout": None})

    action_shape = ((config.get("output_features") or {}).get("action") or {}).get("shape") or []
    absolute = (action_repr.get("action") or {}).get("pose_repr") == "absolute"
    stamped_action = pose_repr.get("action") or {}
    return {
        "observation": {
            "n_obs_steps": int(meta.get("n_obs_steps", 1)),
            "keys": keys,
            "state_input": meta.get("state_input"),
            "rename_map": meta.get("rename_map") or {},
        },
        "action": {
            "dim": int(action_shape[0]) if action_shape else None,
            "layout": stamped_action.get("layout"),
            "pose_repr": "absolute" if absolute else stamped_action.get("pose_repr", "relative"),
            "chunk_size": int(meta.get("n_action_steps", 1)),
            "gripper": (pose_repr.get("gripper") or {}).get("semantics"),
        },
        "normalization": "server_side",
    }


def warmup_window(meta: dict) -> list[dict] | None:
    """A neutral observation window, to pay CUDA / first-call costs before a robot waits.

    Identity pose (zero position, rot6d = first two rows of I), zero gripper,
    black images. None when the checkpoint does not say enough to build one.
    """
    state_dim = client_state_dim(meta)
    if not state_dim or state_dim < 9:
        return None
    state = np.zeros(state_dim, dtype=np.float32)
    state[3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    frame = {"observation.state": state, "task": "warmup"}
    for key, shape in (meta.get("image_shapes") or {}).items():
        chw = [int(d) for d in shape]
        if len(chw) != 3:
            return None
        frame[key] = np.zeros((chw[1], chw[2], chw[0]), dtype=np.uint8)
    return [dict(frame) for _ in range(int(meta.get("n_obs_steps", 1)))]


# ── the server ───────────────────────────────────────────────────────────────

class PolicyServer:
    """Protocol v1 in front of one inference worker. One robot at a time.

    Args:
        worker: a started Worker.
        pretrained_path: the checkpoint directory (for the handshake).
        host, port: where to listen. Port 0 picks a free one (see `.port`).
        image_spec: `{"encoding": "raw" | "jpeg", "quality": int}` — the server
            holds the checkpoint, so it decides what the client sends.
    """

    def __init__(self, worker: Worker, pretrained_path: str, host: str = "0.0.0.0",  # noqa: S104
                 port: int = DEFAULT_PORT, image_spec: dict | None = None):
        """Derive the handshake from the checkpoint. Does not listen yet."""
        self.worker = worker
        self.pretrained_path = str(pretrained_path)
        self.host, self._port = host, int(port)
        self.image_spec = dict(image_spec or {"encoding": "raw"})
        self.spec = checkpoint_spec(self.pretrained_path, worker.meta)
        self.contract = proto.contract_fields(self.spec)
        self._busy = threading.Lock()
        self._server = None
        self._thread: threading.Thread | None = None
        self.requests = 0
        self._infer_ms: collections.deque = collections.deque(maxlen=50)
        #: Set when the inference worker died; the server then stops listening.
        self.worker_died = False

    # ── lifecycle ────────────────────────────────────────────────────────
    @property
    def port(self) -> int:
        """The port actually bound (differs from the argument when that was 0)."""
        if self._server is not None:
            return int(self._server.socket.getsockname()[1])
        return self._port

    def _make_server(self):  # noqa: ANN202
        from websockets.sync.server import serve

        kwargs = dict(max_size=None, compression=None)
        try:
            return serve(self._handle, self.host, self._port,
                         process_request=_health_check, **kwargs)
        except TypeError:  # an older websockets without process_request here
            return serve(self._handle, self.host, self._port, **kwargs)

    def start(self) -> "PolicyServer":
        """Listen in a background thread (tests, embedding)."""
        self._server = self._make_server()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        """Listen in this thread until stop() or KeyboardInterrupt."""
        self._server = self._make_server()
        logger.info("listening on ws://%s:%s  (images: %s)", self.host, self.port,
                    self.image_spec.get("encoding"))
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop listening and close the client's socket."""
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── frames ───────────────────────────────────────────────────────────
    def _hello_ok(self, rid: int) -> dict:
        return proto.hello_ok(
            rid,
            model_id=self.pretrained_path,
            observation=self.spec["observation"],
            action=self.spec["action"],
            chunk_size=self.spec["action"]["chunk_size"],
            normalization=self.spec["normalization"],
            image=self.image_spec,
            contract=self.contract,
            meta=self.worker.meta,
        )

    @staticmethod
    def _send(ws, frame: dict) -> None:  # noqa: ANN001
        ws.send(proto.pack(frame))

    @staticmethod
    def _read(ws, timeout: float | None = None) -> dict | None:  # noqa: ANN001
        """One validated client frame; None when nothing arrives within `timeout`."""
        try:
            raw = ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        return proto.validate(proto.unpack(raw), expect_from=proto.CLIENT_FRAMES)

    # ── one connection ───────────────────────────────────────────────────
    def _handle(self, ws) -> None:  # noqa: ANN001
        from websockets.exceptions import ConnectionClosed

        peer = getattr(ws, "remote_address", None)
        if not self._busy.acquire(blocking=False):
            logger.warning("refusing %s: another client is connected", peer)
            try:
                first = self._read(ws, timeout=5.0)
                self._send(ws, proto.error(
                    first["id"] if first else None, proto.E_NOT_READY,
                    "this server is already serving another client"))
            except Exception:  # noqa: BLE001, S110
                pass
            return
        logger.info("client connected: %s", peer)
        try:
            self._serve(ws)
        except ConnectionClosed:
            pass
        except WorkerDied as exc:
            # Nothing can be served any more. Stop listening so the process
            # exits and whoever started it sees that, instead of answering
            # every later request with an error. (From another thread:
            # shutdown() waits for this handler to return.)
            logger.critical("%s — shutting the server down", exc)
            self.worker_died = True
            threading.Thread(target=self._server.shutdown, daemon=True).start()
        except proto.ProtocolError as exc:
            # The peer is wrong, not unlucky: say why, then drop it.
            code = proto.E_BAD_VERSION if "protocol version" in str(exc) else proto.E_BAD_FRAME
            logger.error("protocol error from %s: %s", peer, exc)
            try:
                self._send(ws, proto.error(None, code, str(exc)))
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            self._busy.release()
            logger.info("client disconnected: %s", peer)

    def _serve(self, ws) -> None:  # noqa: ANN001
        greeted = reset_acked = False
        last_obs_t: float | None = None
        waiting: collections.deque = collections.deque()   # frames read ahead
        while True:
            frame = waiting.popleft() if waiting else self._read(ws)
            kind, rid = frame["t"], frame["id"]

            if kind == proto.T_HELLO:
                greeted = True
                reset_acked = False
                self._send(ws, self._hello_ok(rid))
                continue
            if not greeted:
                raise proto.ProtocolError(f"{kind!r} before hello")
            if kind == proto.T_BYE:
                return
            if kind == proto.T_RESET:
                self.worker.reset()
                reset_acked, last_obs_t = True, None
                self._send(ws, proto.reset_ok(rid))
                continue

            # ── infer ────────────────────────────────────────────────────
            if not reset_acked:
                self._send(ws, proto.error(rid, proto.E_NOT_READY,
                                           "infer before reset was acknowledged"))
                continue
            # The client keeps ONE request outstanding, so a frame already
            # waiting behind this one means the client gave up on it (timeout).
            # Its answer would be dropped on arrival: do not spend a GPU slot.
            newer = self._read(ws, timeout=0)
            if newer is not None:
                waiting.append(newer)
                self._send(ws, proto.error(
                    rid, proto.E_STALE_OBS,
                    f"superseded by request {newer['id']} before it was served"))
                continue
            obs_t = frame.get("obs_t")
            if obs_t is not None and last_obs_t is not None and obs_t < last_obs_t:
                # Compared on the CLIENT's clock only (rule 5): an observation
                # older than one already served is out of order.
                self._send(ws, proto.error(
                    rid, proto.E_STALE_OBS,
                    f"obs_t {obs_t:.3f} is older than the last served {last_obs_t:.3f}"))
                continue

            t0 = time.perf_counter()
            window = proto.decode_window(frame.get("obs"), frame.get("task"))
            t1 = time.perf_counter()
            try:
                chunk = self.worker.infer(window)
            except InferenceFailed as exc:
                logger.error("inference failed: %s", exc)
                self._send(ws, proto.error(rid, proto.E_INFER_FAILED, str(exc)))
                continue
            except WorkerDied as exc:
                self._send(ws, proto.error(rid, proto.E_INFER_FAILED, str(exc)))
                raise
            t2 = time.perf_counter()
            if obs_t is not None:
                last_obs_t = obs_t
            timing = {"decode_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3}
            self._send(ws, proto.actions(rid, chunk, time.time(), timing=timing))
            self._note(timing, t0)

    def _note(self, timing: dict, t0: float) -> None:
        self.requests += 1
        self._infer_ms.append((time.perf_counter() - t0) * 1e3)
        if self.requests == 1 or self.requests % 50 == 0:
            logger.info(
                "request %d: decode %.1f ms, inference %.1f ms; last %d requests "
                "took %.1f ms on average (decode + inference + send)",
                self.requests, timing["decode_ms"], timing["infer_ms"],
                len(self._infer_ms), float(np.mean(self._infer_ms)),
            )


def _health_check(connection, request):  # noqa: ANN001, ANN202
    """`GET /healthz` answers 200 OK without opening a websocket."""
    if request.path == "/healthz":
        import http

        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ── command line ─────────────────────────────────────────────────────────────

#: import name -> pip name, for everything the worker and this script import.
#: crisp_gym itself is NOT installed here (a clone is enough), so its own
#: dependencies that the worker touches have to be named: `rich` is one.
REQUIREMENTS = {
    "torch": "torch", "lerobot": "lerobot", "websockets": "websockets",
    "msgpack": "msgpack", "rich": "rich", "yaml": "pyyaml", "gymnasium": "gymnasium",
}


def missing_requirements() -> list[str]:
    """Return the pip names of what this machine lacks.

    Checked before the worker starts, because a worker that cannot import dies
    without a word.
    """
    import importlib.util

    def importable(module: str) -> bool:
        if module in sys.modules:
            return True
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):   # a broken or half-installed package
            return False

    return [pip for module, pip in REQUIREMENTS.items() if not importable(module)]


def _parse_override(text: str) -> tuple[str, object]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def build_parser() -> argparse.ArgumentParser:
    """Command-line options (see REMOTE_INFERENCE.md, section 4)."""
    p = argparse.ArgumentParser(description="Serve a lerobot checkpoint to a crisp_gym robot machine")
    p.add_argument("--path", required=True, help="checkpoint pretrained_model directory")
    p.add_argument("--host", default="0.0.0.0", help="interface to listen on")  # noqa: S104
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--image-encoding", choices=["raw", "jpeg"], default="raw",
                   help="what the client sends. raw is exactly what the policy would see "
                        "in-process; jpeg is ~10x smaller and lossy")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--state-input", default="auto",
                   choices=["auto", "absolute", "relative", "relative_wrt_start"],
                   help="observation.state mode; auto reads pose_repr.json")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="denoising steps (diffusion policies only)")
    p.add_argument("--scheduler", choices=["ddim", "ddpm"], default=None,
                   help="sampler (diffusion policies only)")
    p.add_argument("--set", dest="overrides", action="append", type=_parse_override, default=[],
                   metavar="KEY=VALUE", help="policy config override, repeatable")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the neutral first inference done before listening")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """Load the checkpoint, warm up, serve. Returns the process exit status."""
    args = build_parser().parse_args(argv)
    # force: an imported module may already have configured the root logger.
    logging.basicConfig(level=args.log_level.upper(), force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = str(Path(args.path).expanduser())
    if not (Path(path) / "config.json").exists():
        logger.error("%s has no config.json — pass the checkpoint's pretrained_model directory", path)
        return 2

    missing = missing_requirements()
    if missing:
        logger.error("this Python is missing: %s. Install with:  pip install %s",
                     ", ".join(missing), " ".join(missing))
        return 2

    logger.info("loading %s (this starts the inference worker) ...", path)
    worker = Worker(rlp.inference_worker, {
        "pretrained_path": path,
        "overrides": dict(args.overrides),
        "state_input": args.state_input,
        "num_inference_steps": args.num_inference_steps,
        "scheduler": args.scheduler,
    })
    logger.info("worker ready: %s", worker.meta)

    if not args.no_warmup:
        window = warmup_window(worker.meta)
        if window is None:
            logger.info("no warm-up: the checkpoint does not describe its inputs fully")
        else:
            try:
                t0 = time.perf_counter()
                worker.infer(window)
                t1 = time.perf_counter()
                worker.infer(window)
                logger.info("warm-up done: first inference %.0f ms, second %.0f ms "
                            "(the second is what a robot will wait for)",
                            (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3)
            except InferenceFailed as exc:
                logger.warning("warm-up inference failed (serving anyway): %s", exc)
            worker.reset()

    image_spec = {"encoding": args.image_encoding}
    if args.image_encoding == "jpeg":
        image_spec["quality"] = args.jpeg_quality
    server = PolicyServer(worker, path, host=args.host, port=args.port, image_spec=image_spec)
    # `kill`, systemd and docker stop send SIGTERM. Treat it like Ctrl-C so the
    # worker (and the GPU memory it holds) is released the same way.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("interrupted — shutting down")
    finally:
        server.stop()      # closes the client's socket: the robot side holds pose
        worker.stop()
    return 1 if server.worker_died else 0


def _raise_keyboard_interrupt(signum, frame):  # noqa: ANN001, ARG001
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
