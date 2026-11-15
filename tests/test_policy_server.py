"""Tests for scripts/policy_server.py and the network transport (remote_worker.py).

The REAL server, the REAL protocol client and the REAL pipe-like transport talk
over a REAL loopback websocket. Only the model is replaced: a stand-in worker
that speaks the inference worker's pipe protocol (meta once, then a chunk per
`("infer", window)`), run in a thread by `Worker(..., in_thread=True)`.

What these establish:

  * the handshake describes the checkpoint (worker meta, action width from
    config.json, pose representation from the stamps next to it);
  * the window the robot side sends is the window the worker receives —
    bit-exact with raw images — and the chunk comes back unchanged;
  * the protocol rules hold on the server side (reset before infer, one
    client, a superseded request is not served, a worker error is retryable,
    a dead worker stops the server);
  * `RemoteWorkerConnection` behaves like the multiprocessing pipe it stands
    in for, including when the network misbehaves (timeout, server gone);
  * the UNMODIFIED `RelativeLerobotPolicy` control loop, handed that
    connection instead of its pipe, commands the ground-truth trajectory —
    synchronous and with prefetch.

The run against a real lerobot checkpoint is not here (it needs torch and
lerobot); see REMOTE_INFERENCE.md, "Checking a server".

Run: python -m pytest tests/test_policy_server.py -v
"""

import contextlib
import importlib.util
import json
import logging
import sys
import threading
import time
import types
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("msgpack")
pytest.importorskip("websockets")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crisp_gym.policy import relative_lerobot_policy as rlp  # noqa: E402
from crisp_gym.policy import remote_protocol as proto  # noqa: E402
from crisp_gym.policy.remote_policy import WebsocketPolicyClient  # noqa: E402
from crisp_gym.policy.remote_worker import RemoteWorkerConnection  # noqa: E402


def _load_server_module():
    spec = importlib.util.spec_from_file_location(
        "policy_server", REPO / "scripts" / "policy_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = _load_server_module()

CAM = "observation.images.cam"
META = {
    "n_obs_steps": 2, "n_action_steps": 8, "image_keys": [CAM],
    "policy_type": "diffusion", "state_input": "relative_wrt_start",
    "state_dim": 16, "rename_map": {}, "image_shapes": {CAM: (3, 24, 32)},
}


# ── the stand-in for the model ───────────────────────────────────────────────

def chunk_for(window: list, steps: int = 8) -> np.ndarray:
    """A chunk that identifies the window it was computed from."""
    state = np.asarray(window[-1]["observation.state"], np.float32)
    return (state[None, :10] + np.arange(steps, dtype=np.float32)[:, None]).astype(np.float32)


def stand_in_worker(conn, meta, seen, script=None, gate=None):
    """Speaks the inference worker's pipe protocol. `script` maps the n-th
    infer request to a misbehaviour: "raise", "die", "gate", or seconds to sleep."""
    conn.send(meta)
    n = 0
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        kind, payload = message
        seen.append((kind, payload))
        if kind != "infer":
            continue
        n += 1
        step = (script or {}).get(n)
        if step == "raise":
            conn.send(("error", "ValueError: boom"))
            continue
        if step == "die":
            conn.close()
            return
        if step == "gate":
            gate.wait(10.0)
        elif isinstance(step, (int, float)):
            time.sleep(step)
        conn.send(chunk_for(payload, meta["n_action_steps"]))


def _exit_at_once(conn):
    """Module-level so a spawned process can import it."""
    raise SystemExit(3)


def make_checkpoint(root: Path, absolute: bool = False) -> Path:
    """A checkpoint directory with the files the handshake reads, laid out as
    training leaves them: stamps at the run root, config.json in pretrained_model."""
    run = root / "run"
    ckpt = run / "checkpoints" / "000100" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps(
        {"type": "diffusion", "output_features": {"action": {"type": "ACTION", "shape": [10]}}}))
    (run / "pose_repr.json").write_text(json.dumps({
        "pose_layout": "pos3_rot6d_rows",
        "action": {"pose_repr": "relative", "layout": "[x, y, z, rot6d(6), gripper]"},
        "gripper": {"semantics": "clip(width_m / reference_width, 0, 1)"},
    }))
    if absolute:
        (run / "action_repr.json").write_text(json.dumps(
            {"action": {"pose_repr": "absolute", "source": "next_pose"}}))
    return ckpt


@contextlib.contextmanager
def serving(tmp_path, *, meta=None, script=None, image_spec=None, gate=None,
            target=None, extra=None, absolute=False, port=0, server_cls=None):
    seen: list = []
    kwargs = {"meta": dict(meta or META), "seen": seen, **(extra or {})}
    if target is None:
        kwargs.update(script=script, gate=gate)
    worker = ps.Worker(target or stand_in_worker, kwargs, in_thread=True)
    server = (server_cls or ps.PolicyServer)(
        worker, str(make_checkpoint(tmp_path, absolute)),
        host="127.0.0.1", port=port, image_spec=image_spec,
    ).start()
    server.seen = seen
    server.uri = f"ws://127.0.0.1:{server.port}"
    try:
        yield server
    finally:
        server.stop()
        worker.stop()


def make_window(seed: int = 0, n: int = 2, task: str | None = "open the door") -> list:
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        frame = {
            "observation.state": rng.normal(size=10).astype(np.float32),
            CAM: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8),
        }
        if task is not None:
            frame["task"] = task
        frames.append(frame)
    return frames


def infers(server) -> list:
    return [payload for kind, payload in server.seen if kind == "infer"]


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


class Raw:
    """A bare websocket speaking protocol frames, for what the client hides."""

    def __init__(self, server):
        from websockets.sync.client import connect

        self.ws = connect(server.uri, max_size=None).__enter__()

    def send(self, frame: dict) -> None:
        self.ws.send(proto.pack(frame))

    def recv(self, timeout: float = 5.0) -> dict:
        return proto.unpack(self.ws.recv(timeout=timeout))

    def ask(self, frame: dict) -> dict:
        self.send(frame)
        return self.recv()

    def close(self) -> None:
        self.ws.close()


# ── 1. the handshake describes the checkpoint ────────────────────────────────

def test_the_handshake_describes_the_checkpoint(tmp_path):
    with serving(tmp_path) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        hello = client.server_meta
        client.close()

    # the worker's own metadata, as a msgpack round trip leaves it
    assert hello["meta"] == {**META, "image_shapes": {CAM: [3, 24, 32]}}
    assert hello["chunk_size"] == 8
    assert hello["image"] == {"encoding": "raw"}
    assert hello["normalization"] == "server_side"
    action = hello["action"]
    assert action["dim"] == 10                       # from config.json
    assert action["pose_repr"] == "relative"         # from pose_repr.json
    assert action["layout"] == "[x, y, z, rot6d(6), gripper]"
    assert action["gripper"] == "clip(width_m / reference_width, 0, 1)"
    keys = {k["name"]: k for k in hello["observation"]["keys"]}
    # The CLIENT sends 10 numbers: the worker appends the 6 wrt-start dims.
    assert keys["observation.state"]["shape"] == [10]
    assert keys["observation.state"]["layout"] == "pos3_rot6d_rows"
    assert keys[CAM]["shape"] == [24, 32, 3]         # HWC, as the robot sends it
    assert keys[CAM]["dtype"] == "uint8"
    assert hello["observation"]["n_obs_steps"] == 2
    assert hello["observation"]["state_input"] == "relative_wrt_start"
    assert hello["contract_digest"] == proto.digest(hello["contract"])


def test_an_absolute_checkpoint_is_announced_as_absolute(tmp_path):
    with serving(tmp_path, absolute=True) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        assert client.server_meta["action"]["pose_repr"] == "absolute"
        client.close()


def test_the_client_state_width_excludes_what_the_worker_appends():
    assert ps.client_state_dim({"state_dim": 16, "state_input": "relative_wrt_start"}) == 10
    assert ps.client_state_dim({"state_dim": 10, "state_input": "relative"}) == 10
    assert ps.client_state_dim({"state_dim": 10, "state_input": "absolute"}) == 10
    assert ps.client_state_dim({"state_input": "relative"}) is None


def test_the_warm_up_window_matches_what_a_robot_sends():
    window = ps.warmup_window(META)
    assert len(window) == 2
    state = window[-1]["observation.state"]
    assert state.shape == (10,) and state.dtype == np.float32
    np.testing.assert_array_equal(state[3:9], [1, 0, 0, 0, 1, 0])   # identity rot6d rows
    assert window[-1][CAM].shape == (24, 32, 3) and window[-1][CAM].dtype == np.uint8
    assert ps.warmup_window({"n_obs_steps": 2}) is None             # not enough to build one


def test_healthz_answers_without_a_websocket(tmp_path):
    with serving(tmp_path) as server:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=5) as r:  # noqa: S310
            assert r.status == 200
            assert r.read().strip() == b"OK"


# ── 2. what goes in is what the worker gets; what it returns comes back ───────

def test_a_raw_window_reaches_the_worker_bit_exact(tmp_path):
    window = make_window(seed=3)
    with serving(tmp_path) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        chunk = client.infer(window)
        client.close()
        (received,) = infers(server)

    assert len(received) == len(window) == 2
    for sent, got in zip(window, received, strict=True):
        assert set(got) == set(sent)
        assert got["task"] == "open the door"
        for key in ("observation.state", CAM):
            assert got[key].dtype == sent[key].dtype
            assert got[key].shape == sent[key].shape
            np.testing.assert_array_equal(got[key], sent[key])
            assert got[key].flags.writeable      # as over a local pipe
    np.testing.assert_array_equal(chunk, chunk_for(window))
    assert chunk.dtype == np.float32 and chunk.shape == (8, 10)


def test_the_reply_reports_where_the_server_spent_the_request(tmp_path):
    with serving(tmp_path) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        client.infer(make_window())
        timing = client.last_timing
        client.close()
    assert set(timing) == {"decode_ms", "infer_ms"}
    assert all(v >= 0 for v in timing.values())


def test_a_single_frame_is_a_window_of_one(tmp_path):
    frame = make_window(n=1)[0]
    with serving(tmp_path) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        client.infer(frame, task="open the door")
        client.close()
        (received,) = infers(server)
    assert len(received) == 1
    assert received[0]["task"] == "open the door"
    np.testing.assert_array_equal(received[0][CAM], frame[CAM])


def test_jpeg_images_arrive_close_but_not_exact(tmp_path):
    pytest.importorskip("cv2")
    yy, xx = np.mgrid[0:24, 0:32]
    smooth = np.stack([yy * 10, xx * 7, (yy + xx) * 4], axis=-1).astype(np.uint8)
    window = make_window()
    for frame in window:
        frame[CAM] = smooth
    with serving(tmp_path, image_spec={"encoding": "jpeg", "quality": 95}) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        assert client.image_spec["encoding"] == "jpeg"   # the server decides
        client.infer(window)
        client.close()
        (received,) = infers(server)
    got = received[-1][CAM]
    assert got.shape == smooth.shape and got.dtype == np.uint8
    assert got.flags.c_contiguous                         # torch.from_numpy needs this
    assert np.abs(got.astype(int) - smooth.astype(int)).mean() < 4.0
    # state is never lossy
    np.testing.assert_array_equal(received[-1]["observation.state"],
                                  window[-1]["observation.state"])


# ── 3. protocol rules, server side ───────────────────────────────────────────

def test_infer_before_reset_is_refused(tmp_path):
    with serving(tmp_path) as server:
        raw = Raw(server)
        assert raw.ask(proto.hello(0, {}))["t"] == proto.T_HELLO_OK
        obs = proto.encode_window(make_window(), {"encoding": "raw"})
        reply = raw.ask(proto.infer(1, obs, None, 1.0))
        assert (reply["t"], reply["code"], reply["id"]) == (proto.T_ERROR, proto.E_NOT_READY, 1)
        assert reply["fatal"] is False
        assert infers(server) == []                       # the model never ran
        # ... and the same socket works once it is reset
        assert raw.ask(proto.reset(2))["t"] == proto.T_RESET_OK
        assert raw.ask(proto.infer(3, obs, None, 2.0))["t"] == proto.T_ACTIONS
        raw.close()


def test_a_frame_before_hello_is_a_protocol_error(tmp_path):
    with serving(tmp_path) as server:
        raw = Raw(server)
        reply = raw.ask(proto.reset(0))
        assert reply["t"] == proto.T_ERROR and reply["fatal"] is True
        assert reply["code"] == proto.E_BAD_FRAME
        raw.close()


def test_another_protocol_version_is_refused_by_name(tmp_path):
    with serving(tmp_path) as server:
        raw = Raw(server)
        reply = raw.ask({**proto.hello(0, {}), "v": 0})
        assert reply["code"] == proto.E_BAD_VERSION and reply["fatal"] is True
        raw.close()


def test_a_reset_is_handed_to_the_worker(tmp_path):
    """The worker does not answer a reset, so the acknowledgement means "it is
    in the worker's queue, ahead of anything you send next" — the pipe is
    first-in first-out, which is what makes that enough (order is checked in
    test_a_reset_is_delivered_in_order)."""
    with serving(tmp_path) as server:
        raw = Raw(server)
        raw.ask(proto.hello(0, {}))
        assert raw.ask(proto.reset(1))["t"] == proto.T_RESET_OK
        assert wait_for(lambda: server.seen == [("reset", None)])
        raw.close()


def test_a_superseded_request_is_not_served(tmp_path):
    """The client keeps one request outstanding. A newer one already waiting
    means it gave up on this one, so the model must not be run for it."""
    gate = threading.Event()
    with serving(tmp_path, script={1: "gate"}, gate=gate) as server:
        raw = Raw(server)
        raw.ask(proto.hello(0, {}))
        raw.ask(proto.reset(1))
        windows = [make_window(seed=s) for s in (10, 11, 12)]
        enc = [proto.encode_window(w, {"encoding": "raw"}) for w in windows]
        raw.send(proto.infer(10, enc[0], None, 1.0))
        assert wait_for(lambda: len(infers(server)) == 1)   # the model is busy with #10
        raw.send(proto.infer(11, enc[1], None, 2.0))
        raw.send(proto.infer(12, enc[2], None, 3.0))
        time.sleep(0.3)                                     # both are waiting behind #10
        gate.set()

        first, second, third = raw.recv(), raw.recv(), raw.recv()
        raw.close()
        assert (first["t"], first["id"]) == (proto.T_ACTIONS, 10)
        assert (second["t"], second["id"], second["code"]) == (proto.T_ERROR, 11, proto.E_STALE_OBS)
        assert second["fatal"] is False
        assert (third["t"], third["id"]) == (proto.T_ACTIONS, 12)
        np.testing.assert_array_equal(third["actions"], chunk_for(windows[2]))
        assert len(infers(server)) == 2                     # #11 never reached the model


def test_an_observation_older_than_one_already_served_is_refused(tmp_path):
    with serving(tmp_path) as server:
        raw = Raw(server)
        raw.ask(proto.hello(0, {}))
        raw.ask(proto.reset(1))
        obs = proto.encode_window(make_window(), {"encoding": "raw"})
        assert raw.ask(proto.infer(2, obs, None, 10.0))["t"] == proto.T_ACTIONS
        reply = raw.ask(proto.infer(3, obs, None, 5.0))
        assert (reply["t"], reply["code"]) == (proto.T_ERROR, proto.E_STALE_OBS)
        # a reset starts a new episode: the clock comparison starts over
        raw.ask(proto.reset(4))
        assert raw.ask(proto.infer(5, obs, None, 5.0))["t"] == proto.T_ACTIONS
        raw.close()


def test_a_worker_error_is_retryable_and_keeps_the_socket(tmp_path):
    with serving(tmp_path, script={1: "raise"}) as server:
        client = WebsocketPolicyClient(server.uri, contract=None)
        client.connect()
        with pytest.raises(RuntimeError, match=r"infer_failed.*boom"):
            client.infer(make_window(seed=1))
        socket_before = client._conn
        second = make_window(seed=2)
        np.testing.assert_array_equal(client.infer(second), chunk_for(second))
        assert client._conn is socket_before
        client.close()


def test_a_second_client_is_refused_while_one_is_connected(tmp_path):
    with serving(tmp_path) as server:
        first = WebsocketPolicyClient(server.uri, contract=None)
        first.connect()
        second = WebsocketPolicyClient(server.uri, contract=None, reconnect_attempts=1)
        with pytest.raises(RuntimeError, match="already serving another client"):
            second.connect()
        # the first one is undisturbed
        window = make_window(seed=5)
        np.testing.assert_array_equal(first.infer(window), chunk_for(window))
        first.close()
        # ... and once it leaves, the next client is welcome
        assert wait_for(lambda: not server._busy.locked())
        third = WebsocketPolicyClient(server.uri, contract=None)
        third.connect()
        third.close()


def test_a_dead_worker_stops_the_server(tmp_path):
    with serving(tmp_path, script={1: "die"}) as server:
        client = WebsocketPolicyClient(server.uri, contract=None, reconnect_attempts=1,
                                       connect_timeout=2.0)
        client.connect()
        with pytest.raises(RuntimeError, match="infer_failed.*worker is gone"):
            client.infer(make_window())
        assert wait_for(lambda: server.worker_died)
        # no longer listening: a restart is needed, and the robot side is told so
        assert wait_for(lambda: server._server.socket.fileno() == -1)
        with pytest.raises(Exception):  # noqa: B017, PT011 — refused or closed, by platform
            late = WebsocketPolicyClient(server.uri, contract=None, reconnect_attempts=1,
                                         connect_timeout=2.0)
            late.connect()


def test_a_worker_that_fails_to_load_is_reported():
    def broken(conn):
        conn.send(("error", "FileNotFoundError: no such checkpoint"))

    with pytest.raises(RuntimeError, match="failed to load.*no such checkpoint"):
        ps.Worker(broken, {}, in_thread=True)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_worker_that_dies_without_a_word_is_not_waited_for_forever():
    """An import that fails before the worker's own error handling leaves
    nothing on the pipe. Found with a real checkpoint on a machine without
    `rich`: the server sat there, silent, until it was killed."""
    def silent(conn):
        raise ModuleNotFoundError("No module named 'rich'")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped before it finished loading"):
        ps.Worker(silent, {}, in_thread=True)
    assert time.monotonic() - t0 < 5.0


def test_a_spawned_worker_that_dies_is_noticed_too():
    """The same, in the mode the server really runs: a separate process."""
    with pytest.raises(RuntimeError, match="stopped before it finished loading"):
        ps.Worker(_exit_at_once, {}, in_thread=False)


def test_the_requirement_check_names_what_is_missing(monkeypatch):
    monkeypatch.setitem(ps.REQUIREMENTS, "surely_not_a_module_xyz", "some-package")
    assert "some-package" in ps.missing_requirements()
    assert "websockets" not in ps.missing_requirements()


# ── 4. the transport behaves like the pipe it replaces ───────────────────────

def test_the_first_recv_is_the_worker_metadata(tmp_path):
    with serving(tmp_path) as server:
        conn = RemoteWorkerConnection(server.uri)
        assert conn.poll(0) is True
        meta = conn.recv()
        assert meta["n_obs_steps"] == 2 and meta["n_action_steps"] == 8
        assert meta["state_input"] == "relative_wrt_start" and meta["state_dim"] == 16
        assert meta["image_keys"] == [CAM]
        # ... plus the one fact a local deployment reads from disk
        assert meta["action_pose_repr"] == "relative"
        assert conn.model_id.endswith("pretrained_model")
        assert conn.poll(0) is False
        conn.send(None)


def test_send_returns_at_once_and_recv_gives_that_windows_chunk(tmp_path):
    gate = threading.Event()
    with serving(tmp_path, script={1: "gate"}, gate=gate) as server:
        conn = RemoteWorkerConnection(server.uri)
        conn.recv()
        window = make_window(seed=7)
        t0 = time.monotonic()
        conn.send(("infer", window))
        assert time.monotonic() - t0 < 0.5         # did not wait for the model
        assert conn.poll(0) is False               # nothing yet: the prefetch keeps going
        assert conn.poll(0.05) is False
        gate.set()
        assert conn.poll(5.0) is True
        chunk = conn.recv()
        np.testing.assert_array_equal(chunk, chunk_for(window))
        assert chunk.flags.writeable
        assert set(conn.last_timing) == {"decode_ms", "infer_ms"}
        (received,) = infers(server)
        np.testing.assert_array_equal(received[-1][CAM], window[-1][CAM])
        conn.close()


def test_a_reset_is_delivered_in_order(tmp_path):
    with serving(tmp_path) as server:
        conn = RemoteWorkerConnection(server.uri)
        conn.recv()
        conn.send(("infer", make_window(seed=1)))
        conn.recv()
        conn.send(("reset", None))
        conn.send(("infer", make_window(seed=2)))
        conn.recv()
        conn.close()
        # (the reset before the very first infer is the protocol's own barrier)
        assert [kind for kind, _ in server.seen] == ["reset", "infer", "reset", "infer"]


def test_a_worker_error_arrives_as_the_pipes_error_tuple(tmp_path):
    with serving(tmp_path, script={1: "raise"}) as server:
        conn = RemoteWorkerConnection(server.uri)
        conn.recv()
        conn.send(("infer", make_window(seed=1)))
        kind, message = conn.recv()
        assert kind == "error" and "boom" in message
        window = make_window(seed=2)
        conn.send(("infer", window))
        np.testing.assert_array_equal(conn.recv(), chunk_for(window))
        conn.close()


def test_a_timeout_is_an_error_and_the_late_chunk_is_never_delivered(tmp_path):
    """The failure protocol v1 exists to prevent, end to end: the answer to a
    request the robot gave up on must not be handed out as the next chunk."""
    with serving(tmp_path, script={1: 0.6}) as server:
        conn = RemoteWorkerConnection(server.uri, infer_timeout=0.2)
        conn.recv()
        slow = make_window(seed=1)
        conn.send(("infer", slow))
        kind, message = conn.recv()
        assert kind == "error" and "TimeoutError" in message
        time.sleep(0.6)                            # the late answer is now in the socket
        fresh = make_window(seed=2)
        conn.send(("infer", fresh))
        chunk = conn.recv()
        np.testing.assert_array_equal(chunk, chunk_for(fresh))
        assert not np.array_equal(chunk, chunk_for(slow))
        conn.close()


def test_a_vanished_server_is_an_error_not_a_hang_and_a_new_one_is_picked_up(tmp_path, caplog):
    with serving(tmp_path) as server:
        port = server.port
        conn = RemoteWorkerConnection(server.uri, connect_timeout=2.0, reconnect_attempts=1)
        conn.recv()
        window = make_window(seed=1)
        conn.send(("infer", window))
        np.testing.assert_array_equal(conn.recv(), chunk_for(window))
    # the server is gone now
    conn.send(("infer", make_window(seed=2)))
    assert conn.poll(10.0) is True                 # recv() would not block forever
    kind, _ = conn.recv()
    assert kind == "error"
    conn.send(("infer", make_window(seed=3)))      # still gone
    assert conn.recv()[0] == "error"

    with serving(tmp_path, port=port) as server:   # restarted on the same port
        window = make_window(seed=4)
        with caplog.at_level(logging.WARNING):
            conn.send(("infer", window))
            np.testing.assert_array_equal(conn.recv(), chunk_for(window))
        # it was reset before it was used, and the operator is told what that means
        assert [kind for kind, _ in server.seen] == ["reset", "infer"]
        assert any("reconnected" in r.getMessage() and "Restart the episode" in r.getMessage()
                   for r in caplog.records)
        conn.close()


def test_stopping_frees_the_server_for_the_next_robot_process(tmp_path):
    with serving(tmp_path) as server:
        conn = RemoteWorkerConnection(server.uri)
        conn.recv()
        conn.send(None)                            # what Policy.shutdown() sends
        assert wait_for(lambda: not conn._thread.is_alive())
        with pytest.raises(OSError, match="closed"):
            conn.send(("infer", make_window()))
        assert wait_for(lambda: not server._busy.locked())
        again = RemoteWorkerConnection(server.uri)
        assert again.recv()["n_action_steps"] == 8
        again.close()


def test_a_server_without_worker_metadata_is_refused(tmp_path):
    class Plain(ps.PolicyServer):
        def _hello_ok(self, rid):
            frame = super()._hello_ok(rid)
            frame.pop("meta")
            return frame

    with serving(tmp_path, server_cls=Plain) as server:
        with pytest.raises(RuntimeError, match="did not send inference-worker metadata"):
            RemoteWorkerConnection(server.uri)


# ── 5. the unmodified RelativeLerobotPolicy loop, over the network ───────────

def _random_traj(n: int, seed: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    T = [np.eye(4)]
    T[0][:3, :3] = Rotation.random(random_state=int(seed)).as_matrix()
    T[0][:3, 3] = rng.uniform(-0.5, 0.5, 3)
    for _ in range(n - 1):
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.03, 3)).as_matrix()
        step[:3, 3] = rng.normal(0, 0.01, 3)
        T.append(T[-1] @ step)
    return np.stack(T)


def _pose9(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[..., :3, 3], T[..., :2, :3].reshape(*T.shape[:-2], 6)], axis=-1)


def _mat(p9: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    a1, a2 = p9[3:6], p9[6:9]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    T[:3, :3] = np.stack([b1, b2, np.cross(b1, b2)])
    T[:3, 3] = p9[:3]
    return T


def ground_truth_worker(conn, meta, seen, GT, delay=0.0):
    """A CORRECT model: the ground-truth relative chunk wrt the newest frame of
    the window it is handed. It knows nothing else — in particular not where
    the robot is by the time it answers."""
    conn.send(meta)
    poses = _pose9(GT)
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        kind, window = message
        seen.append((kind, window))
        if kind != "infer":
            continue
        base9 = np.asarray(window[-1]["observation.state"][:9], np.float64)
        k = int(np.argmin(np.linalg.norm(poses - base9, axis=1)))
        base_inv = np.linalg.inv(_mat(base9))
        if delay:
            time.sleep(delay)
        conn.send(np.stack([
            np.concatenate([_pose9(base_inv @ GT[k + 1 + j]), [0.7]]).astype(np.float32)
            for j in range(meta["n_action_steps"])
        ]))


class _Env:
    """Scripted robot that tracks every command perfectly."""

    def __init__(self, GT):
        from scipy.spatial.transform import Rotation

        self._Rotation = Rotation
        self.GT = GT
        self.k = 0
        self.commands: list = []
        self.config = types.SimpleNamespace(
            orientation_representation="rotation_6d", use_relative_actions=False)
        self.robot = types.SimpleNamespace()
        self._sync()

    def _sync(self):
        T = self.GT[self.k]
        self.robot.end_effector_pose = types.SimpleNamespace(
            position=T[:3, 3].copy(), orientation=self._Rotation.from_matrix(T[:3, :3]))

    def get_obs(self):
        return {
            "observation.state.cartesian": _pose9(self.GT[self.k]).astype(np.float32),
            "observation.state.gripper": np.array([0.6], np.float32),
            "observation.images.primary": np.full((4, 4, 3), self.k, np.uint8),
        }

    def step(self, action, block=False):
        self.commands.append((self.k, np.asarray(action, np.float64).copy()))
        self.k += 1
        self._sync()


GT_META = {"n_obs_steps": 2, "n_action_steps": 8, "state_input": "relative",
           "state_dim": 10, "image_keys": ["observation.images.primary"],
           "policy_type": "diffusion", "rename_map": {},
           "image_shapes": {"observation.images.primary": (3, 4, 4)}}


def _deploy(env, server, **kwargs):
    """The policy exactly as deploy_policy builds it: through its constructor,
    with `server` set. `pretrained_path` is only a label on this side."""
    return rlp.RelativeLerobotPolicy(
        env=env, pretrained_path="remote", device_max_width=0.085,
        reference_width=0.085, server=server.uri, **kwargs)


@pytest.mark.parametrize(("async_inference", "delay"), [
    (False, 0.0),      # the classic loop: block for each chunk
    (True, 0.0),       # prefetch, the answer is back before the swap
    (True, 0.05),      # prefetch, the answer is late: the loop stalls, then swaps
])
def test_the_real_policy_commands_ground_truth_over_the_network(
        tmp_path, async_inference, delay):
    GT = _random_traj(60, seed=41)
    with serving(tmp_path, meta=GT_META, target=ground_truth_worker,
                 extra={"GT": GT, "delay": delay}) as server:
        env = _Env(GT)
        pol = _deploy(env, server, async_inference=async_inference, prefetch_lead=3)
        assert pol.inf_proc is None                 # nothing local to stop
        assert isinstance(pol.parent_conn, RemoteWorkerConnection)
        assert pol.n_obs_steps == 2 and pol.n_action_steps == 8
        fn = pol.make_data_fn()
        for _ in range(28):                      # three and a half chunks
            _, action = fn()
            assert action is not None, "the loop held pose: a request failed"

        # every commanded pose is the ground-truth next pose, as MATRICES
        assert len(env.commands) == 28
        for k, action in env.commands:
            np.testing.assert_allclose(_mat(action[:9]), GT[k + 1], atol=1e-6)
            assert abs(action[9] - 0.7) < 1e-6

        windows = infers(server)
        assert all(len(w) == 2 for w in windows[1:])       # the full obs window travels
        # consecutive frames of a window are ONE control tick apart — the
        # reason the window is built on the robot side and not on the server
        for window in windows[1:]:
            older = int(window[0]["observation.images.primary"][0, 0, 0])
            newer = int(window[1]["observation.images.primary"][0, 0, 0])
            assert newer - older == 1

        # an episode boundary: the policy's reset reaches the server's worker
        before = len(server.seen)
        pol.reset()
        assert wait_for(lambda: ("reset", None) in server.seen[before:])
        assert pol._chunk is None and len(pol._history) == 0

        # shutdown says goodbye; the server and its model stay up for the next process
        pol.shutdown()
        assert wait_for(lambda: not server._busy.locked())
        again = _deploy(_Env(GT), server)
        assert again.make_data_fn()()[1] is not None
        again.shutdown()


# ── 6. the seam: one place decides where the model runs ──────────────────────

def test_both_kinds_of_connection_satisfy_the_one_interface(tmp_path):
    import multiprocessing as mp

    from crisp_gym.policy.worker_connection import WorkerConnection, is_error

    ours, theirs = mp.get_context("spawn").Pipe()
    assert isinstance(ours, WorkerConnection)
    ours.close()
    theirs.close()
    with serving(tmp_path) as server:
        conn = RemoteWorkerConnection(server.uri)
        assert isinstance(conn, WorkerConnection)
        conn.close()
    assert is_error(("error", "boom")) and not is_error(np.zeros((8, 10)))
    assert not is_error({"n_obs_steps": 2})


@pytest.mark.parametrize(("setting", "uri"), [
    ("ws://192.168.1.50:8765", "ws://192.168.1.50:8765"),
    ("192.168.1.50:8765", "ws://192.168.1.50:8765"),
    ({"host": "gpu", "port": 9000}, "ws://gpu:9000"),
    ({"host": "gpu"}, "ws://gpu:8765"),
    ({"uri": "wss://gpu:1"}, "wss://gpu:1"),
])
def test_the_server_setting_names_a_server(setting, uri):
    parsed = rlp.parse_server_setting(setting)
    assert parsed["uri"] == uri
    assert parsed["infer_timeout"] == 5.0 and parsed["connect_timeout"] == 10.0


def test_no_server_setting_means_a_local_worker():
    assert rlp.parse_server_setting(None) is None
    assert rlp.parse_server_setting("") is None


def test_the_server_setting_carries_its_timeouts():
    parsed = rlp.parse_server_setting(
        {"host": "gpu", "infer_timeout_s": 2, "connect_timeout_s": 3, "reconnect_attempts": 7})
    assert (parsed["infer_timeout"], parsed["connect_timeout"],
            parsed["reconnect_attempts"]) == (2.0, 3.0, 7)


def test_a_misspelt_server_key_is_refused_not_ignored():
    with pytest.raises(ValueError, match="infer_timeout"):
        rlp.parse_server_setting({"host": "gpu", "infer_timeout": 2})   # missing _s
    with pytest.raises(ValueError, match="uri.*host"):
        rlp.parse_server_setting({"port": 8765})
    with pytest.raises(ValueError, match="address or a mapping"):
        rlp.parse_server_setting(8765)


@pytest.mark.parametrize(("absolute", "expected"), [(False, "relative"), (True, "absolute")])
def test_remote_action_repr_auto_comes_from_the_servers_stamp(tmp_path, absolute, expected):
    """Locally `auto` reads action_repr.json next to the checkpoint. That file
    is on the GPU machine now; the server has read it and says so."""
    GT = _random_traj(20, seed=3)
    with serving(tmp_path, meta=GT_META, target=ground_truth_worker,
                 extra={"GT": GT}, absolute=absolute) as server:
        pol = _deploy(_Env(GT), server)
        assert pol.meta["action_pose_repr"] == expected
        assert pol.action_repr == expected
        pol.shutdown()
        # an explicit setting still wins, as it does locally
        assert wait_for(lambda: not server._busy.locked())
        forced = _deploy(_Env(GT), server, action_repr="relative")
        assert forced.action_repr == "relative"
        forced.shutdown()


def test_a_state_input_the_server_does_not_run_refuses_to_start(tmp_path):
    GT = _random_traj(20, seed=3)
    with serving(tmp_path, meta=GT_META, target=ground_truth_worker, extra={"GT": GT}) as server:
        with pytest.raises(ValueError, match=r"state_input='absolute'.*runs the checkpoint as 'relative'"):
            _deploy(_Env(GT), server, state_input="absolute")
        # the refused attempt did not leave the server occupied
        assert wait_for(lambda: not server._busy.locked())
        agreeing = _deploy(_Env(GT), server, state_input="relative")
        agreeing.shutdown()


def test_model_settings_given_on_the_robot_side_are_reported_as_not_applied(tmp_path, caplog):
    GT = _random_traj(20, seed=3)
    with serving(tmp_path, meta=GT_META, target=ground_truth_worker, extra={"GT": GT}) as server:
        with caplog.at_level(logging.WARNING):
            pol = _deploy(_Env(GT), server, num_inference_steps=10, scheduler="ddim")
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "NOT APPLIED" in message
        assert "num_inference_steps" in message and "scheduler" in message
        pol.shutdown()
        caplog.clear()
        assert wait_for(lambda: not server._busy.locked())
        with caplog.at_level(logging.WARNING):
            quiet = _deploy(_Env(GT), server)
        assert not any("NOT APPLIED" in r.getMessage() for r in caplog.records)
        quiet.shutdown()


def test_an_unreachable_server_fails_at_construction_not_in_the_control_loop():
    GT = _random_traj(20, seed=3)
    with pytest.raises(Exception, match="."):  # noqa: B017, PT011 — refused, by platform
        rlp.RelativeLerobotPolicy(
            env=_Env(GT), pretrained_path="remote", device_max_width=0.085,
            server={"uri": "ws://127.0.0.1:9", "connect_timeout_s": 1.0, "reconnect_attempts": 1})


def local_stand_in(conn, pretrained_path, state_input, overrides, num_inference_steps, scheduler):
    """Stands where inference_worker stands in a LOCAL deployment (module-level,
    so the spawned process can import it). Reports the arguments it was started
    with, then answers like the stand-in above."""
    conn.send({**GT_META, "started_with": {
        "pretrained_path": pretrained_path, "state_input": state_input,
        "overrides": overrides, "num_inference_steps": num_inference_steps,
        "scheduler": scheduler}})
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        if message[0] == "infer":
            conn.send(chunk_for(message[1], GT_META["n_action_steps"]))


def test_without_a_server_the_constructor_starts_a_local_worker_as_before(tmp_path, monkeypatch):
    """The default path, through the real constructor: a spawned process on a
    multiprocessing pipe, started with the same arguments as ever, the action
    representation read from the stamp on disk, and joined at shutdown."""
    from crisp_gym.policy.worker_connection import WorkerConnection

    monkeypatch.setattr(rlp, "inference_worker", local_stand_in)
    ckpt = make_checkpoint(tmp_path, absolute=True)
    GT = _random_traj(20, seed=3)
    pol = rlp.RelativeLerobotPolicy(
        env=_Env(GT), pretrained_path=str(ckpt), device_max_width=0.085,
        num_inference_steps=10, scheduler="ddim", overrides={"a": 1})
    try:
        assert pol.inf_proc is not None and pol.inf_proc.is_alive()
        assert pol.inf_proc.daemon
        assert isinstance(pol.parent_conn, WorkerConnection)
        assert not isinstance(pol.parent_conn, RemoteWorkerConnection)
        assert pol.meta["started_with"] == {
            "pretrained_path": str(ckpt), "state_input": "auto", "overrides": {"a": 1},
            "num_inference_steps": 10, "scheduler": "ddim"}
        assert pol.action_repr == "absolute"         # action_repr.json, from disk
        _, action = pol.make_data_fn()()
        assert action is not None and action.shape == (10,)
    finally:
        pol.shutdown()
    assert not pol.inf_proc.is_alive()
