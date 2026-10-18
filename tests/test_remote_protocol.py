"""Conformance tests for the remote-inference wire format (protocol v1).

Both real implementations talk to each other over a REAL loopback websocket —
the client is `WebsocketPolicyClient` out of remote_policy.py, the server is a
few lines built on the same shared `remote_protocol` module the production
server will use. Only the policy itself is faked, so the framing, the
handshake, the id correlation and the error paths are genuinely exercised.

The test these exist for is test_a_late_reply_is_never_mistaken_for_the_next:
v0 was an untagged request/reply, so after an infer timeout the late reply sat
in the socket and the NEXT recv() returned the previous observation's chunk.
RemotePolicy then composed it against the base pose captured for the current
observation — the precise error it caches `_chunk_base` to avoid, costing
"~doubles velocity, tens of mm / tens of degrees by step 8" per its own
comment. One dropped deadline was enough, silently.

Run: python -m pytest tests/test_remote_protocol.py -v
"""

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load():
    """Load the two modules by path: crisp_gym.policy's __init__ pulls torch.

    The stub package is REMOVED from sys.modules afterwards. Leaving it there
    poisons every test module imported later in the same pytest process —
    they get the stub instead of the real crisp_gym. Both modules have already
    executed their imports by then and hold direct references, so the restore
    costs nothing.
    """
    names = ("crisp_gym", "crisp_gym.policy", "crisp_gym.policy.policy",
             "crisp_gym.policy.remote_protocol", "crisp_gym.policy.remote_policy")
    saved = {k: sys.modules.get(k) for k in names}

    pkg = types.ModuleType("crisp_gym")
    pol = types.ModuleType("crisp_gym.policy")
    polpol = types.ModuleType("crisp_gym.policy.policy")
    polpol.Policy = object
    polpol.Action = object
    polpol.Observation = dict
    polpol.register_policy = lambda name: (lambda cls: cls)
    sys.modules.update({"crisp_gym": pkg, "crisp_gym.policy": pol,
                        "crisp_gym.policy.policy": polpol})

    def _mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    try:
        base = REPO / "crisp_gym" / "policy"
        proto = _mod("crisp_gym.policy.remote_protocol", base / "remote_protocol.py")
        pol.remote_protocol = proto
        client = _mod("crisp_gym.policy.remote_policy", base / "remote_policy.py")
        return proto, client
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


proto, rp = _load()
WebsocketPolicyClient = rp.WebsocketPolicyClient

CONTRACT = {
    "observation": {"n_obs_steps": 1, "keys": [
        {"name": "observation.state.cartesian", "shape": [9], "layout": "xyz+rot6d"},
    ]},
    "action": {"dim": 10, "layout": "xyz+rot6d+gripper", "pose_repr": "relative",
               "chunk_size": 16, "gripper": "normalized"},
    "normalization": "server_side",
}


# ── a real server on a real socket ───────────────────────────────────────────

class FakeServer:
    """Minimal v1 server. `behaviour` injects the pathological cases."""

    def __init__(self, contract=None, behaviour="normal", image=None):
        self.contract = proto.contract_fields(contract or CONTRACT)
        self.behaviour = behaviour
        self.image = image or {"encoding": "raw"}
        self.seen: list[dict] = []
        self.port = None
        self._server = None
        self._thread = None

    def __enter__(self):
        from websockets.sync.server import serve

        self._server = serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()

    @property
    def uri(self):
        return f"ws://127.0.0.1:{self.port}"

    def _handle(self, conn):
        try:
            for payload in conn:
                frame = proto.validate(proto.unpack(payload),
                                       expect_from=proto.CLIENT_FRAMES)
                self.seen.append(frame)
                t, rid = frame["t"], frame["id"]
                if t == proto.T_HELLO:
                    conn.send(proto.pack(proto.hello_ok(
                        rid, model_id="fake", observation=self.contract["observation"],
                        action=self.contract["action"], chunk_size=16,
                        normalization="server_side", image=self.image,
                        contract=self.contract)))
                elif t == proto.T_RESET:
                    conn.send(proto.pack(proto.reset_ok(rid)))
                elif t == proto.T_INFER:
                    self._on_infer(conn, frame, rid)
                elif t == proto.T_BYE:
                    return
        except Exception:  # noqa: BLE001, S110  (client hung up)
            pass

    def _on_infer(self, conn, frame, rid):
        n_infer = sum(1 for f in self.seen if f["t"] == proto.T_INFER)
        if self.behaviour == "slow_first" and n_infer == 1:
            # Answer request 1 only AFTER the client has given up on it, so
            # the late frame is queued when request 2 goes out.
            time.sleep(0.45)
            conn.send(proto.pack(proto.actions(
                rid, np.full((16, 10), 111.0, np.float32), time.monotonic())))
            return
        if self.behaviour == "stale_obs":
            conn.send(proto.pack(proto.error(rid, proto.E_STALE_OBS, "too old")))
            return
        if self.behaviour == "fatal":
            conn.send(proto.pack(proto.error(rid, proto.E_BAD_FRAME, "nope")))
            return
        value = float(n_infer)
        conn.send(proto.pack(proto.actions(
            rid, np.full((16, 10), value, np.float32), time.monotonic())))


def _client(server, **kw):
    kw.setdefault("contract", CONTRACT)
    kw.setdefault("infer_timeout", 0.3)
    kw.setdefault("connect_timeout", 3.0)
    return WebsocketPolicyClient(server.uri, **kw)


# ── the regression v1 exists for ─────────────────────────────────────────────

def test_a_late_reply_is_never_mistaken_for_the_next():
    with FakeServer(behaviour="slow_first") as srv:
        c = _client(srv)
        c.connect()
        c.reset()
        with pytest.raises(TimeoutError):
            c.infer({"observation.state.cartesian": np.zeros(9, np.float32)})
        # Request 1's answer (all 111.0) lands while we wait for request 2's.
        chunk = c.infer({"observation.state.cartesian": np.zeros(9, np.float32)})
        assert chunk[0, 0] != 111.0, (
            "the client returned the TIMED-OUT request's chunk — it would be "
            "composed against the wrong base pose"
        )
        c.close()


def test_every_frame_carries_version_type_and_id():
    with FakeServer() as srv:
        c = _client(srv)
        c.connect()
        c.reset()
        c.infer({"observation.state.cartesian": np.zeros(9, np.float32)})
        c.close()
        assert srv.seen, "no frames reached the server"
        for f in srv.seen:
            assert f["v"] == proto.PROTOCOL_VERSION
            assert f["t"] in proto.CLIENT_FRAMES
            assert isinstance(f["id"], int)
        ids = [f["id"] for f in srv.seen]
        assert ids == sorted(set(ids)), f"ids must be unique and monotonic: {ids}"


def test_infer_carries_the_task_and_the_capture_time():
    with FakeServer() as srv:
        c = _client(srv)
        c.connect()
        c.reset()
        c.infer({"observation.state.cartesian": np.zeros(9, np.float32),
                 "task": "open the box"}, task="open the box", obs_t=1234.5)
        inf = next(f for f in srv.seen if f["t"] == proto.T_INFER)
        assert inf["task"] == "open the box"
        assert inf["obs_t"] == pytest.approx(1234.5)
        assert "task" not in inf["obs"], "task travels as its own field, not an obs array"
        c.close()


# ── the handshake that never existed ─────────────────────────────────────────

def test_a_contract_mismatch_refuses_to_run():
    wrong = {**CONTRACT, "action": {**CONTRACT["action"], "dim": 16}}
    with FakeServer(contract=wrong) as srv:
        c = _client(srv)
        with pytest.raises(RuntimeError, match="CONTRACT MISMATCH"):
            c.connect()


def test_the_mismatch_message_names_the_field_and_both_sides():
    wrong = {**CONTRACT, "action": {**CONTRACT["action"], "pose_repr": "absolute"}}
    with FakeServer(contract=wrong) as srv:
        c = _client(srv)
        try:
            c.connect()
            raise AssertionError("expected a refusal")
        except RuntimeError as exc:
            msg = str(exc)
        assert "pose_repr" in msg
        assert "relative" in msg and "absolute" in msg
        assert "sha256:" in msg


def test_verify_handshake_off_warns_but_runs():
    wrong = {**CONTRACT, "action": {**CONTRACT["action"], "dim": 16}}
    with FakeServer(contract=wrong) as srv:
        c = _client(srv, verify_handshake=False)
        c.connect()  # must not raise
        assert c.server_meta["model_id"] == "fake"
        c.close()


def test_matching_contracts_connect():
    with FakeServer() as srv:
        c = _client(srv)
        c.connect()
        assert c.server_meta["chunk_size"] == 16
        c.close()


# ── the reset barrier ────────────────────────────────────────────────────────

def test_reset_is_acknowledged_before_inference():
    with FakeServer() as srv:
        c = _client(srv)
        c.connect()
        c.reset()
        kinds = [f["t"] for f in srv.seen]
        assert kinds == [proto.T_HELLO, proto.T_RESET]
        c.close()


def test_inferring_without_a_reset_resets_first():
    """A server still holding the previous episode's queues would return a
    chunk computed from state this episode never saw."""
    with FakeServer() as srv:
        c = _client(srv)
        c.connect()
        c.infer({"observation.state.cartesian": np.zeros(9, np.float32)})
        kinds = [f["t"] for f in srv.seen]
        assert kinds.index(proto.T_RESET) < kinds.index(proto.T_INFER)
        c.close()


# ── error frames ─────────────────────────────────────────────────────────────

def test_a_retryable_error_raises_without_killing_the_socket():
    with FakeServer(behaviour="stale_obs") as srv:
        c = _client(srv)
        c.connect()
        c.reset()
        with pytest.raises(RuntimeError, match="stale_obs"):
            c.infer({"observation.state.cartesian": np.zeros(9, np.float32)})
        c.close()


def test_a_fatal_error_is_marked_fatal_by_its_code():
    assert proto.error(1, proto.E_BAD_FRAME, "x")["fatal"] is True
    assert proto.error(1, proto.E_STALE_OBS, "x")["fatal"] is False
    assert proto.error(1, proto.E_CONTRACT_MISMATCH, "x")["fatal"] is True


# ── codec and validation ─────────────────────────────────────────────────────

def test_arrays_survive_with_dtype_and_shape():
    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    b = (np.random.rand(5, 9) - 0.5).astype(np.float32)
    out = proto.unpack(proto.pack({"v": 1, "t": "infer", "id": 1, "a": a, "b": b}))
    assert out["a"].dtype == np.uint8 and out["a"].shape == (2, 3, 4)
    assert np.array_equal(out["a"], a)
    assert out["b"].dtype == np.float32 and np.allclose(out["b"], b)


def test_a_v0_text_frame_says_so_plainly():
    with pytest.raises(proto.ProtocolError, match="v0 server"):
        proto.unpack("some bare error string")


@pytest.mark.parametrize("frame,match", [
    ({"t": "actions", "id": 1}, "missing required field 'v'"),
    ({"v": 1, "id": 1}, "missing required field 't'"),
    ({"v": 1, "t": "actions"}, "missing required field 'id'"),
    ({"v": 99, "t": "actions", "id": 1}, "protocol version"),
    ({"v": 1, "t": "infer", "id": 1}, "cannot arrive from this direction"),
    ({"v": 1, "t": "actions", "id": -3}, "non-negative"),
])
def test_validate_rejects(frame, match):
    with pytest.raises(proto.ProtocolError, match=match):
        proto.validate(frame, expect_from=proto.SERVER_FRAMES)


def test_the_digest_ignores_key_order():
    a = {"observation": {"n_obs_steps": 1}, "action": {"dim": 10}}
    b = {"action": {"dim": 10}, "observation": {"n_obs_steps": 1}}
    assert proto.digest(a) == proto.digest(b)
    assert proto.digest(a) != proto.digest({**a, "action": {"dim": 16}})


def test_the_digest_ignores_purely_local_settings():
    """Timeouts and hostnames differing between the two sides is harmless; a
    layout or a scale differing is a silent mis-command."""
    base = dict(CONTRACT, server={"host": "a", "infer_timeout_s": 5})
    other = dict(CONTRACT, server={"host": "b", "infer_timeout_s": 99})
    assert proto.digest(proto.contract_fields(base)) == proto.digest(
        proto.contract_fields(other)
    )


def test_an_action_chunk_must_be_two_dimensional():
    with pytest.raises(proto.ProtocolError, match="2-D"):
        proto.actions(1, np.zeros((2, 3, 4)), 0.0)
    assert proto.actions(1, np.zeros(10), 0.0)["actions"].shape == (1, 10)


# ── image negotiation ────────────────────────────────────────────────────────

def test_raw_images_pass_through_untouched():
    img = np.random.randint(0, 255, (8, 12, 3), dtype=np.uint8)
    assert np.array_equal(proto.decode_image(
        proto.encode_image(img, {"encoding": "raw"})), img)


def test_an_unknown_encoding_is_refused_not_guessed():
    with pytest.raises(proto.ProtocolError, match="raw|jpeg"):
        proto.encode_image(np.zeros((4, 4, 3), np.uint8), {"encoding": "webp"})


def test_jpeg_round_trips_through_the_wire():
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    img = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    blob = proto.encode_image(img, {"encoding": "jpeg", "quality": 95})
    assert isinstance(blob, dict) and blob["__jpeg__"]
    assert len(blob["data"]) < img.nbytes, "jpeg should be smaller than raw"
    out = proto.decode_image(proto.unpack(proto.pack(
        {"v": 1, "t": "infer", "id": 1, "img": blob}))["img"])
    assert out.shape == img.shape and out.dtype == np.uint8


def test_the_server_decides_the_encoding():
    """The client obeys hello_ok.image; it never picks lossy on its own."""
    with FakeServer(image={"encoding": "jpeg", "quality": 80}) as srv:
        c = _client(srv)
        c.connect()
        assert c.image_spec == {"encoding": "jpeg", "quality": 80}
        c.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
