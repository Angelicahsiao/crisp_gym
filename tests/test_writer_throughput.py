"""Tests for the writer-decoupling changes (recording back-pressure fixes).

Covers the four things that made a 800x1280 single-camera recording stall:
  * queue slack expressed in seconds rather than a magic frame count,
  * the async-image-writer / video-encoder kwargs reaching lerobot (and being
    filtered per lerobot version instead of TypeError-ing in the subprocess),
  * deadline pacing, so a late wake does not permanently shift the schedule
    and desynchronise the dataset's fixed-fps timestamps,
  * a dead writer raising instead of wedging the recording loop forever.

Run:  python tests/test_writer_throughput.py   (or via pytest)
"""

import multiprocessing as mp
import queue as queue_mod
import sys
import time
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _install_stubs() -> None:
    """Stub the ROS/lerobot-coupled imports so this file stays numpy-only.

    Only module-level symbols are needed: every method under test is called
    UNBOUND with a fake `self`, so no dataset is created and no writer process
    is started. Pre-seeding sys.modules keeps the real modules (which reach for
    rclpy, lerobot and importlib.resources over an installed crisp_py) from
    ever executing.
    """

    def _mod(name):
        return sys.modules.setdefault(name, types.ModuleType(name))

    for name in (
        "rclpy",
        "rclpy.executors",
        "std_msgs",
        "std_msgs.msg",
        "lerobot",
        "lerobot.utils",
        "lerobot.utils.constants",
        "lerobot.datasets",
        "lerobot.datasets.lerobot_dataset",
    ):
        _mod(name)

    sys.modules["rclpy"].ok = lambda: False
    sys.modules["rclpy"].create_node = lambda *a, **k: None
    sys.modules["rclpy.executors"].SingleThreadedExecutor = object
    sys.modules["std_msgs.msg"].String = object
    sys.modules["lerobot.utils.constants"].HF_LEROBOT_HOME = Path("/tmp/lerobot_test_home")

    class _FakeDataset:
        @staticmethod
        def add_frame(frame, task=None):  # its signature is probed at import
            pass

    sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = _FakeDataset

    # crisp_gym.config.path resolves packaged config dirs via
    # importlib.resources.files("crisp_py"), which needs crisp_py installed.
    path_mod = _mod("crisp_gym.config.path")
    path_mod.find_config = lambda name: None
    path_mod.list_configs_in_folder = lambda folder: []

    # crisp_gym.policy.__init__ pulls in every lerobot policy.
    policy_pkg = _mod("crisp_gym.policy")
    policy_pkg.__path__ = []
    policy_mod = _mod("crisp_gym.policy.policy")
    policy_mod.Action = object
    policy_mod.Observation = object

    features_mod = _mod("crisp_gym.util.lerobot_features")
    features_mod.concatenate_state_features = lambda obs, features=None: None


_install_stubs()

from crisp_gym.util.loop_timing import DeadlinePacer  # noqa: E402


# ── queue slack in seconds ───────────────────────────────────────────────────


def _load_config_module():
    return SourceFileLoader(
        "rmc_throughput",
        str(REPO / "crisp_gym" / "record" / "recording_manager_config.py"),
    ).load_module()


def _config(**overrides):
    mod = _load_config_module()
    base = {"features": {}, "repo_id": "t", "fps": 15}
    base.update(overrides)
    return mod.RecordingManagerConfig(**base)


def test_queue_seconds_overrides_queue_size():
    assert _config(queue_size=64).resolved_queue_size() == 64
    # 6 s at 15 fps == 90 frames, whatever queue_size says.
    assert _config(queue_size=64, queue_seconds=6.0).resolved_queue_size() == 90


def test_queue_seconds_rounds_up_and_is_never_zero():
    assert _config(fps=15, queue_seconds=0.5).resolved_queue_size() == 8  # ceil(7.5)
    assert _config(fps=15, queue_seconds=0.001).resolved_queue_size() == 1


# ── deadline pacing ──────────────────────────────────────────────────────────


def test_pacer_absorbs_a_late_wake_instead_of_accumulating():
    """The regression: 14.5 ms of oversleep per frame cost 15 -> 12.29 FPS."""
    period = 0.020
    pacer = DeadlinePacer(period)
    start = pacer.next_deadline - period

    slept_total = 0.0
    for i in range(10):
        # Simulate a wake that overshoots on every other frame.
        if i % 2:
            time.sleep(0.004)  # stand-in for the scheduler returning late
        requested, _ = pacer.wait()
        slept_total += requested

    elapsed = time.perf_counter() - start
    # 10 periods of work+sleep must still take ~10 periods, not 10 periods plus
    # the accumulated lateness.
    assert elapsed < 10 * period * 1.35, f"drifted: {elapsed:.4f}s for {10 * period:.4f}s"
    assert pacer.resyncs == 0


def test_pacer_shortens_the_next_sleep_after_a_late_frame():
    period = 0.050
    pacer = DeadlinePacer(period)
    time.sleep(0.030)  # frame work ate 30 of the 50 ms
    requested, _ = pacer.wait()
    assert 0.010 < requested < 0.025, requested  # ~20 ms left, not a full period


def test_pacer_resyncs_instead_of_bursting_after_a_long_stall():
    """After a multi-second writer stall, catching up would fire a burst of
    zero-sleep frames — worse for the data than the gap itself."""
    period = 0.020
    pacer = DeadlinePacer(period)
    time.sleep(5 * period)  # stall
    requested, _ = pacer.wait()
    assert requested == 0.0
    assert pacer.resyncs == 1
    # The very next frame must get a full period again, not a catch-up burst.
    requested, _ = pacer.wait()
    assert requested > 0.5 * period, requested


def test_pacer_reports_requested_and_actual_separately():
    """The gap between the two is the oversleep the summary reports."""
    pacer = DeadlinePacer(0.030)
    requested, slept = pacer.wait()
    assert requested > 0 and slept > 0
    assert slept >= requested * 0.9


# ── writer kwargs + dead-writer guard (module-level logic, no live writer) ───


def _load_recording_manager():
    """Import recording_manager against the stubs installed at module load."""
    import crisp_gym.record.recording_manager as rm

    return rm


class _FakeEncoder:
    """Stand-in for lerobot's RGBEncoderConfig in the preflight tests."""

    vcodec = "h264_nvenc"
    pix_fmt = "yuv420p"

    def get_codec_options(self):
        return {"g": 2, "rc": 0, "qp": 21, "bf": 0}


def _install_rgb_encoder_stub(resolved_vcodec: str = "h264_nvenc"):
    """Stub lerobot.configs.RGBEncoderConfig.

    `resolved_vcodec` stands in for what the real __post_init__ resolves
    `vcodec: "auto"` to after probing PyAV — the crisp_gym side can only apply
    the NVENC B-frame workaround AFTER that resolution, which is exactly what
    these tests pin.
    """
    captured: dict = {"build_count": 0}

    class _RGBEncoderConfig:
        def __init__(self, **kw):
            captured.update(kw)
            captured["build_count"] = captured.get("build_count", 0) + 1
            self.vcodec = kw.get("vcodec", "libsvtav1")
            if self.vcodec == "auto":
                self.vcodec = resolved_vcodec
            self.crf = kw.get("crf", 30)
            self.g = kw.get("g", 2)
            self.preset = kw.get("preset")
            self.extra_options = kw.get("extra_options")

    configs = sys.modules.setdefault("lerobot.configs", types.ModuleType("lerobot.configs"))
    configs.RGBEncoderConfig = _RGBEncoderConfig
    return captured


def _install_av_stub(open_ok: bool):
    """Stub PyAV so the preflight can be exercised without a GPU or codec."""
    opened: dict = {"sizes": [], "options": None}

    class _Ctx:
        def __init__(self):
            self.width = self.height = 0
            self.pix_fmt = None
            self.time_base = None
            self.options = None

        def open(self):
            if not open_ok:
                raise ValueError("[Errno 22] Invalid argument: 'avcodec_open2'")
            opened["sizes"].append((self.width, self.height))
            opened["options"] = dict(self.options or {})

    class _CodecContext:
        @staticmethod
        def create(name, mode):
            return _Ctx()

    class _Logging:
        VERBOSE = 40
        _level = 8

        @classmethod
        def get_level(cls):
            return cls._level

        @classmethod
        def set_level(cls, level):
            cls._level = level
            opened.setdefault("levels", []).append(level)

    av_mod = sys.modules.setdefault("av", types.ModuleType("av"))
    av_mod.CodecContext = _CodecContext
    av_mod.logging = _Logging
    return opened


def test_writer_kwargs_are_filtered_to_what_this_lerobot_accepts():
    """Passing an unknown kwarg would TypeError inside the writer subprocess,
    surfacing in the parent only as a misleading startup timeout."""
    rm = _load_recording_manager()
    cfg = _config(
        image_writer_threads=4,
        image_writer_processes=0,
        encoder_threads=8,
        vcodec=None,
        video_crf=None,
        video_gop=None,
    )
    fake_self = types.SimpleNamespace(config=cfg)
    fake_self._rgb_encoder = lambda: rm.RecordingManager._rgb_encoder(fake_self)

    def modern(
        repo_id,
        image_writer_processes=0,
        image_writer_threads=0,
        rgb_encoder=None,
        encoder_threads=None,
    ):
        pass

    def ancient(repo_id, fps=None):  # a lerobot with none of the knobs
        pass

    modern_kwargs = rm.RecordingManager._writer_kwargs(fake_self, modern)
    assert modern_kwargs["image_writer_threads"] == 4
    assert modern_kwargs["image_writer_processes"] == 0
    assert modern_kwargs["encoder_threads"] == 8

    assert rm.RecordingManager._writer_kwargs(fake_self, ancient) == {}


def test_rgb_encoder_is_none_when_nothing_is_configured():
    """Leaving all three unset must keep lerobot's own encoder defaults."""
    rm = _load_recording_manager()
    fake_self = types.SimpleNamespace(config=_config())
    assert rm.RecordingManager._rgb_encoder(fake_self) is None


def test_rgb_encoder_passes_crf_and_gop_through():
    rm = _load_recording_manager()
    captured = _install_rgb_encoder_stub("libsvtav1")

    fake_self = types.SimpleNamespace(config=_config(vcodec="auto", video_crf=21))
    encoder = rm.RecordingManager._rgb_encoder(fake_self)
    assert encoder is not None
    assert captured["vcodec"] == "auto" and captured["crf"] == 21
    # video_gop left None must NOT be forced — lerobot's g=2 keeps the training
    # dataloader's random-access seeking fast.
    assert "g" not in captured


def test_nvenc_gets_bf0_at_small_gop():
    """Regression: NVENC requires gop_size > b_frames + 1 and its presets
    enable B-frames, so lerobot's g=2 fails to open with
    "Gop Length should be greater than number of B frames + 1". Verified on an
    RTX 5090: {g:2, rc:0, qp:21} FAILS, the same set + bf=0 OPENS OK.
    """
    rm = _load_recording_manager()
    captured = _install_rgb_encoder_stub("h264_nvenc")

    fake_self = types.SimpleNamespace(config=_config(vcodec="auto", video_crf=21))
    encoder = rm.RecordingManager._rgb_encoder(fake_self)
    assert encoder.extra_options.get("bf") == 0, encoder.extra_options
    assert captured["crf"] == 21


def test_preset_with_auto_vcodec_warns():
    """Preset vocabularies are per-codec (libx264 veryfast, NVENC p1..p7,
    libsvtav1 0..13) and "auto" does not know which codec it will get, so the
    combination is a trap worth naming before the preflight rejects it."""
    import io
    import logging as _logging

    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264_nvenc")
    fake_self = types.SimpleNamespace(config=_config(vcodec="auto", video_preset="veryfast"))

    stream = io.StringIO()
    handler = _logging.StreamHandler(stream)
    rm.logger.addHandler(handler)
    rm.logger.setLevel(_logging.WARNING)
    try:
        rm.RecordingManager._rgb_encoder(fake_self)
    finally:
        rm.logger.removeHandler(handler)
    out = stream.getvalue()
    assert "veryfast" in out and "h264_nvenc" in out, out
    assert "codec-specific" in out


def test_preset_with_an_explicit_vcodec_does_not_warn():
    import io
    import logging as _logging

    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264")
    fake_self = types.SimpleNamespace(config=_config(vcodec="h264", video_preset="veryfast"))

    stream = io.StringIO()
    handler = _logging.StreamHandler(stream)
    rm.logger.addHandler(handler)
    rm.logger.setLevel(_logging.WARNING)
    try:
        encoder = rm.RecordingManager._rgb_encoder(fake_self)
    finally:
        rm.logger.removeHandler(handler)
    assert "codec-specific" not in stream.getvalue()
    assert encoder.preset == "veryfast"


def test_bf0_is_not_forced_on_software_codecs():
    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264")
    fake_self = types.SimpleNamespace(config=_config(vcodec="h264", video_crf=21))
    encoder = rm.RecordingManager._rgb_encoder(fake_self)
    assert not (encoder.extra_options or {}).get("bf")


def test_bf0_is_not_forced_at_a_large_gop():
    """A big GOP satisfies the constraint on its own; B-frames stay available."""
    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264_nvenc")
    fake_self = types.SimpleNamespace(config=_config(vcodec="auto", video_crf=21, video_gop=60))
    encoder = rm.RecordingManager._rgb_encoder(fake_self)
    assert not (encoder.extra_options or {}).get("bf")


def test_explicit_extra_options_win_over_the_bf0_default():
    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264_nvenc")
    fake_self = types.SimpleNamespace(
        config=_config(vcodec="auto", video_crf=21, video_extra_options={"bf": 2})
    )
    encoder = rm.RecordingManager._rgb_encoder(fake_self)
    assert encoder.extra_options["bf"] == 2


def test_rgb_encoder_is_memoized():
    """Resolving vcodec:auto probes PyAV and logs; preflight + writer kwargs
    both need the encoder, so it must be built once."""
    rm = _load_recording_manager()
    captured = _install_rgb_encoder_stub("h264_nvenc")
    fake_self = types.SimpleNamespace(config=_config(vcodec="auto"))
    first = rm.RecordingManager._rgb_encoder(fake_self)
    second = rm.RecordingManager._rgb_encoder(fake_self)
    assert first is second
    assert captured["build_count"] == 1


def test_preflight_raises_before_any_frame_is_recorded():
    """The encoder used to be first opened inside save_episode — after a whole
    episode was teleoperated, and halfway through a write that had already
    committed the parquet rows."""
    rm = _load_recording_manager()
    _install_av_stub(open_ok=False)
    fake_self = types.SimpleNamespace(
        config=_config(
            fps=15,
            features={"observation.images.primary": {"dtype": "video", "shape": (800, 1280, 3)}},
        )
    )
    encoder = _FakeEncoder()
    try:
        rm.RecordingManager._preflight_encoder(fake_self, encoder)
        raise AssertionError("a codec that cannot open must be refused")
    except RuntimeError as exc:
        assert "cannot be opened" in str(exc)
        assert "1280x800" in str(exc), str(exc)
        assert 'vcodec: "h264"' in str(exc), "must name the software fallback"


def test_preflight_retries_verbosely_to_capture_ffmpegs_reason():
    """Hardware encoders fail with a bare AVERROR_UNKNOWN. Without FFmpeg's own
    line ("Gop Length should be greater than...", "OpenEncodeSessionEx failed")
    the operator has nothing to act on — so a failure is retried at VERBOSE and
    the detail is folded into the raised message."""
    rm = _load_recording_manager()
    opened = _install_av_stub(open_ok=False)
    fake_self = types.SimpleNamespace(
        config=_config(
            fps=15,
            features={"observation.images.primary": {"dtype": "video", "shape": (800, 1280, 3)}},
        )
    )
    try:
        rm.RecordingManager._preflight_encoder(fake_self, _FakeEncoder())
        raise AssertionError("must raise")
    except RuntimeError as exc:
        assert "FFmpeg detail:" in str(exc), str(exc)
    # Verbose was turned on and then restored, not left on.
    levels = opened.get("levels", [])
    assert levels[0] == sys.modules["av"].logging.VERBOSE, levels
    assert levels[-1] != sys.modules["av"].logging.VERBOSE, levels


def test_preflight_opens_at_the_declared_image_size():
    rm = _load_recording_manager()
    opened = _install_av_stub(open_ok=True)
    fake_self = types.SimpleNamespace(
        config=_config(
            fps=15,
            features={
                "observation.images.primary": {"dtype": "video", "shape": (800, 1280, 3)},
                "observation.state.cartesian": {"dtype": "float32", "shape": (9,)},
            },
        )
    )
    rm.RecordingManager._preflight_encoder(fake_self, _FakeEncoder())
    assert opened["sizes"] == [(1280, 800)], opened["sizes"]
    assert opened["options"] == {"g": "2", "rc": "0", "qp": "21", "bf": "0"}


def test_preflight_is_a_no_op_without_an_encoder():
    rm = _load_recording_manager()
    rm.RecordingManager._preflight_encoder(types.SimpleNamespace(config=_config()), None)


def test_writer_defaults_to_spawn():
    """fork is unsafe here: the parent has ROS2/DDS threads and, via
    lerobot/torch, an initialised CUDA context. Verified on the owner's box —
    a CUDA-initialised parent forking a child makes h264_nvenc fail with
    AVERROR_UNKNOWN; spawn opens it fine."""
    cfg = _config()
    assert cfg.writer_start_method == "spawn"
    # Spawn re-imports lerobot/torch in the child, which the old 10 s
    # writer_timeout could not cover.
    assert cfg.writer_startup_timeout > cfg.writer_timeout


def test_manager_state_is_picklable_for_spawn():
    """ctx.Process(target=self._writer_proc) pickles the whole manager. A
    threading.Lock and the Process's own self-reference cannot cross, and the
    writer needs neither."""
    import pickle
    import threading

    rm = _load_recording_manager()
    cfg = _config(queue_seconds=6.0)

    holder = types.SimpleNamespace()
    holder.__dict__.update(
        {
            "_state_lock": threading.Lock(),
            "writer": object(),  # the Process object itself
            "_thread": threading.Thread(target=lambda: None),
            "node": object(),  # ROSRecordingManager's rclpy node
            "config": cfg,
            "queue_capacity": 90,
            "_log_level": 20,
            "episode_count": 3,
        }
    )

    state = rm.RecordingManager.__getstate__(holder)
    for dropped in ("_state_lock", "writer", "_thread", "node"):
        assert dropped not in state, dropped
    assert state["config"] is cfg
    assert state["queue_capacity"] == 90 and state["episode_count"] == 3

    # The point of dropping them: what remains must actually pickle.
    pickle.loads(pickle.dumps(state))

    target = types.SimpleNamespace()
    rm.RecordingManager.__setstate__(target, state)
    assert isinstance(target._state_lock, type(threading.Lock()))
    assert target.config is not None


# ── streaming encoding + the dropped-frame guard ─────────────────────────────


def test_streaming_kwargs_reach_lerobot_when_enabled():
    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264_nvenc")
    cfg = _config(fps=15, streaming_encoding=True, encoder_queue_seconds=4.0)
    fake_self = types.SimpleNamespace(config=cfg)
    fake_self._rgb_encoder = lambda: rm.RecordingManager._rgb_encoder(fake_self)

    def create(
        repo_id,
        image_writer_processes=0,
        image_writer_threads=0,
        streaming_encoding=False,
        encoder_queue_maxsize=30,
        rgb_encoder=None,
        encoder_threads=None,
    ):
        pass

    kwargs = rm.RecordingManager._writer_kwargs(fake_self, create)
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_queue_maxsize"] == 60  # 4 s at 15 fps


def test_streaming_kwargs_absent_when_disabled():
    rm = _load_recording_manager()
    _install_rgb_encoder_stub("h264")
    fake_self = types.SimpleNamespace(config=_config(streaming_encoding=False))
    fake_self._rgb_encoder = lambda: rm.RecordingManager._rgb_encoder(fake_self)

    def create(repo_id, image_writer_threads=0, streaming_encoding=False, encoder_queue_maxsize=30):
        pass

    kwargs = rm.RecordingManager._writer_kwargs(fake_self, create)
    assert "streaming_encoding" not in kwargs
    assert "encoder_queue_maxsize" not in kwargs


def _dataset_with_drops(dropped):
    encoder = types.SimpleNamespace(_dropped_frames=dropped)
    return types.SimpleNamespace(writer=types.SimpleNamespace(_streaming_encoder=encoder))


def test_drop_guard_raises_on_any_dropped_frame():
    """feed_frame drops and only WARNS while add_frame still appends a parquet
    row — the rows and the video silently stop lining up."""
    rm = _load_recording_manager()
    fake_self = types.SimpleNamespace(config=_config(streaming_encoding=True))
    dataset = _dataset_with_drops({"observation.images.primary": 3})
    try:
        rm.RecordingManager._check_streaming_drops(fake_self, dataset)
        raise AssertionError("dropped frames must fail the recording")
    except RuntimeError as exc:
        assert "dropped frames" in str(exc)
        assert "CORRUPT" in str(exc)
        assert "encoder_queue_seconds" in str(exc), "must name the remedy"


def test_drop_guard_passes_when_nothing_was_dropped():
    rm = _load_recording_manager()
    fake_self = types.SimpleNamespace(config=_config(streaming_encoding=True))
    rm.RecordingManager._check_streaming_drops(
        fake_self, _dataset_with_drops({"observation.images.primary": 0})
    )


def test_drop_guard_fails_loudly_if_it_cannot_read_the_counters():
    """The counters are private to lerobot. If they move, the guard must break
    loudly — quietly ceasing to protect the data is the worst outcome."""
    rm = _load_recording_manager()
    fake_self = types.SimpleNamespace(config=_config(streaming_encoding=True))

    no_encoder = types.SimpleNamespace(writer=types.SimpleNamespace())
    try:
        rm.RecordingManager._check_streaming_drops(fake_self, no_encoder)
        raise AssertionError("a missing encoder must fail")
    except RuntimeError as exc:
        assert "cannot be detected" in str(exc)

    renamed = types.SimpleNamespace(
        writer=types.SimpleNamespace(_streaming_encoder=types.SimpleNamespace())
    )
    try:
        rm.RecordingManager._check_streaming_drops(fake_self, renamed)
        raise AssertionError("missing counters must fail")
    except RuntimeError as exc:
        assert "counters are unavailable" in str(exc)


def test_drop_guard_is_inert_when_streaming_is_off():
    """The PNG path has no encoder to inspect and never drops."""
    rm = _load_recording_manager()
    fake_self = types.SimpleNamespace(config=_config(streaming_encoding=False))
    rm.RecordingManager._check_streaming_drops(fake_self, types.SimpleNamespace())


def test_encoder_queue_seconds_converts_to_frames():
    assert _config(fps=15, encoder_queue_seconds=4.0).resolved_encoder_queue_size() == 60
    assert _config(fps=15, encoder_queue_seconds=0.01).resolved_encoder_queue_size() == 1


def _writer_that_dies_immediately():
    """A writer killed before it can set writer_error (OOM, SIGSEGV, ...)."""
    return


def _full_queue_with_dead_writer(rm):
    q = mp.JoinableQueue(1)
    q.cancel_join_thread()  # test leaves an item queued on purpose
    proc = mp.Process(target=_writer_that_dies_immediately)
    proc.start()
    proc.join(timeout=10)
    assert not proc.is_alive(), "the fake writer should have exited"
    q.put({"type": "FILLER"})  # queue now full, nothing will ever drain it
    time.sleep(0.2)  # let the feeder thread hand it to the pipe
    return types.SimpleNamespace(queue=q, writer=proc), q


def test_put_raises_when_the_writer_is_gone_instead_of_hanging():
    """writer_error only covers exceptions the writer CATCHES. A writer killed
    outright (OOM, SIGSEGV, an earlier terminate()) used to wedge the loop."""
    rm = _load_recording_manager()
    fake_self, _ = _full_queue_with_dead_writer(rm)

    t0 = time.perf_counter()
    try:
        rm.RecordingManager._put_blocking(fake_self, {"type": "FRAME"}, poll_s=0.2)
        raise AssertionError("a dead writer with a full queue must raise")
    except RuntimeError as exc:
        assert "writer process is gone" in str(exc)
    assert time.perf_counter() - t0 < 10.0, "should fail fast, not hang"


def test_shutdown_put_logs_instead_of_raising_on_a_dead_writer():
    """__exit__ enqueues PUSH_TO_HUB/SHUTDOWN. Raising there would mask the
    exception being handled and skip the terminate/join cleanup below it."""
    rm = _load_recording_manager()
    fake_self, _ = _full_queue_with_dead_writer(rm)

    # Must return normally — no exception.
    rm.RecordingManager._put_blocking(fake_self, {"type": "SHUTDOWN"}, poll_s=0.2, required=False)


def test_put_does_not_drop_frames_while_the_writer_is_alive():
    """Dataset first: back-pressure stalls the loop, it never skips a frame."""
    rm = _load_recording_manager()

    q = mp.JoinableQueue(1)
    q.cancel_join_thread()
    q.put({"type": "FILLER"})

    class _AliveWriter:
        exitcode = None

        @staticmethod
        def is_alive():
            return True

    fake_self = types.SimpleNamespace(queue=q, writer=_AliveWriter)

    import threading

    done = threading.Event()

    def producer():
        rm.RecordingManager._put_blocking(fake_self, {"type": "FRAME"}, poll_s=0.05)
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    assert not done.wait(0.3), "must block while the queue is full, not drop"

    assert q.get()["type"] == "FILLER"  # drain one slot
    assert done.wait(3.0), "must deliver the frame once space appears"
    assert q.get()["type"] == "FRAME"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    raise SystemExit(1 if failures else 0)
