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
    captured = {}

    class _RGBEncoderConfig:
        def __init__(self, **kw):
            captured.update(kw)
            self.vcodec = kw.get("vcodec", "libsvtav1")
            self.crf = kw.get("crf", 30)
            self.g = kw.get("g", 2)

    configs = sys.modules.setdefault("lerobot.configs", types.ModuleType("lerobot.configs"))
    configs.RGBEncoderConfig = _RGBEncoderConfig

    fake_self = types.SimpleNamespace(config=_config(vcodec="auto", video_crf=21))
    encoder = rm.RecordingManager._rgb_encoder(fake_self)
    assert encoder is not None
    assert captured == {"vcodec": "auto", "crf": 21}, captured
    # video_gop left None must NOT be forced — lerobot's g=2 keeps the training
    # dataloader's random-access seeking fast.
    assert "g" not in captured


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
