"""Tests for the record/deploy loop instrumentation (util/loop_timing.py).

`RecordingManager.record_episode` is one thread that drives teleop, steps the
env, collects the observation, hands the frame to the writer process and sleeps
to hold the rate — so the teleop (and, on the deploy path, the policy) command
rate IS this loop's rate. When it drops, the time went to exactly one of:

    data  — producer work (drive_fn / env.step / observation collection)
    put   — back-pressure: the bounded queue to the writer process is full
    sleep — healthy idling

These tests pin that the recorder attributes each regime correctly, including
against a REAL multiprocessing queue with a deliberately slow consumer, and
that `make_record_fn` publishes the per-phase breakdown the loop reads.

Run:  python tests/test_loop_timing.py   (or via pytest)
"""

import multiprocessing as mp
import sys
import time
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from crisp_gym.util.loop_timing import (  # noqa: E402
    LoopTimingRecorder,
    WriterTimingRecorder,
    percentile,
)

# ── percentile ───────────────────────────────────────────────────────────────


def test_percentile_edges():
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([0.0, 1.0], 0.5) == 0.5
    assert percentile([0.0, 10.0, 20.0], 1.0) == 20.0
    assert percentile([0.0, 10.0, 20.0], 0.0) == 0.0


# ── producer-side recorder ───────────────────────────────────────────────────


def _slow_consumer(queue, per_frame_s):
    """Stand-in for the writer process: a fixed cost per frame."""
    while True:
        msg = queue.get()
        if msg["type"] == "SHUTDOWN":
            queue.task_done()
            break
        time.sleep(per_frame_s)
        queue.task_done()


def _run_loop(label, fps, writer_per_frame_s, data_per_frame_s, n, queue_size=8):
    """Mirror record_episode's structure against a real mp.JoinableQueue."""
    queue = mp.JoinableQueue(queue_size)
    proc = mp.Process(target=_slow_consumer, args=(queue, writer_per_frame_s), daemon=True)
    proc.start()
    try:
        budget = 1.0 / fps
        rec = LoopTimingRecorder(label=label, budget_s=budget, queue_capacity=queue_size)
        for _ in range(n):
            frame_start = time.time()
            t_frame = time.perf_counter()

            t_mark = time.perf_counter()
            time.sleep(data_per_frame_s)  # stands in for data_fn()
            data_s = time.perf_counter() - t_mark

            depth = queue.qsize()
            t_mark = time.perf_counter()
            queue.put({"type": "FRAME", "data": b"x"})
            put_s = time.perf_counter() - t_mark

            sleep_time = budget - (time.time() - frame_start)
            t_mark = time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            sleep_s = time.perf_counter() - t_mark if sleep_time > 0 else 0.0

            rec.add_frame(
                data_s=data_s,
                put_s=put_s,
                sleep_s=sleep_s,
                total_s=time.perf_counter() - t_frame,
                queue_depth=depth,
            )
        return rec
    finally:
        queue.put({"type": "SHUTDOWN"})
        proc.join(timeout=10)


def test_writer_bound_loop_is_diagnosed_as_back_pressure():
    """A writer slower than the frame budget fills the queue and blocks put()."""
    rec = _run_loop("writer_bound", fps=15, writer_per_frame_s=0.090, data_per_frame_s=0.005, n=40)
    assert rec.dominant_phase() == "put"
    assert rec.queue_full_frames > 0, "queue should have hit capacity"
    assert "WRITER-BOUND" in rec.verdict()


def test_producer_bound_loop_is_not_blamed_on_the_queue():
    """A slow data_fn with a fast writer must never read as back-pressure."""
    rec = _run_loop(
        "producer_bound", fps=15, writer_per_frame_s=0.001, data_per_frame_s=0.080, n=20
    )
    assert rec.dominant_phase() == "data"
    assert rec.queue_full_frames == 0
    assert "PRODUCER-BOUND" in rec.verdict()


def test_rate_miss_is_not_reported_as_healthy():
    """Regression: episode_0000 of the field run delivered 12.29 of 15 FPS with
    ZERO late frames and an empty queue — time.sleep() simply returned ~14 ms
    late every frame. The old verdict called that HEALTHY and hid a 2.7 FPS
    loss. A rate miss must be named, and must name oversleep as the cause.
    """
    rec = LoopTimingRecorder(label="rate_miss", budget_s=1 / 15, queue_capacity=64)
    rec.wall_start = time.perf_counter() - 60.9  # 748 frames took 60.9 s
    for _ in range(748):
        rec.add_frame(
            data_s=0.0017,
            put_s=0.0,
            sleep_s=0.0795,  # measured
            total_s=0.0812,
            queue_depth=0,
            sleep_requested_s=0.0650,  # asked for
        )
    assert rec.late_frames == 0, "no frame overran its budget"
    assert rec.queue_full_frames == 0, "the queue never filled"
    assert 12.0 < rec.effective_fps() < 12.6
    assert "RATE MISS" in rec.verdict(), rec.verdict()
    assert "sleep" in rec.verdict()
    oversleep = rec.phases["oversleep"].summary_ms()
    assert 14.0 < oversleep["mean"] < 15.0, oversleep


def test_oversleep_is_not_charged_when_no_sleep_was_requested():
    """A late frame sleeps zero; that is an overrun, not an overshoot."""
    rec = LoopTimingRecorder(label="no_sleep", budget_s=1 / 15, queue_capacity=64)
    rec.add_frame(
        data_s=0.001, put_s=0.150, sleep_s=0.0, total_s=0.151, queue_depth=64, sleep_requested_s=0.0
    )
    assert rec.phases["oversleep"].samples == []


def test_writer_bound_wins_over_rate_miss():
    """Back-pressure must be reported even though the rate is also missed."""
    rec = LoopTimingRecorder(label="both", budget_s=1 / 15, queue_capacity=64)
    rec.wall_start = time.perf_counter() - 24.2
    for _ in range(112):
        rec.add_frame(
            data_s=0.0017,
            put_s=0.1611,
            sleep_s=0.0474,
            total_s=0.2102,
            queue_depth=64,
            sleep_requested_s=0.0,
        )
    assert "WRITER-BOUND" in rec.verdict(), rec.verdict()


def test_healthy_loop_reports_healthy():
    rec = _run_loop("healthy", fps=15, writer_per_frame_s=0.005, data_per_frame_s=0.005, n=20)
    assert rec.late_frames == 0
    assert rec.queue_full_frames == 0
    assert "HEALTHY" in rec.verdict()


def test_phase_shares_sum_to_about_one_hundred_percent():
    """Shares are taken against recorded frame time, not wall clock."""
    rec = LoopTimingRecorder(label="shares", budget_s=1 / 15, queue_capacity=8)
    for _ in range(10):
        rec.add_frame(data_s=0.010, put_s=0.020, sleep_s=0.030, total_s=0.060, queue_depth=0)
    total = sum(rec.phases[p].total for p in ("data", "put", "sleep"))
    assert abs(total - rec.phases["total"].total) < 1e-9


def test_late_frames_ignore_sleep():
    """A frame is late only if its WORK exceeded the budget — sleep never counts."""
    rec = LoopTimingRecorder(label="late", budget_s=0.100, queue_capacity=8)
    rec.add_frame(data_s=0.010, put_s=0.005, sleep_s=0.085, total_s=0.100, queue_depth=0)
    assert rec.late_frames == 0
    rec.add_frame(data_s=0.010, put_s=0.150, sleep_s=0.0, total_s=0.160, queue_depth=8)
    assert rec.late_frames == 1
    assert abs(rec.overrun_total - 0.060) < 1e-6


def test_unavailable_queue_depth_is_not_recorded():
    """qsize() is unimplemented on some platforms; -1 must not pollute stats."""
    rec = LoopTimingRecorder(label="nodepth", budget_s=1 / 15, queue_capacity=8)
    rec.add_frame(data_s=0.001, put_s=0.001, sleep_s=0.060, total_s=0.062, queue_depth=-1)
    assert rec.queue_depths == []
    rec.log_summary()  # must not raise


def test_disabled_recorder_is_a_no_op():
    rec = LoopTimingRecorder(label="off", budget_s=1 / 15, queue_capacity=8, enabled=False)
    rec.add_frame(data_s=1.0, put_s=1.0, sleep_s=0.0, total_s=2.0, queue_depth=5)
    rec.log_summary()
    assert rec.rows == [] and rec.frames == 0


def test_csv_trace_has_one_row_per_frame(tmp_path=None):
    import csv
    import tempfile

    out = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    rec = LoopTimingRecorder(label="episode_0001", budget_s=1 / 15, queue_capacity=8, csv_dir=out)
    for i in range(5):
        rec.add_frame(
            data_s=0.001 * i,
            put_s=0.002,
            sleep_s=0.010,
            total_s=0.020,
            queue_depth=i,
            sub_timing={"drive": 0.0001, "step": 0.0002, "collect": 0.0003},
        )
    rec.log_summary()
    rows = list(csv.DictReader((out / "loop_timing_episode_0001.csv").open()))
    assert len(rows) == 5
    assert rows[0]["queue_depth"] == "0"
    assert {"data_ms", "put_ms", "sleep_ms", "drive_ms", "collect_ms"} <= set(rows[0])


# ── writer-side recorder ─────────────────────────────────────────────────────


def test_writer_idle_before_the_first_frame_is_not_counted():
    """Regression: the writer idles while the operator decides when to press
    `r`. Folding that into the window reported "sustained 9.04 FPS / 44% idle
    / has headroom" for a writer that actually needed 62 of its 67 ms budget.
    """
    rec = WriterTimingRecorder(budget_s=1 / 15)
    rec.add_idle(8.0)  # operator has not pressed `r` yet
    assert rec.idle.samples == [], "pre-episode wait must not enter the window"
    rec.add_frame(0.0623)
    rec.add_idle(0.004)  # a real between-frames wait
    assert rec.idle.samples == [0.004]


def test_writer_near_budget_is_reported_as_tight_not_as_headroom():
    """62.3 ms of a 66.7 ms budget is 4.4 ms of slack, not 'has headroom'."""
    import io
    import logging as _logging

    from crisp_gym.util import loop_timing as lt

    rec = WriterTimingRecorder(budget_s=1 / 15)
    for _ in range(748):
        rec.add_frame(0.0623)
        rec.add_idle(0.048)  # 44% idle, as in the field run

    stream = io.StringIO()
    handler = _logging.StreamHandler(stream)
    lt.logger.addHandler(handler)
    lt.logger.setLevel(_logging.INFO)
    try:
        rec.log_summary("episode")
    finally:
        lt.logger.removeHandler(handler)
    out = stream.getvalue()
    assert "TIGHT" in out, out
    assert "has headroom" not in out and "ms of headroom per frame)" not in out


def test_writer_over_budget_is_called_out():
    rec = WriterTimingRecorder(budget_s=1 / 15)
    for _ in range(112):
        rec.add_frame(0.0936)  # episode_0001 of the field run
        rec.add_idle(0.001)
    import io
    import logging as _logging

    from crisp_gym.util import loop_timing as lt

    stream = io.StringIO()
    handler = _logging.StreamHandler(stream)
    lt.logger.addHandler(handler)
    lt.logger.setLevel(_logging.INFO)
    try:
        rec.log_summary("episode")
    finally:
        lt.logger.removeHandler(handler)
    assert "OVER BUDGET" in stream.getvalue(), stream.getvalue()


def test_saturated_writer_reports_low_idle():
    rec = WriterTimingRecorder(budget_s=1 / 15)
    for _ in range(30):
        rec.add_idle(0.001)
        rec.add_frame(0.070)
    assert rec.idle.total / (rec.idle.total + rec.frame.total) < 0.10
    rec.log_summary("saturated")
    assert rec.frame.samples == [] and rec.idle.samples == []  # window reset


def test_starved_writer_reports_high_idle():
    rec = WriterTimingRecorder(budget_s=1 / 15)
    for _ in range(30):
        rec.add_idle(0.060)
        rec.add_frame(0.005)
    assert rec.idle.total / (rec.idle.total + rec.frame.total) > 0.50


# ── make_record_fn publishes the sub-phase breakdown ─────────────────────────


def _stub_crisp_py():
    """Stub the one crisp_py symbol record_config touches, as the other
    numpy-only tests in this suite do (record_config._pose_to_array imports
    OrientationRepresentation lazily). Mirrors the real enum — keep in sync,
    see crisp_py/HANDOFF.md 1.1.
    """
    for name in ("crisp_py", "crisp_py.utils", "crisp_py.utils.geometry"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class _OrientationRepresentation:
        EULER = "euler"
        QUATERNION = "quaternion"
        ANGLE_AXIS = "angle_axis"
        ROTATION_6D = "rotation_6d"

        def __init__(self, value):
            self.value = value

    sys.modules["crisp_py.utils.geometry"].OrientationRepresentation = _OrientationRepresentation


def _load_record_functions():
    """Import record_functions with only its ROS-free deps present."""
    _stub_crisp_py()
    import crisp_gym.record.record_functions as rf

    return rf


class _FakeCamera:
    def __init__(self, name):
        self.config = types.SimpleNamespace(camera_name=name)

    @property
    def current_image(self):
        import numpy as np

        time.sleep(0.002)  # observation collection is the expensive phase
        return np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeEnv:
    """Minimal env: a tcp pose, a gripper and one camera."""

    def __init__(self):
        import numpy as np

        self.task = "t"
        self.cameras = [_FakeCamera("primary")]
        self.sensors = []
        self.gripper = types.SimpleNamespace(value=0.5)
        self.robot = types.SimpleNamespace(
            end_effector_pose=types.SimpleNamespace(
                to_array=lambda representation=None: np.zeros(9, dtype=np.float32)
            )
        )
        self.steps = 0

    def step(self, action, block=False):
        self.steps += 1
        time.sleep(0.004)  # env.step also builds a full observation it discards
        return {}, 0.0, False, False, {}


def test_make_record_fn_publishes_per_phase_timing():
    """record_episode reads data_fn.timing to split a slow frame apart."""
    rf = _load_record_functions()
    _stub_crisp_py()
    rc = SourceFileLoader(
        "rc_loop_timing", str(REPO / "crisp_gym" / "record" / "record_config.py")
    ).load_module()

    config = rc.RecordConfig(
        name="t",
        rate_hz=15.0,
        observations=[
            rc.ObsFieldConfig(
                key="observation.state.cartesian",
                source="robot.tcp_pose",
                params={"representation": "rotation_6d"},
            ),
            rc.ObsFieldConfig(
                key="observation.images.primary",
                source="camera.image",
                params={"camera": "primary"},
                shape=[4, 4, 3],
            ),
        ],
        action=rc.ActionConfig(
            definition="next_tcp_pose",
            lookahead=1,
            representation="rotation_6d",
            include_gripper=False,
        ),
    )

    env = _FakeEnv()
    drive_calls = {"n": 0}

    def drive_fn():
        drive_calls["n"] += 1
        time.sleep(0.001)
        return None if drive_calls["n"] == 1 else [0.0]

    fn = rf.make_record_fn(env, config, drive_fn=drive_fn)
    assert hasattr(fn, "timing"), "make_record_fn must publish .timing"

    # 1st tick: drive warm-up — only `drive` is charged, env is not stepped.
    obs, action = fn()
    assert obs is None and action is None
    assert fn.timing["drive"] > 0.0
    assert fn.timing["step"] == 0.0 and fn.timing["collect"] == 0.0
    assert env.steps == 0

    # 2nd tick: lookahead buffer still filling, but the env IS stepped and the
    # observation IS collected, so both phases must be charged.
    fn()
    assert env.steps == 1
    assert fn.timing["step"] > 0.0
    assert fn.timing["collect"] > 0.0

    # 3rd tick: a full frame comes out and every phase is populated.
    obs, action = fn()
    assert obs is not None and action is not None
    assert fn.timing["collect"] >= 0.002, "camera read should dominate collect"
    assert fn.timing["step"] >= 0.004, "env.step should be charged to step"


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
