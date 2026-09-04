"""Phase timing for the recording / deployment control loop (measurement only).

Why this exists
---------------
`RecordingManager.record_episode` is ONE thread that does everything: it calls
the teleop `drive_fn`, steps the env, collects the observation, hands the frame
to the dataset-writer process, and sleeps to hold the rate. The teleop command
rate is therefore *identical* to this loop's rate, and the same loop is used to
deploy policies (`scripts/deploy_policy.py` passes `policy.make_data_fn()`), so
a stall shows up as degraded teleop AND as degraded policy control.

When the loop misses its budget there are only three places the time can have
gone, and until now the log could not tell them apart:

    data   — producer work: drive_fn + env.step + observation collection
    put    — BACK-PRESSURE: the bounded queue to the writer process is full,
             so `queue.put()` blocks until the writer drains one slot
    sleep  — the loop is healthy and idling to hold the rate

This module records those phases and prints a per-episode summary. It changes
no control flow: it only reads clocks and appends floats.

Reading the summary
-------------------
* `put` dominating + `queue depth` pinned at capacity => writer-bound. The loop
  runs at the writer's throughput, not at `fps`. Fix the writer, not the loop.
* `data` dominating + queue depth ~0 => producer-bound (slow observation
  collection, slow leader read, blocking env.step).
* Writer-side `idle` near 0% => the writer never waits for work, i.e. it is
  saturated; near 100% => the writer is starved and the producer is the limit.

Stdlib only, no ROS/numpy/lerobot imports — safe to import from the writer
subprocess and from ROS-free machines.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)

__all__ = ["PhaseStats", "LoopTimingRecorder", "WriterTimingRecorder", "percentile"]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an ALREADY SORTED sequence (q in [0,1])."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


class PhaseStats:
    """Running samples of one phase, summarized in milliseconds."""

    def __init__(self, name: str) -> None:
        """Create an empty accumulator for the phase called `name`."""
        self.name = name
        self.samples: List[float] = []

    def add(self, seconds: float) -> None:
        """Record one observation of this phase, in seconds."""
        self.samples.append(seconds)

    @property
    def total(self) -> float:
        """Total time spent in this phase, in seconds."""
        return math.fsum(self.samples)

    def summary_ms(self) -> Dict[str, float]:
        """Return count/mean/p50/p95/max/total for this phase (ms, except count)."""
        if not self.samples:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "total": 0.0}
        ordered = sorted(self.samples)
        return {
            "count": len(ordered),
            "mean": 1e3 * math.fsum(ordered) / len(ordered),
            "p50": 1e3 * percentile(ordered, 0.50),
            "p95": 1e3 * percentile(ordered, 0.95),
            "max": 1e3 * ordered[-1],
            "total": 1e3 * math.fsum(ordered),
        }


class LoopTimingRecorder:
    """Per-frame phase timing for the record/deploy loop (producer side).

    One instance per episode. `add_frame` is called once per loop iteration and
    does nothing but append floats; the formatting happens in `log_summary`.
    """

    #: Phases always present. Sub-phases (drive/step/collect/action) are picked
    #: up opportunistically from `data_fn.timing` when the data function
    #: exposes one (see `record/record_functions.py::make_record_fn`).
    BASE_PHASES = ("data", "put", "sleep", "total")
    SUB_PHASES = ("drive", "step", "collect", "action")
    #: `time.sleep(x)` returning late. Non-zero here means the thread is not
    #: getting the CPU back on time (GIL contention with the ROS executor
    #: threads, or plain core oversubscription) — a rate loss that has nothing
    #: to do with the writer and is invisible in the other phases.
    EXTRA_PHASES = ("oversleep",)

    def __init__(
        self,
        label: str,
        budget_s: float,
        queue_capacity: int,
        enabled: bool = True,
        csv_dir: str | Path | None = None,
    ) -> None:
        """Set up a recorder for one episode.

        Args:
            label: Human-readable episode label used in log lines / CSV names.
            budget_s: Per-frame time budget (1 / fps).
            queue_capacity: Size of the writer queue, to report depth as n/max.
            enabled: When False every method is a no-op.
            csv_dir: Directory to dump a per-frame CSV into. None = no CSV.
        """
        self.label = label
        self.budget_s = budget_s
        self.queue_capacity = queue_capacity
        self.enabled = enabled
        self.csv_dir = Path(csv_dir) if csv_dir is not None else None

        self.phases: Dict[str, PhaseStats] = {
            name: PhaseStats(name)
            for name in (*self.BASE_PHASES, *self.SUB_PHASES, *self.EXTRA_PHASES)
        }
        self.rows: List[Dict[str, float]] = []
        self.queue_depths: List[int] = []
        self.frames = 0
        self.skipped = 0
        self.late_frames = 0
        self.queue_full_frames = 0
        self.overrun_total = 0.0
        self.wall_start = time.perf_counter()

    def add_frame(
        self,
        *,
        data_s: float,
        put_s: float,
        sleep_s: float,
        total_s: float,
        queue_depth: int,
        sleep_requested_s: float = 0.0,
        sub_timing: Dict[str, float] | None = None,
        skipped: bool = False,
    ) -> None:
        """Record one loop iteration. Cheap: appends, no formatting, no I/O."""
        if not self.enabled:
            return

        if skipped:
            self.skipped += 1
        else:
            self.frames += 1

        self.phases["data"].add(data_s)
        self.phases["put"].add(put_s)
        self.phases["sleep"].add(sleep_s)
        self.phases["total"].add(total_s)

        if sub_timing:
            for name in self.SUB_PHASES:
                if name in sub_timing:
                    self.phases[name].add(sub_timing[name])

        if sleep_requested_s > 0.0:
            self.phases["oversleep"].add(max(0.0, sleep_s - sleep_requested_s))

        if queue_depth >= 0:
            self.queue_depths.append(queue_depth)
            if self.queue_capacity > 0 and queue_depth >= self.queue_capacity:
                self.queue_full_frames += 1

        work_s = total_s - sleep_s
        if work_s > self.budget_s:
            self.late_frames += 1
            self.overrun_total += work_s - self.budget_s

        row: Dict[str, float] = {
            "index": len(self.rows),
            "t_rel_s": time.perf_counter() - self.wall_start,
            "data_ms": 1e3 * data_s,
            "put_ms": 1e3 * put_s,
            "sleep_ms": 1e3 * sleep_s,
            "total_ms": 1e3 * total_s,
            "queue_depth": queue_depth,
            "skipped": int(skipped),
            "sleep_requested_ms": 1e3 * sleep_requested_s,
            "oversleep_ms": 1e3 * max(0.0, sleep_s - sleep_requested_s)
            if sleep_requested_s > 0.0
            else 0.0,
        }
        for name in self.SUB_PHASES:
            row[f"{name}_ms"] = 1e3 * (sub_timing.get(name, 0.0) if sub_timing else 0.0)
        self.rows.append(row)

    # ── reporting ────────────────────────────────────────────────────────────

    def dominant_phase(self) -> str:
        """Whichever of `data` / `put` consumed more time (sleep excluded)."""
        candidates = {name: self.phases[name].total for name in ("data", "put")}
        return max(candidates, key=lambda k: candidates[k])

    def effective_fps(self) -> float:
        """Frames actually delivered per second of wall clock."""
        return self.frames / max(time.perf_counter() - self.wall_start, 1e-9)

    def verdict(self) -> str:
        """One-line diagnosis of this episode.

        Order matters: back-pressure and slow producer work are checked first
        because they explain late frames. A loop that misses its rate with NO
        late frame and an empty queue is a third, distinct failure — it slept
        too long — and must not be reported as healthy just because every
        individual frame fit its budget.
        """
        if self.queue_full_frames > 0 and self.dominant_phase() == "put":
            return (
                "WRITER-BOUND — the loop is blocked handing frames to the "
                "dataset writer, so the teleop/policy rate is capped by writer "
                "throughput, not by fps."
            )
        if self.late_frames > 0:
            return (
                "PRODUCER-BOUND — frames overran the budget inside data_fn "
                "(leader read / env.step / observation collection)."
            )

        target_fps = 1.0 / self.budget_s if self.budget_s > 0 else 0.0
        achieved = self.effective_fps()
        if target_fps > 0 and achieved < 0.95 * target_fps:
            oversleep = self.phases["oversleep"].summary_ms()
            return (
                f"RATE MISS — every frame fit its budget and the writer queue "
                f"never filled, yet only {achieved:.2f} of {target_fps:.2f} FPS "
                f"was delivered: time.sleep() returned "
                f"{oversleep['mean']:.1f} ms late on average "
                f"(p95 {oversleep['p95']:.1f} ms). The loop thread is not "
                "getting the CPU back on time — GIL contention with the ROS "
                "executor threads in this process, or CPU oversubscription."
            )
        return (
            "HEALTHY — the loop held its rate; no frame overran the budget "
            "and the writer queue never filled."
        )

    def log_summary(self, extra: str = "") -> None:
        """Log the episode summary at INFO. Safe to call on an empty episode."""
        if not self.enabled or not self.rows:
            return

        elapsed = max(time.perf_counter() - self.wall_start, 1e-9)
        # `oversleep` is a COMPONENT of `sleep`, so it is excluded from the
        # share denominator (data + put + sleep == total).
        effective_fps = self.effective_fps()
        target_fps = 1.0 / self.budget_s if self.budget_s > 0 else 0.0
        # Shares are taken against the sum of per-frame totals rather than wall
        # clock, so they add up to ~100% and stay meaningful even if the
        # recorder outlives the loop it measured.
        accounted_ms = max(1e3 * self.phases["total"].total, 1e-9)

        lines = [
            f"[timing] {self.label}: {self.frames} frames "
            f"({self.skipped} skipped) in {elapsed:.1f} s "
            f"-> {effective_fps:.2f} FPS (target {target_fps:.2f}, "
            f"budget {1e3 * self.budget_s:.1f} ms/frame)",
            "[timing]   phase        mean     p50     p95     max    share",
        ]
        for name in ("data", "drive", "step", "collect", "action", "put", "sleep", "oversleep"):
            stats = self.phases[name].summary_ms()
            if stats["count"] == 0:
                continue
            share = 100.0 * stats["total"] / accounted_ms
            indent = "  " if name in (*self.SUB_PHASES, *self.EXTRA_PHASES) else ""
            lines.append(
                f"[timing]   {indent}{name}".ljust(24)
                + f"{stats['mean']:7.1f} {stats['p50']:7.1f} "
                f"{stats['p95']:7.1f} {stats['max']:7.1f}  {share:5.1f}%"
            )

        if self.queue_depths:
            mean_depth = math.fsum(self.queue_depths) / len(self.queue_depths)
            pct_full = 100.0 * self.queue_full_frames / len(self.queue_depths)
            lines.append(
                f"[timing]   writer queue depth: mean {mean_depth:.1f} "
                f"max {max(self.queue_depths)}/{self.queue_capacity} "
                f"— full on {self.queue_full_frames}/{len(self.queue_depths)} "
                f"frames ({pct_full:.0f}%)"
            )
        else:
            lines.append("[timing]   writer queue depth: unavailable on this platform")

        if self.frames + self.skipped:
            pct_late = 100.0 * self.late_frames / (self.frames + self.skipped)
            lines.append(
                f"[timing]   late frames: {self.late_frames}/{self.frames + self.skipped} "
                f"({pct_late:.0f}%), total overrun {self.overrun_total:.2f} s"
            )

        lines.append(f"[timing]   VERDICT: {self.verdict()}")
        if extra:
            lines.append(f"[timing]   {extra}")

        logger.info("\n".join(lines))
        self._dump_csv()

    def _dump_csv(self) -> None:
        """Write the per-frame rows to `csv_dir` (no-op when unset)."""
        if self.csv_dir is None or not self.rows:
            return
        try:
            self.csv_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.label)
            path = self.csv_dir / f"loop_timing_{safe}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)
            logger.info(f"[timing] per-frame trace written to {path}")
        except Exception as exc:  # never let instrumentation break recording
            logger.warning(f"[timing] could not write the per-frame CSV: {exc}")


class WriterTimingRecorder:
    """Per-message timing for the dataset-writer process (consumer side).

    Complements `LoopTimingRecorder`: `idle` (time spent waiting in
    `queue.get()`) is the decisive number. Near 0% means the writer never runs
    out of work — it is the bottleneck and the producer is blocked behind it.
    """

    def __init__(self, budget_s: float, enabled: bool = True) -> None:
        """Args: budget_s = 1/fps, the per-frame time the writer must beat."""
        self.budget_s = budget_s
        self.enabled = enabled
        self.frame = PhaseStats("add_frame")
        self.idle = PhaseStats("idle")
        self.reset_window()

    def reset_window(self) -> None:
        """Start a fresh reporting window (called after each episode summary)."""
        self.frame = PhaseStats("add_frame")
        self.idle = PhaseStats("idle")
        self.window_start = time.perf_counter()

    def _started(self) -> bool:
        """True once this window has seen its first frame."""
        return bool(self.frame.samples)

    def add_frame(self, seconds: float) -> None:
        """Record the writer's total cost for one frame.

        Spans building the LeRobot frame dict AND `dataset.add_frame` — i.e.
        everything the writer must do per frame, which is what has to beat the
        budget for the recording loop never to block.
        """
        if self.enabled:
            self.frame.add(seconds)

    def add_idle(self, seconds: float) -> None:
        """Record time the writer spent blocked in `queue.get()`.

        Only counted once the window has a frame: the writer idles for as long
        as the operator takes to press `r`, and folding that wait into the
        window would report an idle share (and a sustained FPS) that describes
        the operator, not the writer.
        """
        if self.enabled and self._started():
            self.idle.add(seconds)

    def log_summary(self, label: str) -> None:
        """Log the writer-side summary for the current window, then reset it."""
        if not self.enabled or not self.frame.samples:
            return
        stats = self.frame.summary_ms()
        # Denominator = the writer's own accounted time (draining frames plus
        # waiting for them). Wall clock would also include save_episode, which
        # runs BETWEEN episodes and would understate per-frame throughput.
        accounted = max(self.frame.total + self.idle.total, 1e-9)
        sustained = stats["count"] / accounted
        idle_share = min(100.0, 100.0 * self.idle.total / accounted)
        budget_ms = 1e3 * self.budget_s

        # The decisive comparison is per-frame COST vs budget, not idle share:
        # a writer that idles 44% but needs 62 ms of a 67 ms budget has ~5 ms
        # of headroom and will block the loop on the first hiccup.
        headroom_ms = budget_ms - stats["mean"]
        if stats["mean"] > budget_ms:
            assessment = (
                f"OVER BUDGET by {-headroom_ms:.1f} ms — cannot sustain "
                f"{1.0 / self.budget_s:.1f} FPS, the loop WILL block"
            )
        elif headroom_ms < 0.2 * budget_ms:
            assessment = (
                f"TIGHT — only {headroom_ms:.1f} ms of headroom per frame; "
                "any jitter fills the queue"
            )
        elif idle_share < 10:
            assessment = "SATURATED — the writer is the bottleneck"
        else:
            assessment = f"{headroom_ms:.1f} ms of headroom per frame"

        logger.info(
            f"[timing/writer] {label}: {int(stats['count'])} frames — "
            f"per-frame (build + add_frame) mean {stats['mean']:.1f} / "
            f"p50 {stats['p50']:.1f} / "
            f"p95 {stats['p95']:.1f} / max {stats['max']:.1f} ms "
            f"(budget {budget_ms:.1f} ms) -> sustained "
            f"{sustained:.2f} FPS; idle waiting for frames {idle_share:.0f}% "
            f"({assessment})"
        )
        self.reset_window()
