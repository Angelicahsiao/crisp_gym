"""Keep CPython's garbage collector out of the control loop.

Why
---
Per-frame traces of the recording loop show a stall of 95-161 ms roughly every
17-20 frames, on top of a ~5 ms floor. The floor is exactly
`sys.getswitchinterval()` (0.005 s by default): the loop thread's sleep
expires, another thread holds the GIL, and it is only handed over at the next
switch check. That fixes the MECHANISM — the loop is waiting for the GIL, not
for work of its own (`data_fn` is 1-2 ms and `queue.put` is 0.0 ms on those
frames).

The big stalls are a generational GC pass. A 1 Hz ROS diagnostics timer was
ruled out: the observed interval is 1.27-1.43 s, not 1.0 s. Two things point
at gen2 collections instead — the interval tracks ALLOCATIONS rather than time
(the episode doing more work per frame had a 12% shorter interval measured in
frames), and the magnitude is what a full sweep of a heap holding multi-MB
camera frames costs. A collection triggered by ANY thread holds the GIL for its
whole duration, which is why it lands on the loop as oversleep.

What this does
--------------
`gc.freeze()` moves everything alive at entry into a permanent generation that
collections never scan again — the env, its ROS nodes, the loaded lerobot
modules. Raising the thresholds then makes a gen2 pass far rarer. Together:
rare, and cheap when it does happen.

Correctness: this changes only WHEN reference cycles are reclaimed, never
whether. Refcounting still frees the per-frame arrays immediately — they are
not cyclic. Cyclic garbage accumulates for longer, so the collector is run once
on exit, between episodes, where a pause costs nothing.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)

#: gen0 / gen1 / gen2 thresholds. CPython's defaults are (700, 10, 10), so a
#: gen2 pass happens about every 700*10*10 = 70,000 surviving allocations —
#: every ~18 frames at the measured allocation rate. These make that ~70x
#: rarer while keeping gen0 frequent enough that it stays cheap.
DEFAULT_THRESHOLDS: Tuple[int, int, int] = (2000, 50, 50)


@contextmanager
def reduced_gc_pauses(
    enabled: bool = True,
    thresholds: Tuple[int, int, int] = DEFAULT_THRESHOLDS,
) -> Iterator[None]:
    """Freeze the existing heap and defer collections for the duration.

    Args:
        enabled: When False this is a no-op, so the loop can be A/B tested
            against the same build.
        thresholds: gen0/gen1/gen2 thresholds to install.
    """
    if not enabled:
        yield
        return

    previous = gc.get_threshold()
    # Collect first so the freeze captures a tidy heap rather than pinning
    # garbage in the permanent generation for the rest of the process.
    gc.collect()
    gc.freeze()
    gc.set_threshold(*thresholds)
    logger.info(
        f"GC tuned for the control loop: {gc.get_freeze_count()} objects frozen "
        f"out of collection, thresholds {previous} -> {thresholds}. "
        "Set reduce_gc_pauses: false to compare against stock behaviour."
    )
    try:
        yield
    finally:
        gc.set_threshold(*previous)
        gc.unfreeze()
        # Between episodes: pay for the deferred cycles where a pause is free.
        collected = gc.collect()
        logger.debug(f"GC restored to {previous}; {collected} objects collected.")
