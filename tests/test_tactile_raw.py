"""Tests for crisp_gym/util/tactile_raw.py — the high-rate tactile CSV loader.

The properties that matter downstream are pinned on hand-built streams:

  * a window never contains a sample from AFTER its grid time (causal hold),
  * gaps show up as age, the stretch before the stream starts as NEGATIVE age,
  * frames are joined by exact int64 stamp, and a stamp that lost precision
    (float32) is NOT accepted as a match,
  * the taxel count comes from the file.

Run:  python tests/test_tactile_raw.py   (or via pytest)
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


def _load_by_path(name, path):
    """Import the file at `path` as module `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # registered first: dataclasses and pickle look a module up by its name
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


REPO = Path(__file__).resolve().parents[1]
MS = 1_000_000
T0 = 1_794_190_351_560_399_616  # a real Unix-epoch ns stamp: above 2**53


def _load():
    return _load_by_path("tactile_raw", str(REPO / "crisp_gym" / "util" / "tactile_raw.py"))


def test_side_of():
    mod = _load()
    assert mod.side_of("extra.tactile.left") == "left"
    assert mod.side_of("left_tactile") == "left"
    assert mod.side_of("extra.right_tactile") == "right"
    assert mod.csv_path("/d/tactile_raw", 7, "left").name == "episode_000007_left.csv"


def test_window_is_a_causal_hold_on_a_time_grid():
    mod = _load()
    # samples at 0, 5, 10, 40, 45 ms (a 30 ms gap); value = its own time in ms
    t_ms = np.array([0, 5, 10, 40, 45])
    stamps = T0 + t_ms * MS
    taxels = np.repeat(t_ms[:, None], 3, axis=1).astype(np.float32)
    end = np.array([T0 + 45 * MS, T0 + 22 * MS])
    values, age = mod.windows(stamps, taxels, end, length=4, rate_hz=100.0)
    assert values.shape == (2, 4, 3) and age.shape == (2, 4)
    # window 0, grid 15/25/35/45 ms -> held samples 10/10/10/45
    assert values[0, :, 0].tolist() == [10, 10, 10, 45]
    assert np.allclose(age[0], [5, 15, 25, 0])
    # window 1, grid -8/2/12/22 ms: before the stream -> first sample, negative age
    assert values[1, :, 0].tolist() == [0, 0, 10, 10]
    assert np.allclose(age[1], [-8, 2, 2, 12])
    # never a sample from after the grid time
    grid_ms = (end[:, None] - T0) / MS - (3 - np.arange(4)) * 10.0
    assert np.all((values[..., 0] <= grid_ms) | (age < 0))


def test_window_ends_exactly_on_the_frame_sample():
    """The last column is the sample the dataset stored for that frame."""
    mod = _load()
    rng = np.random.default_rng(0)
    stamps = T0 + np.cumsum(rng.integers(4 * MS, 9 * MS, 500))
    taxels = rng.normal(size=(500, 72)).astype(np.float32)
    pick = np.arange(20, 480, 23)
    values, age = mod.windows(stamps, taxels, stamps[pick], length=16, rate_hz=200.0)
    assert values.shape == (len(pick), 16, 72)
    assert np.array_equal(values[:, -1], taxels[pick])
    assert np.all(age[:, -1] == 0) and np.all(age >= 0)


def test_match_frames_is_exact_in_int64():
    mod = _load()
    stamps = T0 + np.arange(0, 1000 * 5 * MS, 5 * MS)
    frames = stamps[[3, 500, 999]]
    assert mod.match_frames(stamps, frames).tolist() == [3, 500, 999]
    # one nanosecond off, before the first, after the last: no match
    assert mod.match_frames(stamps, frames + 1).tolist() == [-1, -1, -1]
    assert mod.match_frames(stamps, [stamps[0] - 5, stamps[-1] + 5]).tolist() == [-1, -1]
    # float32 round trip has a resolution of minutes at this magnitude
    assert (mod.match_frames(stamps, frames.astype(np.float32).astype(np.int64)) == -1).all()
    assert mod.match_frames(np.empty(0, np.int64), frames).tolist() == [-1, -1, -1]


def test_interval_stats_report_gaps():
    mod = _load()
    dt = np.r_[np.full(90, 5.0), np.full(10, 30.0)]  # 450 ms regular + 300 ms in gaps
    s = mod.interval_stats(dt)
    assert s["dt_ms_median"] == 5.0 and s["dt_ms_max"] == 30.0
    assert abs(s["gap_time_fraction"] - 300 / 750) < 1e-9
    assert abs(s["mean_rate_hz"] - 100 / 0.75) < 1e-6
    assert mod.interval_stats(np.empty(0)) == {"intervals": 0}
    stamps = T0 + np.cumsum(np.r_[0, dt] * MS).astype(np.int64)
    assert mod.stream_stats(stamps)["dt_ms_max"] == 30.0


def test_load_csv_sorts_rows_and_taxel_columns():
    try:
        import pandas as pd
    except ImportError:
        print("skip: pandas not installed")
        return
    mod = _load()
    stamps = T0 + np.array([10, 0, 5]) * MS  # rows out of order
    df = pd.DataFrame({"episode_index": 0, "source_stamp_ns": stamps, "frame_id": "left"})
    for j in (10, 2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 11):  # taxel_10 before taxel_2
        df[f"taxel_{j}"] = np.array([10, 0, 5]) * 100 + j
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episode_000000_left.csv"
        df.to_csv(path, index=False)
        got_stamps, taxels = mod.load_csv(path)
    assert got_stamps.dtype == np.int64 and got_stamps.tolist() == sorted(stamps.tolist())
    assert taxels.shape == (3, 12) and taxels.dtype == np.float32
    assert taxels[0].tolist() == list(range(12))  # t=0 row, columns in taxel order
    assert taxels[2].tolist() == [1000 + j for j in range(12)]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL OK")
