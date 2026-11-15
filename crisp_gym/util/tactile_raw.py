"""High-rate tactile CSVs recorded next to a LeRobot dataset (`tactile_raw/`).

The dataset stores ONE tactile sample per frame (15 Hz) under
`extra.tactile.<side>` together with `extra.tactile.<side>.source_stamp_ns`,
the stamp of the sample that was picked. The full stream (~200 Hz, with gaps)
sits in

    <dataset>/tactile_raw/episode_<NNNNNN>_<side>.csv

with the columns `episode_index, raw_episode_index, source_stamp_ns, ...,
taxel_0 .. taxel_<T-1>`. `source_stamp_ns` is int64 Unix-epoch nanoseconds and
is the SAME clock as the per-frame stamp in the dataset, so a frame is joined to
its CSV row by exact stamp equality — no value matching, no resampling of the
dataset. The number of taxels is read from the file.

Two things this module is careful about:

  * CAUSALITY. `windows()` holds the last sample AT OR BEFORE each grid time.
    It never interpolates towards a later sample, so a window is exactly what a
    process running online at that instant could have seen.
  * GAPS. The stream is bursty (median interval ~5 ms, single intervals of
    30 ms and more). Windows are therefore cut on a fixed TIME grid, not by row
    count, and the age of every held sample is returned so a caller can mask or
    weight stale values.

numpy + pandas only. No ROS, no crisp_py, no lerobot: runs on the GPU PC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STAMP_COLUMN = "source_stamp_ns"
TAXEL_PREFIX = "taxel_"


def side_of(key: str) -> str:
    """Dataset column -> the <side> used in the CSV file name.

    `extra.tactile.left` -> `left`; `left_tactile` / `extra.left_tactile` -> `left`.
    """
    last = key.split(".")[-1]
    for affix in ("_tactile", "tactile_"):
        last = last.replace(affix, "")
    return last


def csv_path(raw_dir: Path | str, episode: int, side: str) -> Path:
    """Path of one episode's stream for one sensor."""
    return Path(raw_dir) / f"episode_{int(episode):06d}_{side}.csv"


def load_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read one stream. Returns (stamps int64 [M], taxels float32 [M, T]).

    Rows are sorted by stamp (stable), taxel columns by their index.
    """
    import pandas as pd

    df = pd.read_csv(path, dtype={STAMP_COLUMN: "int64"})
    if STAMP_COLUMN not in df.columns:
        raise KeyError(f"{path}: no '{STAMP_COLUMN}' column. Columns: {list(df.columns)[:8]} ...")
    taxel_cols = [c for c in df.columns if c.startswith(TAXEL_PREFIX)]
    if not taxel_cols:
        raise KeyError(f"{path}: no '{TAXEL_PREFIX}*' columns.")
    taxel_cols.sort(key=lambda c: int(c[len(TAXEL_PREFIX):]))
    stamps = df[STAMP_COLUMN].to_numpy(np.int64)
    order = np.argsort(stamps, kind="stable")
    return stamps[order], df[taxel_cols].to_numpy(np.float32)[order]


def match_frames(stamps: np.ndarray, frame_stamps: np.ndarray) -> np.ndarray:
    """CSV row of each frame by EXACT stamp equality; -1 where there is none.

    Compare in int64: a stamp that went through float32 has a resolution of
    minutes and must not be accepted as a match.
    """
    stamps = np.asarray(stamps, dtype=np.int64)
    frame_stamps = np.asarray(frame_stamps, dtype=np.int64).reshape(-1)
    if len(stamps) == 0:
        return np.full(len(frame_stamps), -1, dtype=np.int64)
    i = np.clip(np.searchsorted(stamps, frame_stamps), 0, len(stamps) - 1)
    return np.where(stamps[i] == frame_stamps, i, -1)


def interval_stats(dt_ms: np.ndarray, *, gap_factor: float = 2.0) -> dict:
    """Statistics of sample intervals (ms), e.g. pooled over many episodes.

    `gap_time_fraction` is the share of the recording spent inside intervals
    longer than `gap_factor` x the median interval.
    """
    dt = np.asarray(dt_ms, dtype=np.float64)
    if len(dt) == 0:
        return {"intervals": 0}
    med = float(np.median(dt))
    total = float(dt.sum())
    return {
        "intervals": int(len(dt)),
        "mean_rate_hz": float(len(dt) / total * 1e3) if total > 0 else float("nan"),
        "dt_ms_median": med,
        "dt_ms_p99": float(np.percentile(dt, 99)),
        "dt_ms_max": float(dt.max()),
        "gap_time_fraction": float(dt[dt > gap_factor * med].sum() / total) if total > 0 else 0.0,
    }


def stream_stats(stamps: np.ndarray, *, gap_factor: float = 2.0) -> dict:
    """Rate and gaps of ONE stream (see `interval_stats`)."""
    stamps = np.asarray(stamps, dtype=np.int64)
    return interval_stats(np.diff(stamps) / 1e6, gap_factor=gap_factor)


def windows(
    stamps: np.ndarray,
    taxels: np.ndarray,
    end_stamps: np.ndarray,
    *,
    length: int,
    rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-length causal windows on a time grid ending at each `end_stamps`.

    Grid time j of window n is  end_stamps[n] - (length - 1 - j) / rate_hz, so
    the LAST column is the instant `end_stamps[n]` itself. Each grid point takes
    the last sample at or before it (zero-order hold).

    Returns:
        values  float32 [N, length, T]
        age_ms  float32 [N, length]   grid time minus the stamp of the held
                sample. Large = a gap in the stream. NEGATIVE = the grid point
                lies before the first sample of the stream (the first sample
                was used); treat those as padding.
    """
    stamps = np.asarray(stamps, dtype=np.int64)
    taxels = np.asarray(taxels)
    end_stamps = np.asarray(end_stamps, dtype=np.int64).reshape(-1)
    if length < 1:
        raise ValueError("length must be >= 1")
    if len(stamps) == 0:
        raise ValueError("empty stream")
    if len(stamps) != len(taxels):
        raise ValueError(f"{len(stamps)} stamps but {len(taxels)} taxel rows")
    period_ns = 1e9 / float(rate_hz)
    back = np.round((length - 1 - np.arange(length)) * period_ns).astype(np.int64)
    grid = end_stamps[:, None] - back[None, :]
    idx = np.searchsorted(stamps, grid, side="right") - 1
    idx = np.clip(idx, 0, len(stamps) - 1)
    age_ms = ((grid - stamps[idx]) / 1e6).astype(np.float32)
    return taxels[idx].astype(np.float32, copy=False), age_ms
