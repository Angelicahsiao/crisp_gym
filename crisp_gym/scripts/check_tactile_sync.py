r"""Check, for one episode, how the 15 Hz tactile columns line up with the high-rate CSVs.

For each sensor it prints:

  * the dtypes the dataset declares for the tactile column and its sync columns
    (`.source_stamp_ns`, `.sync_offset_ms`, `.sync_error_ms`, `.sync_valid`) —
    a stamp stored as float32 has a resolution of minutes and cannot be joined;
  * the non-taxel columns of the CSV, its row count, duration and interval
    statistics (median / p99 / max, in ms);
  * whether each frame's stamp is found in the CSV exactly (int64), how far the
    nearest CSV row is, and whether the taxels at that row equal the frame's;
  * CSV rows per frame, the share of `sync_valid` frames, and the sync offset
    and error percentiles.

It also prints the gap between the commanded and measured position and how
much the joints move: a handheld recording shows a placeholder command and
joints that never move.

One episode, in detail. For every episode at once (rate, gaps, join rate) use
`audit_command_residual.py DATASET --tactile-only`.

    python crisp_gym/scripts/check_tactile_sync.py ../../datasets/umi_electricbox/lerobot
    python crisp_gym/scripts/check_tactile_sync.py DATASET --episode 5 --sides left

numpy + pandas + pyarrow. No ROS, no crisp_py, no lerobot. This is the inline
check from the tactile investigation, made a script; on a dataset that has all
the columns it prints the same lines. Differences: a missing column or CSV is
reported instead of raising, and CSV rows are sorted by stamp before the join.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

SYNC_SUFFIXES = ("", ".source_stamp_ns", ".sync_offset_ms", ".sync_error_ms", ".sync_valid")


def read_episode(root: Path, episode: int, wanted: list[str]) -> pd.DataFrame:
    """The rows of one episode, only the columns that exist, sorted by frame."""
    import pandas as pd
    import pyarrow.parquet as pq

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    present = set(pq.ParquetFile(files[0]).schema_arrow.names)
    cols = ["episode_index", "frame_index"] + [c for c in wanted if c in present]
    parts = [
        pd.read_parquet(f, columns=cols, filters=[("episode_index", "==", episode)]) for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        raise ValueError(f"episode {episode} has no rows in {root / 'data'}")
    return df.sort_values("frame_index").reset_index(drop=True)


def stack(df: pd.DataFrame, column: str) -> np.ndarray:
    """A list/array column -> [frames, values]."""
    return np.stack([np.asarray(v).reshape(-1) for v in df[column]])


def nearest_rows(stamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the nearest stamp for each query, compared in float64."""
    t = stamps.astype(np.float64)
    q = np.asarray(query, dtype=np.float64)
    i = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    return i - (np.abs(t[i - 1] - q) < np.abs(t[i] - q))


def command_lines(df: pd.DataFrame, episode: int, pose: str, target: str, joints: str) -> str:
    """Commanded-vs-measured position gap and joint motion (robot vs handheld)."""
    parts = [f"episode {episode}: {len(df)} frames"]
    if pose in df and target in df:
        gap = np.linalg.norm(stack(df, target)[:, :3] - stack(df, pose)[:, :3], axis=1) * 1e3
        parts.append(
            f"command-measured position gap mm: median {np.median(gap):.2f} max {gap.max():.2f}"
        )
    else:
        parts.append(f"command-measured gap: needs {pose} and {target}")
    if joints in df:
        parts.append(f"joints moving: {stack(df, joints).std(0).max():.3f} rad std")
    else:
        parts.append(f"joints: no {joints} column")
    return " | ".join(parts)


def sensor_lines(
    df: pd.DataFrame,
    features: dict,
    raw_dir: Path,
    episode: int,
    key: str,
    side: str,
) -> list[str]:
    """Every line for one sensor."""
    import pandas as pd

    out = [
        f"--- {side}: dataset dtypes "
        + str({x or "taxels": (features.get(key + x) or {}).get("dtype") for x in SYNC_SUFFIXES})
    ]
    path = raw_dir / f"episode_{episode:06d}_{side}.csv"
    if not path.is_file():
        pointer = path.with_name(path.name + ".dvc")
        out.append(
            f"csv: {path} not found"
            + (" (only its .dvc pointer: pull it)" if pointer.is_file() else "")
        )
        return out
    csv = pd.read_csv(path, dtype={"source_stamp_ns": "int64"})
    out.append(f"csv non-taxel columns: {[c for c in csv.columns if not c.startswith('taxel_')]}")
    if "source_stamp_ns" not in csv:
        out.append("csv has no source_stamp_ns column: nothing to join on")
        return out
    csv = csv.sort_values("source_stamp_ns", kind="stable").reset_index(drop=True)
    t = csv["source_stamp_ns"].to_numpy(np.int64)
    if len(t) < 2:
        out.append(f"csv: {len(t)} row(s), too few to measure")
        return out
    dt = np.diff(t) / 1e6
    out.append(
        f"csv: {len(t)} rows over {(t[-1] - t[0]) / 1e9:.2f} s | dt ms median {np.median(dt):.2f} "
        f"p99 {np.percentile(dt, 99):.2f} max {dt.max():.2f}"
    )
    stamp_col = key + ".source_stamp_ns"
    if stamp_col not in df:
        out.append(f"dataset has no {stamp_col}: frames cannot be joined to the csv by stamp")
        return out
    st = stack(df, stamp_col)[:, 0]
    exact = np.isin(st.astype(np.int64), t).mean() * 100
    i = nearest_rows(t, st)
    err = np.abs(t[i].astype(np.float64) - st.astype(np.float64)) / 1e6
    line = (
        f"dataset stamp ({st.dtype}) found in csv: exact {exact:.0f}% | nearest row off by "
        f"median {np.median(err):.3f} max {err.max():.3f} ms"
    )
    taxels = csv.filter(like="taxel_").to_numpy(np.float64)
    if key in df:
        frames = stack(df, key).astype(np.float64)
        if frames.shape[1] == taxels.shape[1]:
            same = np.isclose(taxels[i], frames, atol=1e-3).all(1).mean() * 100
            line += f" | taxels equal at that row {same:.0f}%"
        else:
            line += f" | taxel counts differ: dataset {frames.shape[1]}, csv {taxels.shape[1]}"
    out.append(line)
    step = np.diff(i)
    tail = [f"csv rows per frame: median {np.median(step):.0f} min {step.min()} max {step.max()}"]
    if key + ".sync_valid" in df:
        tail.append(f"sync_valid {stack(df, key + '.sync_valid').mean() * 100:.0f}%")
    if key + ".sync_offset_ms" in df:
        off = stack(df, key + ".sync_offset_ms")
        tail.append(
            f"sync_offset_ms median {np.median(off):.2f} p99 {np.percentile(np.abs(off), 99):.2f}"
        )
    if key + ".sync_error_ms" in df:
        tail.append(
            f"sync_error_ms p99 {np.percentile(np.abs(stack(df, key + '.sync_error_ms')), 99):.2f}"
        )
    out.append(" | ".join(tail))
    return out


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", type=Path, help="LeRobot dataset root (contains data/ and meta/)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--sides", nargs="+", default=["left", "right"])
    p.add_argument(
        "--key-prefix",
        default="extra.tactile",
        help="tactile column of a side is <prefix>.<side> (default extra.tactile)",
    )
    p.add_argument("--raw-dir", default="tactile_raw", help="folder of the high-rate CSVs")
    p.add_argument("--pose-key", default="observation.state.cartesian")
    p.add_argument("--target-key", default="extra.target_cartesian")
    p.add_argument("--joints-key", default="extra.joints")
    return p


def main(argv: list[str] | None = None) -> int:
    """Print the report for one episode."""
    args = build_parser().parse_args(argv)
    features = json.loads((args.dataset / "meta" / "info.json").read_text()).get("features", {})
    keys = {side: f"{args.key_prefix}.{side}" for side in args.sides}
    wanted = [args.pose_key, args.target_key, args.joints_key]
    wanted += [k + x for k in keys.values() for x in SYNC_SUFFIXES]
    df = read_episode(args.dataset, args.episode, wanted)
    lines = [command_lines(df, args.episode, args.pose_key, args.target_key, args.joints_key)]
    for side, key in keys.items():
        lines += sensor_lines(df, features, args.dataset / args.raw_dir, args.episode, key, side)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
