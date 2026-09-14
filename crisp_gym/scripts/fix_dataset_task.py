#!/usr/bin/env python3
"""Inspect and correct the task strings of a LeRobot dataset, in place.

WHY THIS EXISTS
    The task string is not decoration. For a language-conditioned policy
    (SmolVLA and other VLAs) it is a model INPUT: the checkpoint learns to map
    that exact sentence onto a behavior, and at deployment the same sentence has
    to come back. A dataset recorded under a stale default -- the hazard that
    `--tasks` being optional used to create -- therefore trains the wrong
    mapping, and the damage is silent: nothing fails, the policy simply learns
    the task under a name nobody will ever say to it again.

WHERE THE TASK LIVES (LeRobot v3.0)
    Four places, and they must agree:
      meta/tasks.parquet     the string <-> task_index table
      meta/info.json         total_tasks (a COUNT, not the strings -- this is
                             why "check info.json" does not answer "what task?")
      meta/episodes/*.parquet  a per-episode `tasks` column, when present
      data/**/*.parquet      a `task_index` column on every FRAME
    Editing one and not the others is what produces a dataset that loads but
    trains on a label you did not intend.

OPERATIONS
    --list              show every task, its index, and how many frames and
                        episodes use it. Run this first; it is read-only.
    --validate          check the four locations agree (exit 1 if not).
    --rename OLD NEW    fix one string, keeping its task_index. Frame data is
                        untouched, so this is the cheap, safe repair for a
                        mislabelled single task.
    --set-all NEW       collapse every task to one. REWRITES task_index on every
                        frame -- use only when the dataset really is one task
                        recorded under several names.

Usage:
    python fix_dataset_task.py <dataset_root> --list
    python fix_dataset_task.py <dataset_root> --rename "pick the lego block." "open the power switch"
    python fix_dataset_task.py <dataset_root> --set-all "open the power switch" --dry-run

<dataset_root> is the directory holding meta/ and data/ (e.g. .../franka_electricbox/lerobot).

If the dataset is DVC-tracked, `dvc status` should be clean before you start and
you should `dvc commit` afterwards -- editing a workspace whose recorded hashes
no longer match is how a dataset ends up half-corrected across two machines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── layout helpers ────────────────────────────────────────────────────────────


def data_files(root: Path) -> list[Path]:
    """Every frame-data parquet, layout-agnostic (v2.x per-episode or v3.0 packed)."""
    return sorted((root / "data").rglob("*.parquet"))


def episode_files(root: Path) -> list[Path]:
    """Every per-episode metadata parquet under meta/episodes/."""
    return sorted((root / "meta" / "episodes").rglob("*.parquet"))


def load_info(root: Path) -> dict:
    """meta/info.json as a dict."""
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def save_info(root: Path, info: dict) -> None:
    """Write meta/info.json back, preserving LeRobot's 4-space indent."""
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def read_tasks(root: Path) -> tuple[pd.DataFrame, bool]:
    """The task table as {task, task_index} columns, plus how it was stored.

    LeRobot writes tasks.parquet with the task string as the INDEX and
    task_index as a column. Older/hand-built datasets sometimes carry `task` as
    a plain column instead. Normalize here and restore the original shape on
    write, so this script never silently changes the file's layout.

    Returns:
        (frame with `task` and `task_index` columns, task_was_the_index)
    """
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found — is {root} a LeRobot dataset root?")

    df = pd.read_parquet(path)
    if "task" in df.columns:
        return df.reset_index(drop=True)[["task", "task_index"]], False
    # task string is the index
    out = df.reset_index()
    index_name = out.columns[0]
    out = out.rename(columns={index_name: "task"})
    return out[["task", "task_index"]], True


def write_tasks(root: Path, df: pd.DataFrame, task_was_index: bool) -> None:
    """Write the task table back in the layout it was read in."""
    path = root / "meta" / "tasks.parquet"
    if task_was_index:
        df.set_index("task").to_parquet(path)
    else:
        df.to_parquet(path, index=False)


# ── read-only reporting ───────────────────────────────────────────────────────


def frame_counts(root: Path) -> Counter:
    """How many frames reference each task_index."""
    counts: Counter = Counter()
    for path in data_files(root):
        df = pd.read_parquet(path, columns=["task_index"])
        counts.update(df["task_index"].tolist())
    return counts


def episode_task_strings(root: Path) -> Counter:
    """How many episodes name each task string, from meta/episodes/*.parquet."""
    counts: Counter = Counter()
    for path in episode_files(root):
        df = pd.read_parquet(path)
        if "tasks" not in df.columns:
            continue
        for value in df["tasks"]:
            if value is None:
                continue
            if isinstance(value, str):
                counts[value] += 1
            else:
                for task in value:
                    counts[str(task)] += 1
    return counts


def report(root: Path) -> tuple[pd.DataFrame, Counter, Counter]:
    """Print every task with its frame and episode usage; flag dangling references."""
    tasks, _ = read_tasks(root)
    info = load_info(root)
    frames = frame_counts(root)
    episodes = episode_task_strings(root)

    print(f"\n=== {root}")
    print(f"  info.json total_tasks : {info.get('total_tasks')}")
    print(f"  info.json total_frames: {info.get('total_frames')}")
    print(f"  tasks.parquet rows    : {len(tasks)}\n")
    print(f"  {'idx':>4}  {'frames':>9}  {'episodes':>8}  task")
    for _, row in tasks.iterrows():
        idx = int(row["task_index"])
        print(
            f"  {idx:>4}  {frames.get(idx, 0):>9}  "
            f"{episodes.get(row['task'], 0):>8}  {row['task']!r}"
        )

    orphan_idx = sorted(set(frames) - set(tasks["task_index"].astype(int)))
    if orphan_idx:
        print(f"\n  ** frames reference task_index not in tasks.parquet: {orphan_idx}")
    orphan_str = sorted(set(episodes) - set(tasks["task"]))
    if orphan_str:
        print(f"  ** episodes name task strings not in tasks.parquet: {orphan_str}")
    return tasks, frames, episodes


def validate(root: Path) -> bool:
    """True when tasks.parquet, info.json, the episodes parquet and data/ agree."""
    tasks, frames, episodes = report(root)
    info = load_info(root)
    ok = True

    if info.get("total_tasks") != len(tasks):
        print(
            f"\n  ** info.json total_tasks ({info.get('total_tasks')}) != "
            f"tasks.parquet rows ({len(tasks)})"
        )
        ok = False
    if set(frames) - set(tasks["task_index"].astype(int)):
        ok = False
    if set(episodes) - set(tasks["task"]):
        ok = False
    unused = sorted(set(tasks["task_index"].astype(int)) - set(frames))
    if unused:
        print(f"  -- task_index with no frames (harmless but stale): {unused}")

    print("\n  OK — task metadata is self-consistent." if ok else "\n  ** INCONSISTENT")
    return ok


# ── mutations ─────────────────────────────────────────────────────────────────


def rename_task(root: Path, old: str, new: str, dry_run: bool) -> None:
    """Change one task string, preserving its task_index.

    Frame data is never touched: task_index is the only thing frames store, and
    it does not move. That makes this repair cheap and hard to get wrong.
    """
    tasks, was_index = read_tasks(root)
    if old not in set(tasks["task"]):
        raise SystemExit(
            f"task {old!r} not found. Present: {sorted(tasks['task'])}"
        )
    if new in set(tasks["task"]) and new != old:
        raise SystemExit(
            f"task {new!r} already exists — renaming {old!r} onto it would merge "
            "two indices, which this flag does not do. Use --set-all if the "
            "dataset really is one task."
        )

    idx = int(tasks.loc[tasks["task"] == old, "task_index"].iloc[0])
    print(f"\n  rename task_index {idx}: {old!r} -> {new!r}")
    if dry_run:
        print("  [DRY RUN] nothing written")
        return

    tasks.loc[tasks["task"] == old, "task"] = new
    write_tasks(root, tasks, was_index)
    print("  meta/tasks.parquet updated")

    for path in episode_files(root):
        df = pd.read_parquet(path)
        if "tasks" not in df.columns:
            continue
        changed = False

        def _swap(value):  # noqa: ANN001, ANN202
            nonlocal changed
            if value is None:
                return value
            if isinstance(value, str):
                if value == old:
                    changed = True
                    return new
                return value
            # pandas hands back a numpy array for a parquet list column; return
            # a plain list either way — parquet round-trips both identically.
            out = [new if str(t) == old else t for t in value]
            if out != [str(t) for t in value]:
                changed = True
            return out

        df["tasks"] = [_swap(v) for v in df["tasks"]]
        if changed:
            df.to_parquet(path, index=False)
            print(f"  meta/episodes/{path.name} updated")

    print("  data/ untouched (task_index unchanged) — this is correct")


def set_all_tasks(root: Path, new: str, dry_run: bool) -> None:
    """Collapse every task to a single one at index 0.

    This DOES rewrite every frame's task_index, so it is the expensive and more
    dangerous operation. Only correct when the dataset genuinely holds one
    behavior that was recorded under several names.
    """
    tasks, was_index = read_tasks(root)
    print(f"\n  collapse {len(tasks)} task(s) -> 1: {new!r}")
    for _, row in tasks.iterrows():
        print(f"    was index {int(row['task_index'])}: {row['task']!r}")
    if dry_run:
        print("  [DRY RUN] nothing written")
        return

    write_tasks(root, pd.DataFrame({"task": [new], "task_index": [0]}), was_index)
    print("  meta/tasks.parquet updated")

    for path in data_files(root):
        df = pd.read_parquet(path)
        if "task_index" in df.columns and (df["task_index"] != 0).any():
            df["task_index"] = 0
            df.to_parquet(path, index=False)
            print(f"  data/{path.name}: task_index -> 0")

    for path in episode_files(root):
        df = pd.read_parquet(path)
        if "tasks" not in df.columns:
            continue
        sample = next((v for v in df["tasks"] if v is not None), None)
        df["tasks"] = [new if isinstance(sample, str) else [new]] * len(df)
        df.to_parquet(path, index=False)
        print(f"  meta/episodes/{path.name} updated")

    info = load_info(root)
    info["total_tasks"] = 1
    save_info(root, info)
    print("  meta/info.json total_tasks -> 1")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path, help="Dataset root holding meta/ and data/")
    ap.add_argument("--list", action="store_true", help="Show tasks and usage counts.")
    ap.add_argument("--validate", action="store_true", help="Check the four locations agree.")
    ap.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"),
                    help="Rename one task, keeping its task_index.")
    ap.add_argument("--set-all", metavar="NEW",
                    help="Collapse every task to one. Rewrites task_index on every frame.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    if not (args.root / "meta" / "info.json").exists():
        raise SystemExit(f"{args.root}/meta/info.json not found — not a dataset root.")

    if args.rename and args.set_all:
        raise SystemExit("--rename and --set-all are mutually exclusive.")

    if args.rename:
        report(args.root)
        rename_task(args.root, args.rename[0], args.rename[1], args.dry_run)
        if not args.dry_run:
            sys.exit(0 if validate(args.root) else 1)
    elif args.set_all:
        report(args.root)
        set_all_tasks(args.root, args.set_all, args.dry_run)
        if not args.dry_run:
            sys.exit(0 if validate(args.root) else 1)
    elif args.validate:
        sys.exit(0 if validate(args.root) else 1)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
