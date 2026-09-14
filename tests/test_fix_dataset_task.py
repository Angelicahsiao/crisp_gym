"""Tests for scripts/fix_dataset_task.py.

The task string is a model input for a VLA, so a repair that updates one of the
four locations and not the others produces a dataset that loads fine and trains
on the wrong label. These tests pin that all four stay in agreement, under both
tasks.parquet layouts LeRobot datasets appear with in the wild (task as the
index, which is what LeRobot writes, and task as a plain column).

Run: python -m pytest tests/test_fix_dataset_task.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fix_dataset_task", REPO / "crisp_gym" / "scripts" / "fix_dataset_task.py"
)
fdt = importlib.util.module_from_spec(_spec)
sys.modules["fix_dataset_task"] = fdt
_spec.loader.exec_module(fdt)


def _make_dataset(root: Path, tasks: list[str], frames_per_task: int = 3,
                  task_as_index: bool = True, episode_tasks_as_list: bool = True) -> Path:
    """A minimal v3.0-shaped dataset with `tasks` distinct task strings."""
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rows, episodes = [], []
    for task_index, task in enumerate(tasks):
        for i in range(frames_per_task):
            rows.append({
                "task_index": task_index,
                "episode_index": task_index,
                "frame_index": i,
                "index": task_index * frames_per_task + i,
            })
        episodes.append({
            "episode_index": task_index,
            "tasks": [task] if episode_tasks_as_list else task,
            "length": frames_per_task,
        })
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(episodes).to_parquet(
        root / "meta/episodes/chunk-000/file-000.parquet", index=False
    )

    table = pd.DataFrame({"task": tasks, "task_index": list(range(len(tasks)))})
    if task_as_index:
        table.set_index("task").to_parquet(root / "meta/tasks.parquet")
    else:
        table.to_parquet(root / "meta/tasks.parquet", index=False)

    json.dump(
        {"total_tasks": len(tasks), "total_frames": len(rows), "total_episodes": len(tasks)},
        open(root / "meta/info.json", "w"),
        indent=4,
    )
    return root


# ── reading ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task_as_index", [True, False])
def test_read_tasks_handles_both_layouts(tmp_path, task_as_index):
    root = _make_dataset(tmp_path / "ds", ["a", "b"], task_as_index=task_as_index)
    df, was_index = fdt.read_tasks(root)
    assert was_index is task_as_index
    assert list(df["task"]) == ["a", "b"]
    assert list(df["task_index"]) == [0, 1]


@pytest.mark.parametrize("task_as_index", [True, False])
def test_write_preserves_the_original_layout(tmp_path, task_as_index):
    """A repair must not silently restructure the file it repaired."""
    root = _make_dataset(tmp_path / "ds", ["a"], task_as_index=task_as_index)
    fdt.rename_task(root, "a", "b", dry_run=False)
    raw = pd.read_parquet(root / "meta/tasks.parquet")
    assert ("task" in raw.columns) is (not task_as_index)


def test_validate_accepts_a_consistent_dataset(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["open the power switch"])
    assert fdt.validate(root) is True


def test_validate_rejects_a_stale_total_tasks(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["a", "b"])
    info = fdt.load_info(root)
    info["total_tasks"] = 5
    fdt.save_info(root, info)
    assert fdt.validate(root) is False


def test_validate_rejects_frames_pointing_at_a_missing_task(tmp_path):
    """The failure mode that matters: a frame referencing an index nobody defines."""
    root = _make_dataset(tmp_path / "ds", ["a"])
    p = root / "data/chunk-000/file-000.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "task_index"] = 7
    df.to_parquet(p, index=False)
    assert fdt.validate(root) is False


# ── rename ────────────────────────────────────────────────────────────────────


def test_rename_updates_every_location_but_not_frame_data(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["pick the lego block.", "other"])
    before = pd.read_parquet(root / "data/chunk-000/file-000.parquet")

    fdt.rename_task(root, "pick the lego block.", "open the power switch", dry_run=False)

    tasks, _ = fdt.read_tasks(root)
    assert set(tasks["task"]) == {"open the power switch", "other"}
    # index preserved -> frame data must be byte-identical
    assert int(tasks.loc[tasks["task"] == "open the power switch", "task_index"].iloc[0]) == 0
    after = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    pd.testing.assert_frame_equal(before, after)
    # episodes follow the string
    eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    assert list(eps["tasks"].iloc[0]) == ["open the power switch"]
    assert fdt.validate(root) is True


def test_rename_handles_a_scalar_episode_tasks_column(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["old"], episode_tasks_as_list=False)
    fdt.rename_task(root, "old", "new", dry_run=False)
    eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    assert eps["tasks"].iloc[0] == "new"


def test_rename_dry_run_writes_nothing(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["old"])
    fdt.rename_task(root, "old", "new", dry_run=True)
    tasks, _ = fdt.read_tasks(root)
    assert list(tasks["task"]) == ["old"]


def test_rename_refuses_an_unknown_task(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["a"])
    with pytest.raises(SystemExit, match="not found"):
        fdt.rename_task(root, "nope", "b", dry_run=False)


def test_rename_refuses_to_silently_merge(tmp_path):
    """Renaming onto an existing name would merge two indices — refuse, don't guess."""
    root = _make_dataset(tmp_path / "ds", ["a", "b"])
    with pytest.raises(SystemExit, match="already exists"):
        fdt.rename_task(root, "a", "b", dry_run=False)


# ── set-all ───────────────────────────────────────────────────────────────────


def test_set_all_collapses_indices_and_frames(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["a", "b", "c"])
    fdt.set_all_tasks(root, "open the power switch", dry_run=False)

    tasks, _ = fdt.read_tasks(root)
    assert list(tasks["task"]) == ["open the power switch"]
    assert list(tasks["task_index"]) == [0]

    frames = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    assert set(frames["task_index"]) == {0}
    assert fdt.load_info(root)["total_tasks"] == 1

    eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    assert all(list(v) == ["open the power switch"] for v in eps["tasks"])
    assert fdt.validate(root) is True


def test_set_all_dry_run_writes_nothing(tmp_path):
    root = _make_dataset(tmp_path / "ds", ["a", "b"])
    fdt.set_all_tasks(root, "x", dry_run=True)
    tasks, _ = fdt.read_tasks(root)
    assert list(tasks["task"]) == ["a", "b"]
    assert fdt.load_info(root)["total_tasks"] == 2
