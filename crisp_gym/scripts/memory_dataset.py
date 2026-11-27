"""A low-dimensional LeRobotDataset held in memory, for fast training.

Runs on the GPU PC, imported by lerobot_relative_pose.py when it is given
`--in-memory` (it sits beside it, so it imports as a sibling module). Imports
torch only; the dataset it wraps is the trainer's LeRobotDataset.
Tests: tests/test_memory_dataset.py.

Why: lerobot 0.6.1 reads every delta-timestamp window through the Hugging Face
dataset, column by column, for every item. On the simulated press-button data
(20,111 frames, no cameras) that is ~11 ms per item, so a batch of 320 takes
~0.45 s with 8 workers against 0.06 s of GPU work: the GPU waited ~92 % of
every step (sheet 5, 2026-11-27). Here the columns are read ONCE into tensors
and an item is a few tensor lookups.

What it returns is exactly LeRobotDataset's item (DatasetReader.get_item): the
row's columns, every delta-timestamp key as a window clamped to its episode,
`<key>_is_pad` for the steps outside it, and the task string. Anything else
(meta, num_frames, episodes, ...) is the wrapped dataset's own attribute.

Datasets with cameras are refused: their frames are decoded from video or
images per item, which is the slow part there, and holding them in memory is
not what this does.
"""

from __future__ import annotations

from typing import Any

import torch

FLAG = "--in-memory"


def pop_in_memory_flag(argv: list[str]) -> tuple[bool, list[str]]:
    """Take `--in-memory` out of argv (ours, not lerobot's). Returns (given, the rest)."""
    rest = [a for a in argv if a != FLAG]
    return len(rest) != len(argv), rest


class InMemoryDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset without cameras; items come from tensors in memory."""

    def __init__(self, dataset: Any) -> None:
        """Read every column of `dataset` (a LeRobotDataset) once; refuse one with cameras."""
        cameras = list(getattr(dataset.meta, "camera_keys", []) or [])
        if cameras:
            raise ValueError(
                f"--in-memory holds only datasets without cameras; this one has {cameras}. "
                "Train without --in-memory."
            )
        self._dataset = dataset
        table = dataset.hf_dataset[:]  # one read, already converted to torch by lerobot
        self._columns = {}
        for key, values in table.items():
            if values and all(isinstance(v, torch.Tensor) for v in values):
                self._columns[key] = torch.stack(values)
            else:
                self._columns[key] = list(values)
        self._delta = {
            key: torch.tensor(deltas, dtype=torch.long)
            for key, deltas in (dataset.reader.delta_indices or {}).items()
        }
        self._relative = dataset.absolute_to_relative_idx  # None unless episode-filtered
        episodes = sorted({int(e) for e in self._columns["episode_index"].tolist()})
        self._bounds = {
            ep: (
                int(dataset.meta.episodes[ep]["dataset_from_index"]),
                int(dataset.meta.episodes[ep]["dataset_to_index"]),
            )
            for ep in episodes
        }
        self._tasks = {i: name for i, name in enumerate(dataset.meta.tasks.index)}

    def __len__(self) -> int:
        """Frames in the wrapped dataset."""
        return len(self._columns["index"])

    def __getitem__(self, idx: int) -> dict:
        """The item LeRobotDataset returns for `idx`, built from the tensors in memory."""
        item = {key: _row(column, idx) for key, column in self._columns.items()}
        abs_idx = int(item["index"])
        start, end = self._bounds[int(item["episode_index"])]
        for key, deltas in self._delta.items():
            wanted = abs_idx + deltas
            item[f"{key}_is_pad"] = (wanted < start) | (wanted >= end)
        for key, deltas in self._delta.items():
            rows = (abs_idx + deltas).clamp(start, end - 1)
            if self._relative is not None:
                rows = torch.tensor([self._relative[int(r)] for r in rows], dtype=torch.long)
            item[key] = self._columns[key][rows]
        item["task"] = self._tasks[int(item["task_index"])]
        return item

    def __getattr__(self, name: str) -> Any:
        """Everything else (meta, num_frames, episodes, ...) is the wrapped dataset's."""
        try:
            inner = self.__dict__["_dataset"]
        except KeyError:  # during unpickling, before __dict__ is filled
            raise AttributeError(name) from None
        return getattr(inner, name)


def _row(column: torch.Tensor | list, idx: int) -> Any:
    """One row of a column, as a tensor of its own (an in-place edit must not reach the cache)."""
    if isinstance(column, torch.Tensor):
        return column[idx].clone()
    return column[idx]
