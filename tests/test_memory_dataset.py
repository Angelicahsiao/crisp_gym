"""crisp_gym/scripts/memory_dataset.py: a low-dimensional dataset held in memory.

The in-memory dataset must return EXACTLY what LeRobotDataset returns for every
index: the same keys, dtypes, shapes and values, including the delta-timestamp
windows clamped at episode edges, their `_is_pad` masks, and the task string.
Checked against the real LeRobotDataset on a small dataset written here, whole
and episode-filtered. Needs torch (the refusal and flag tests) and lerobot (the
comparisons); each test is skipped where its package is missing.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crisp_gym" / "scripts"))

FPS = 10
EPISODE_LENGTHS = (4, 1, 6)  # a one-frame episode: every window is padding but its own frame
TASKS = ("press the button", "slide on the panel", "press the button")
DELTA_TIMESTAMPS = {
    "observation.state": [-2 / FPS, -1 / FPS, 0.0],
    "action": [i / FPS for i in range(-1, 4)],
}


def _memory_dataset():
    pytest.importorskip("torch")
    import memory_dataset

    return memory_dataset


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    pytest.importorskip("lerobot")
    np = pytest.importorskip("numpy")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path_factory.mktemp("mem") / "toy"
    features = {
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.gripper": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="local/toy", fps=FPS, features=features, root=root, use_videos=False
    )
    task_argument = "task" in inspect.signature(ds.add_frame).parameters
    value = 0.0
    for length, task in zip(EPISODE_LENGTHS, TASKS, strict=True):
        for _ in range(length):
            frame = {
                "observation.state": np.array([value, value + 0.5, -value], dtype=np.float32),
                "observation.gripper": np.array([value / 10], dtype=np.float32),
                "action": np.array([2 * value, value - 1], dtype=np.float32),
            }
            value += 1.0
            if task_argument:
                ds.add_frame(frame, task=task)
            else:
                ds.add_frame({**frame, "task": task})
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    return root


def _load(root, **kwargs):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("local/toy", root=root, delta_timestamps=DELTA_TIMESTAMPS, **kwargs)


def _assert_same_item(expected, got, idx):
    import torch

    assert set(got) == set(expected), (idx, set(got) ^ set(expected))
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert isinstance(got[key], torch.Tensor), (idx, key)
            assert got[key].dtype == value.dtype, (idx, key, got[key].dtype, value.dtype)
            assert got[key].shape == value.shape, (idx, key, got[key].shape, value.shape)
            assert torch.equal(got[key], value), (idx, key, got[key], value)
        else:
            assert got[key] == value, (idx, key)


@pytest.mark.lerobot
def test_every_item_matches_lerobots_including_windows_and_padding(dataset_root):
    md = _memory_dataset()
    original = _load(dataset_root)
    memory = md.InMemoryDataset(original)
    assert len(memory) == len(original) == sum(EPISODE_LENGTHS)
    for idx in range(len(original)):
        _assert_same_item(original[idx], memory[idx], idx)
    # the edges really are exercised: padded steps at both ends, and both tasks
    first, last = memory[0], memory[len(memory) - 1]
    assert first["observation.state_is_pad"].tolist() == [True, True, False]
    assert last["action_is_pad"].tolist() == [False, False, True, True, True]
    assert {memory[i]["task"] for i in range(len(memory))} == set(TASKS)


@pytest.mark.lerobot
def test_an_episode_filtered_dataset_matches_too(dataset_root):
    md = _memory_dataset()
    original = _load(dataset_root, episodes=[0, 2])
    memory = md.InMemoryDataset(original)
    assert len(memory) == len(original) == EPISODE_LENGTHS[0] + EPISODE_LENGTHS[2]
    for idx in range(len(original)):
        _assert_same_item(original[idx], memory[idx], idx)


@pytest.mark.lerobot
def test_items_are_independent_of_the_cache(dataset_root):
    # a training transform that edits an item in place must not change the cache
    md = _memory_dataset()
    memory = md.InMemoryDataset(_load(dataset_root))
    before = memory[2]["observation.state"].clone()
    memory[2]["observation.state"].add_(100.0)
    memory[2]["observation.gripper"].add_(100.0)
    assert memory[2]["observation.state"].equal(before)


@pytest.mark.lerobot
def test_everything_else_is_the_original_datasets(dataset_root):
    # lerobot_train reads meta, num_frames, episodes ... from the dataset it gets
    md = _memory_dataset()
    original = _load(dataset_root)
    memory = md.InMemoryDataset(original)
    assert memory.meta is original.meta
    assert memory.num_frames == original.num_frames
    assert memory.num_episodes == original.num_episodes
    assert memory.episodes == original.episodes
    assert memory.absolute_to_relative_idx == original.absolute_to_relative_idx
    with pytest.raises(AttributeError):
        memory.no_such_attribute  # noqa: B018


def test_a_dataset_with_cameras_is_refused_before_anything_is_read():
    md = _memory_dataset()

    class Unreadable:
        meta = SimpleNamespace(camera_keys=["observation.images.wrist"])

        @property
        def hf_dataset(self):
            raise AssertionError("read the table of a dataset it should refuse")

    with pytest.raises(ValueError, match="observation.images.wrist"):
        md.InMemoryDataset(Unreadable())


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--a=1", "--in-memory", "--b"], (True, ["--a=1", "--b"])),
        (["--a=1", "--b"], (False, ["--a=1", "--b"])),
        (["--in-memory", "--in-memory"], (True, [])),
    ],
)
def test_the_flag_is_taken_out_before_lerobot_parses_the_rest(argv, expected):
    md = _memory_dataset()
    assert md.pop_in_memory_flag(argv) == expected
