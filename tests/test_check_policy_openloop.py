"""check_policy_openloop.py: what each policy type is fed per frame.

Before this, the frame builder kept only `observation*` keys, so the dataset's
task string never reached the preprocessor. multi_task_dit then fails with a
shape error (its conditioning size includes the text features) and GR00T is
evaluated without its instruction. It also filled `policy._queues` and stacked
`observation.images` unconditionally, which GR00T has neither of.

The script imports lerobot_relative_pose (and through it torch) at module
level, so this loads it with a stand-in for that module, and drives the frame
builder with numpy in place of torch. Needs numpy only.
"""

from __future__ import annotations

import sys
import types
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "crisp_gym" / "scripts" / "check_policy_openloop.py"

TASK = "open the electrical box"
OBS_IMAGES = "observation.images"
CAM = "observation.images.oakw_cam"


def _load():
    stub = types.ModuleType("lerobot_relative_pose")
    stub.POSE_ACTION_KEY = "action"
    saved = sys.modules.get("lerobot_relative_pose")
    sys.modules["lerobot_relative_pose"] = stub
    try:
        return SourceFileLoader("check_policy_openloop_under_test", str(SCRIPT)).load_module()
    finally:
        if saved is None:
            sys.modules.pop("lerobot_relative_pose", None)
        else:
            sys.modules["lerobot_relative_pose"] = saved


ol = _load()


def _item(n_obs=2):
    """A dataset item as RelativePoseDataset returns it (numpy instead of torch)."""
    return {
        "observation.state": np.arange(n_obs * 16, dtype=float).reshape(n_obs, 16),
        CAM: np.ones((n_obs, 3, 8, 8)),
        "action": np.zeros((16, 10)),          # the recorded chunk: never a policy input
        "task": TASK,
        "task_index": np.array(0),
        "episode_index": np.array(3),
    }


def _to_batch(v):
    return np.asarray(v, dtype=float)[None]


class _RecordingPre:
    """Stands in for the preprocessor: records its input, injects an action
    (as lerobot's does) and tokens when a task is present (as a tokenizer does)."""

    def __init__(self):
        self.seen = None

    def __call__(self, frame):
        self.seen = dict(frame)
        out = dict(frame)
        out["action"] = np.zeros((1, 10))
        if "task" in frame:
            out["observation.language.tokens"] = np.ones((1, 77))
        return out


# ── task_for_frame ───────────────────────────────────────────────────────────

def test_task_string_becomes_a_batch_of_one():
    assert ol.task_for_frame({"task": TASK}) == [TASK]


def test_task_list_is_kept():
    assert ol.task_for_frame({"task": [TASK]}) == [TASK]


@pytest.mark.parametrize("value", [None, 3, "", [], [1, 2]])
def test_missing_or_non_string_task_gives_none(value):
    item = {} if value is None else {"task": value}
    # An empty string is still a string the dataset stored; keep it.
    expected = [""] if value == "" else None
    assert ol.task_for_frame(item) == expected


# ── which policies get queues / stacked images ──────────────────────────────

class _Diffusion:          # and multi_task_dit: queues built in reset()
    def __init__(self, with_images=True):
        self._queues = {"observation.state": deque(), "action": deque()}
        if with_images:
            self._queues[OBS_IMAGES] = deque()


class _Groot:              # only an internal action queue
    def __init__(self):
        self._action_queue = deque()


def test_queue_based_policies_are_recognised():
    assert ol.policy_obs_queues(_Diffusion()) is not None
    assert ol.policy_obs_queues(_Groot()) is None


def test_queues_that_are_none_count_as_absent():
    p = _Diffusion()
    p._queues = None
    assert ol.policy_obs_queues(p) is None


def test_images_are_stacked_only_for_policies_with_an_image_queue():
    assert ol.stacks_images(_Diffusion(), [CAM], OBS_IMAGES) is True
    assert ol.stacks_images(_Diffusion(with_images=False), [CAM], OBS_IMAGES) is False
    assert ol.stacks_images(_Groot(), [CAM], OBS_IMAGES) is False
    assert ol.stacks_images(_Diffusion(), [], OBS_IMAGES) is False


# ── build_batch_frame ────────────────────────────────────────────────────────

def test_task_reaches_the_preprocessor():
    """The fix itself: tokenizers read `task` from the preprocessor input."""
    pre = _RecordingPre()
    ol.build_batch_frame(_item(), 1, 2, skip_key="action", to_batch_tensor=_to_batch, pre=pre)
    assert pre.seen["task"] == [TASK]
    assert "observation.state" in pre.seen


def test_recorded_action_and_metadata_are_not_inputs():
    pre = _RecordingPre()
    ol.build_batch_frame(_item(), 0, 2, skip_key="action", to_batch_tensor=_to_batch, pre=pre)
    assert "action" not in pre.seen                     # the recorded chunk
    assert "task_index" not in pre.seen and "episode_index" not in pre.seen


def test_injected_action_is_dropped_and_tokens_survive():
    frame = ol.build_batch_frame(
        _item(), 0, 2, skip_key="action", to_batch_tensor=_to_batch, pre=_RecordingPre()
    )
    assert "action" not in frame                       # would pollute the ACTION queue
    assert "observation.language.tokens" in frame


def test_obs_window_is_indexed_by_t():
    item = _item()
    for t in (0, 1):
        frame = ol.build_batch_frame(item, t, 2, skip_key="action", to_batch_tensor=_to_batch)
        np.testing.assert_array_equal(frame["observation.state"][0], item["observation.state"][t])


def test_single_frame_policies_take_values_as_they_are():
    """GR00T: n_obs = 1 and un-windowed keys must not be sliced."""
    item = {"observation.state": np.arange(16, dtype=float), "task": TASK}
    frame = ol.build_batch_frame(item, 0, 1, skip_key="action", to_batch_tensor=_to_batch)
    assert frame["observation.state"].shape == (1, 16)
    assert frame["task"] == [TASK]


def test_cameras_are_stacked_only_when_asked():
    stacked = ol.build_batch_frame(
        _item(), 0, 2, skip_key="action", to_batch_tensor=_to_batch,
        image_keys=[CAM], obs_images_key=OBS_IMAGES,
        stack_images=lambda imgs: np.stack(imgs, axis=-4),
    )
    assert stacked[OBS_IMAGES].shape == (1, 1, 3, 8, 8)
    plain = ol.build_batch_frame(
        _item(), 0, 2, skip_key="action", to_batch_tensor=_to_batch,
        image_keys=[CAM], obs_images_key=OBS_IMAGES, stack_images=None,
    )
    assert OBS_IMAGES not in plain                     # GR00T reads per-camera keys


def test_a_dataset_without_task_strings_still_builds_frames():
    item = _item()
    del item["task"]
    pre = _RecordingPre()
    frame = ol.build_batch_frame(item, 0, 2, skip_key="action", to_batch_tensor=_to_batch, pre=pre)
    assert "task" not in pre.seen
    assert "observation.state" in frame
