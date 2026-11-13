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


# ── alignment, augmentation, and the motion-relative report ─────────────────
# The recorded action window of an n_obs_steps=2 policy starts at t-1 while its
# prediction starts at t, so comparing row k with row k measured the motion of
# one step as "error" (step 0 of a perfect policy read ~one step of motion).
# Errors are now also reported against how far the demo moved, because at
# 15 Hz one step is only a few millimetres.

from types import SimpleNamespace  # noqa: E402


def _to_mat(p9):
    """rot6d = first two ROWS of R (crisp_gym's convention), Gram-Schmidt."""
    p9 = np.asarray(p9, dtype=float)
    a, b = p9[3:6], p9[6:9]
    r0 = a / np.linalg.norm(a)
    b = b - (r0 @ b) * r0
    r1 = b / np.linalg.norm(b)
    T = np.eye(4)
    T[:3, :3] = np.stack([r0, r1, np.cross(r0, r1)])
    T[:3, 3] = p9[:3]
    return T


def _pose(x=0.0, yaw_deg=0.0, grip=1.0):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([x, 0.0, 0.0, c, -s, 0.0, s, c, 0.0, grip])


STEP = 0.005   # 5 mm per step, like a slow 15 Hz demo


def _window(first_delta, n):
    """Recorded relative targets for deltas first_delta.., constant velocity."""
    return np.stack([_pose(STEP * (d + 1)) for d in range(first_delta, first_delta + n)])


def test_current_step_offset_matches_each_policy_family():
    assert ol.current_step_offset(range(-1, 63)) == 1     # diffusion / multi_task_dit, n_obs 2
    assert ol.current_step_offset(range(-2, 14)) == 2     # n_obs 3
    assert ol.current_step_offset(range(0, 40)) == 0      # GR00T
    assert ol.current_step_offset(range(1, 10)) is None   # no current step: cannot align
    assert ol.current_step_offset(None) is None


def test_misalignment_reads_one_step_of_motion_as_error():
    """The bug this fixes, reproduced with a perfect predictor."""
    rec_window = _window(-1, 17)                          # starts at t-1, as diffusion loads it
    pred = _window(0, 8)                                  # perfect prediction from t
    unaligned = ol.chunk_metrics(pred, rec_window, _to_mat)
    assert np.allclose(unaligned["pos_err"], STEP)        # every step "wrong" by 5 mm
    offset = ol.current_step_offset(range(-1, 16))
    aligned = ol.chunk_metrics(pred, rec_window[offset:], _to_mat)
    assert np.allclose(aligned["pos_err"], 0.0)           # actually perfect


def test_standing_still_has_err_over_motion_of_one():
    rec = _window(0, 10)
    still = np.stack([_pose(0.0)] * 10)
    m = ol.chunk_metrics(still, rec, _to_mat)
    assert np.allclose(m["pos_err"], m["rec_motion"])
    assert np.allclose(m["pred_motion"], 0.0)
    assert ol.ratio_of_means(m["pos_err"], m["rec_motion"]) == pytest.approx(1.0)


def test_half_speed_prediction_is_reported_as_moving_too_little():
    rec = _window(0, 10)
    half = np.stack([_pose(0.5 * STEP * (k + 1)) for k in range(10)])
    m = ol.chunk_metrics(half, rec, _to_mat)
    assert ol.ratio_of_means(m["pred_motion"], m["rec_motion"]) == pytest.approx(0.5)
    assert ol.ratio_of_means(m["pos_err"], m["rec_motion"]) == pytest.approx(0.5)


def test_rotation_motion_and_error():
    rec = np.stack([_pose(0.0, yaw_deg=2.0 * (k + 1)) for k in range(5)])
    still = np.stack([_pose(0.0)] * 5)
    m = ol.chunk_metrics(still, rec, _to_mat)
    assert np.allclose(m["rec_rot"], [2, 4, 6, 8, 10], atol=1e-6)
    assert np.allclose(m["rot_err"], m["rec_rot"], atol=1e-6)


def test_absolute_mode_measures_motion_from_the_current_pose():
    current = _pose(1.0)                                  # robot is at x = 1 m
    rec = np.stack([_pose(1.0 + STEP * (k + 1)) for k in range(4)])
    m = ol.chunk_metrics(rec, rec, _to_mat, current_mat=_to_mat(current[:9]))
    assert np.allclose(m["pos_err"], 0.0)
    assert np.allclose(m["rec_motion"], [STEP * (k + 1) for k in range(4)])


def test_gripper_error_is_per_step():
    rec = np.stack([_pose(grip=0.2)] * 3)
    pred = np.stack([_pose(grip=0.5)] * 3)
    assert np.allclose(ol.chunk_metrics(pred, rec, _to_mat)["grip_err"], 0.3)


def test_motion_verdict_bands():
    assert ol.motion_verdict(0.0) == "GOOD"
    assert ol.motion_verdict(ol.ERR_OVER_MOTION_GOOD) == "GOOD"
    assert ol.motion_verdict(0.5) == "PARTIAL"
    assert ol.motion_verdict(0.71) == "POOR"
    assert ol.motion_verdict(float("inf")) == "UNKNOWN"   # demo did not move


def test_ratio_of_means_with_no_motion_is_inf():
    assert ol.ratio_of_means([0.001], [0.0]) == float("inf")
    assert ol.ratio_of_means([0.002, 0.004], [0.004, 0.004]) == pytest.approx(0.75)


def test_eval_augmentation_is_switched_off():
    cfg = SimpleNamespace(dataset=SimpleNamespace(image_transforms=SimpleNamespace(enable=True)))
    assert ol.disable_eval_augmentation(cfg) is True
    assert cfg.dataset.image_transforms.enable is False
    assert ol.disable_eval_augmentation(cfg) is False     # already off: nothing to do
    assert ol.disable_eval_augmentation(SimpleNamespace()) is False


def test_current_pose_prefers_cartesian_and_takes_the_latest_obs_step():
    window = np.stack([np.arange(16, dtype=float), np.arange(16, dtype=float) + 100])
    assert np.allclose(ol.current_pose9({"observation.state": window}, 2), window[1][:9])
    cart = np.arange(9, dtype=float) + 50
    item = {"observation.state": window, "observation.state.cartesian": cart}
    assert np.allclose(ol.current_pose9(item, 2), cart)
    assert ol.current_pose9({"observation.state": np.zeros(4)}, 1) is None


# ── episode selection ────────────────────────────────────────────────────────
# lerobot >= 0.4 (v3.0 datasets) has no meta.episode_data_index. The old lookup
# failed silently and evaluated the whole dataset from row 0, whatever
# --episodes said. Episodes are now located from the episode_index column.

def test_episode_rows_come_from_the_episode_column():
    column = np.repeat([0, 1, 2], [4, 3, 5])
    assert ol.episode_row_range(column, 0) == (0, 4)
    assert ol.episode_row_range(column, 1) == (4, 7)
    assert ol.episode_row_range(column, 2) == (7, 12)


def test_episode_rows_are_positions_in_a_filtered_dataset():
    """Loaded with episodes=[3, 7]: episode 7 sits at rows 2..4, not at its global index."""
    column = np.array([3, 3, 7, 7, 7])
    assert ol.episode_row_range(column, 7) == (2, 5)


def test_a_missing_episode_is_an_error_that_names_what_exists():
    with pytest.raises(ValueError, match=r"episode 5 is not in this dataset \(it has episodes 0\.\.2\)"):
        ol.episode_row_range(np.repeat([0, 1, 2], 3), 5)
    with pytest.raises(ValueError, match=r"it has 2 episodes: 3, 7"):
        ol.episode_row_range(np.array([3, 3, 7]), 0)


def test_an_episode_split_into_two_blocks_is_an_error():
    with pytest.raises(ValueError, match="not one contiguous block"):
        ol.episode_row_range(np.array([0, 0, 1, 0]), 0)


class _ArrowColumn:
    def __init__(self, values): self.values = np.asarray(values)
    def to_numpy(self): return self.values


class _HF:
    """Stands in for datasets.Dataset: an Arrow table plus an optional row selection."""
    def __init__(self, table_column, selected=None):
        self.data = types.SimpleNamespace(column=lambda name: _ArrowColumn(table_column))
        self._selected = table_column if selected is None else selected
    def with_format(self, fmt):
        assert fmt is None
        return {"episode_index": list(self._selected)}


class _DS:
    def __init__(self, hf, n): self.hf_dataset, self._n = hf, n
    def __len__(self): return self._n


def test_episode_column_is_read_from_the_arrow_table():
    column = ol.dataset_episode_column(_DS(_HF([0, 0, 1, 1, 1]), 5))
    assert column.tolist() == [0, 0, 1, 1, 1]


def test_episode_column_honours_a_row_selection():
    """The table has 5 rows but the dataset selects 3 of them: use the selection."""
    column = ol.dataset_episode_column(_DS(_HF([0, 0, 1, 1, 1], selected=[1, 1, 1]), 3))
    assert column.tolist() == [1, 1, 1]


def test_a_dataset_without_an_hf_dataset_cannot_be_searched():
    with pytest.raises(ValueError, match="no hf_dataset"):
        ol.dataset_episode_column(object())


# ── main() end to end, with numpy stand-ins for torch and lerobot ───────────
# Runs the real main(): dataset window -> alignment -> frames -> prediction ->
# per-step report -> verdict. A fake n_obs_steps=2 policy predicts the demo
# PERFECTLY, so the report must show zero error and a GOOD verdict; before the
# alignment fix it showed one step of motion as error at every row. In a
# subprocess, because it installs fake `torch` / `lerobot` modules.

DRIVER = r'''
import json, sys, types
from types import SimpleNamespace
from importlib.machinery import SourceFileLoader
import numpy as np

script, mode = sys.argv[1], sys.argv[2]
STEP, N, H, EXEC = 0.005, 60, 16, 8

def pose(x, grip=1.0):
    return np.array([x, 0, 0, 1, 0, 0, 0, 1, 0, grip], dtype=float)

class T:  # the few torch.Tensor methods the script uses
    def __init__(self, a): self.a = np.asarray(a)
    def unsqueeze(self, d): return T(np.expand_dims(self.a, d))
    def squeeze(self, d): return T(np.squeeze(self.a, d))
    def to(self, *_): return self
    def float(self): return T(self.a.astype(np.float32))
    def numpy(self): return self.a

class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False

torch = types.ModuleType("torch")
torch.device = lambda s: s
torch.cuda = SimpleNamespace(is_available=lambda: False)
torch.as_tensor = lambda a: T(a)
torch.stack = lambda xs, dim=0: T(np.stack([x.a for x in xs], axis=dim))
torch.inference_mode = lambda: _NoGrad()
sys.modules["torch"] = torch

def _mod(name):
    m = types.ModuleType(name); sys.modules[name] = m; return m
for n in ("lerobot", "lerobot.configs", "lerobot.policies", "lerobot.datasets"):
    _mod(n)

cfg = SimpleNamespace(
    dataset=SimpleNamespace(repo_id=None, root=None, image_transforms=SimpleNamespace(enable=True)),
    policy=SimpleNamespace(type="fake"))
_mod("lerobot.configs.train").TrainPipelineConfig = SimpleNamespace(from_pretrained=lambda p: cfg)

delta = list(range(-1, H - 1)) if mode == "nobs2" else list(range(0, H))

class Policy:
    def __init__(self):
        self.config = SimpleNamespace(n_obs_steps=2 if mode == "nobs2" else 1,
                                      action_delta_indices=delta, n_action_steps=EXEC,
                                      image_features=[])
    @classmethod
    def from_pretrained(cls, p): return cls()
    def to(self, d): return self
    def eval(self): return self
    def reset(self): self._queues = {"observation.state": [], "action": []}
    def predict_action_chunk(self, batch):
        t = int(round(float(batch["observation.state"].a[0, 0]) / STEP))
        # Perfect: action(j) = next pose = x(j+1); predict action(t..t+EXEC-1).
        return T(np.stack([pose(STEP * (t + k + 1)) for k in range(EXEC)])[None])

factory = _mod("lerobot.policies.factory")
factory.get_policy_class = lambda name: Policy
factory.make_pre_post_processors = lambda **k: (None, None)
def populate_queues(q, b):
    for k in q:
        if k in b: q[k].append(b[k])
    return q
_mod("lerobot.policies.utils").populate_queues = populate_queues

# Two episodes of N rows each, located ONLY through the episode_index column
# (lerobot >= 0.4 datasets have no meta.episode_data_index).
EP_COL = np.repeat([0, 1], N)
FETCHED = []

class _Column:
    def to_numpy(self): return EP_COL

class Dataset:
    hf_dataset = SimpleNamespace(data=SimpleNamespace(column=lambda name: _Column()))
    def __len__(self): return 2 * N
    def __getitem__(self, t):
        if t not in FETCHED: FETCHED.append(t)
        n_obs = 2 if mode == "nobs2" else 1
        obs = [pose(STEP * max(t - n_obs + 1 + i, 0)) for i in range(n_obs)]
        state = np.stack(obs) if n_obs > 1 else obs[0]
        act = np.stack([pose(STEP * (t + d + 1)) for d in delta])
        return {"observation.state": state, "action": act, "task": "open the electrical box"}
_mod("lerobot.datasets.factory").make_dataset = lambda c: Dataset()

lrp = types.ModuleType("lerobot_relative_pose")
lrp.POSE_ACTION_KEY = "action"
def pose9d_to_mat(p9):
    p9 = np.asarray(p9, float); a, b = p9[3:6], p9[6:9]
    r0 = a / np.linalg.norm(a); b = b - (r0 @ b) * r0; r1 = b / np.linalg.norm(b)
    M = np.eye(4); M[:3, :3] = np.stack([r0, r1, np.cross(r0, r1)]); M[:3, 3] = p9[:3]
    return M
lrp.pose9d_to_mat = pose9d_to_mat
sys.modules["lerobot_relative_pose"] = lrp

ol = SourceFileLoader("openloop_e2e", script).load_module()
episodes = sys.argv[3:] or ["0"]
sys.argv = ["x", "--path", "ckpt", "--repo-id", "ds", "--action-repr", "absolute",
            "--episodes", *episodes, "--stride", "5", "--max-frames", "6"]
try:
    ol.main()
finally:
    print("ROWS", json.dumps(FETCHED))
print("AUG_AFTER", json.dumps(cfg.dataset.image_transforms.enable))
'''


@pytest.mark.parametrize("mode", ["nobs2", "nobs1"])
def test_main_end_to_end_perfect_policy_reads_zero_error(tmp_path, mode):
    import subprocess
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    proc = subprocess.run([sys.executable, str(driver), str(SCRIPT), mode],
                          capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "AUG_AFTER false" in out                             # augmentation switched off
    assert "switched OFF for evaluation" in out
    assert ("comparing from recorded row 1" if mode == "nobs2" else "comparing from recorded row 0") in out
    assert "verdict: GOOD" in out and "err/move 0.00" in out    # a perfect policy reads perfect
    assert "predicted motion = 1.00x the demo's" in out
    assert "<- re-observe" in out                               # row for the 8th executed step
    assert "task string fed to the policy: 'open the electrical box'" in out
    assert "episode 0: dataset rows 0..59 (60 frames)" in out
    assert _rows(out) == [0, 5, 10, 15, 20, 25]                 # episode 0 only, --max-frames 6


def _run_driver(tmp_path, mode, *episodes):
    import subprocess
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    proc = subprocess.run([sys.executable, str(driver), str(SCRIPT), mode, *episodes],
                          capture_output=True, text=True, timeout=120)
    return proc, proc.stdout + proc.stderr


def _rows(out):
    import json
    line = [l for l in out.splitlines() if l.startswith("ROWS ")][-1]
    return json.loads(line[len("ROWS "):])


def test_main_evaluates_the_requested_episode_not_the_first(tmp_path):
    """--episodes 1 must read episode 1's rows (60..119), not rows from 0."""
    proc, out = _run_driver(tmp_path, "nobs2", "1")
    assert proc.returncode == 0, out[-3000:]
    assert "episode 1: dataset rows 60..119 (60 frames)" in out
    assert _rows(out) == [60, 65, 70, 75, 80, 85]
    assert "frames evaluated per episode: 1: 6" in out
    assert "verdict: GOOD" in out


def test_main_says_when_max_frames_cuts_an_episode_off(tmp_path):
    proc, out = _run_driver(tmp_path, "nobs2", "0", "1")
    assert proc.returncode == 0, out[-3000:]
    assert "frames evaluated per episode: 0: 6, 1: 0" in out
    assert "was reached before episode(s) [1]" in out
    assert _rows(out) == [0, 5, 10, 15, 20, 25]


def test_main_stops_on_an_episode_that_is_not_in_the_dataset(tmp_path):
    """No fallback to other frames: a missing episode is an error, before any inference."""
    proc, out = _run_driver(tmp_path, "nobs2", "0", "7")
    assert proc.returncode != 0
    assert "cannot select --episodes: episode 7 is not in this dataset (it has episodes 0..1)" in out
    assert _rows(out) == []                                     # nothing was evaluated
    assert "verdict" not in out
