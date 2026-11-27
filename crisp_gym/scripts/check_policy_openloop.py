"""Open-loop policy check: does the trained model reproduce the recorded actions?

STANDALONE — runs on the GPU/training PC with only lerobot + torch + numpy and
a LOCAL copy of `lerobot_relative_pose.py` sitting in the SAME folder (no
crisp_gym import, matching how the training scripts are deployed there).

It feeds the policy the EXACT training-time observations (via the same
RelativePoseDataset wrapper + the policy's delta_timestamps window) and compares
the predicted action to the RECORDED action, frame by frame. No robot.

Errors are judged RELATIVE TO MOTION, per step: err/move = (prediction error)
/ (how far the recorded demo moved from the current pose by that step).
  * 0   -> the chunk reproduces the demo.
  * 1   -> no better than predicting "stand still".
  * >1  -> worse than standing still (wrong direction).
A fixed millimetre threshold cannot work here: at 15 Hz one step moves only a
few millimetres, so a small-looking error can be most of the motion. The
predicted motion is reported next to the recorded one, which separates "too
slow / idle" (pred move << rec move) from "wrong direction or timing" (similar
size, large error).

Decision rule for real-robot drift:
  * LOW err/move  -> policy reproduces the demos. A drifting rollout is a
                     DEPLOY problem (images OOD, control rate, timing).
  * HIGH err/move -> policy did not learn the motion; retrain.
  * Rising err/move along the chunk -> execute fewer steps per chunk
                     (n_action_steps) so the robot re-observes sooner.

Alignment: the recorded action window and the predicted chunk do not always
start at the same timestep. Policies with n_obs_steps = 2 (diffusion,
multi_task_dit) load actions from t-1 but predict from t; GR00T both from t.
Each prediction is compared with the recorded step at the SAME time, located
by the position of delta 0 in the policy's action_delta_indices.

Training image augmentation is switched off for evaluation, so every frame is
judged on the images as recorded.

Camera frames: lerobot >= 0.5 builds its datasets with return_uint8=True and
converts the frames to float in [0, 1] inside its TRAINING LOOP, not in the
dataset or the preprocessor. This script feeds the policy itself, so it does
the same division by 255. (Before that was added, every image-conditioned
policy evaluated on lerobot >= 0.5 was fed frames 255 times too bright and
scored err/move ~ 1 whatever it had learned. Results from then are void.)

--diagnose explains a result instead of only reporting it:
  1) prepares each frame twice, as a training batch (collate -> lerobot's
     uint8 conversion -> preprocessor) and the way this script feeds the
     policy, and prints the largest difference. Anything but ~0 means this
     script is not showing the model what training showed it.
  2) the model's training loss on those frames, with its own observations and
     with images / state taken from another frame: does the loss depend on
     each input at all?
  3) several chunks per frame with fixed seeds: sampling noise vs systematic
     error, and how far the output moves when an input is swapped.

Policies: diffusion, and language-conditioned ones (multi_task_dit, GR00T).
Each frame carries the dataset's own task string into the preprocessor, as in
training — those policies tokenize it, and multi_task_dit fails with a shape
error without it. Observation queues and the stacked `observation.images`
tensor are used only when the policy keeps them (diffusion, multi_task_dit);
GR00T has neither and takes the single latest frame directly.

Point tracks: a checkpoint trained on a build_track_dataset.py copy predicts,
between pose and gripper, each point's (x, y, visible) in the camera image. The
pose report above reads only dims 0-8 and the gripper, so it is the same as for
a model without tracks. When the checkpoint's pose_repr.json lists the tracks
and --repo-id is that track copy, a track section follows the report: per step,
the pixel error against the tracker, the error of "the point stays where it is
now" (so err/move reads like the pose table's), and how often the predicted
visibility matches the tracker's. --track-sheets DIR also draws them: per
episode a PNG of evenly spread frames and an MP4 of all evaluated frames, each
showing the recorded and predicted path of every point over the chunk. A
checkpoint without tracks runs exactly as before; --track-sheets is then
ignored with a warning.

Usage (GPU PC, training env, script next to lerobot_relative_pose.py):
    python3 check_policy_openloop.py \
        --path .../checkpoints/100000/pretrained_model \
        --repo-id /abs/path/to/dataset \
        --episodes 0 1 2 --stride 5 --max-frames 200 [--diagnose]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

# Import the LOCAL lerobot_relative_pose.py (same directory), the exact wrapper
# used to train this checkpoint — NOT crisp_gym.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lerobot_relative_pose as lrp  # noqa: E402


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def task_for_frame(item: dict) -> list[str] | None:
    """The item's task string as a batch of one, or None when it has none.

    LeRobotDataset attaches ``item["task"]`` as a plain string (looked up from
    ``task_index``). The preprocessor moves a batch's ``task`` key into its
    complementary data, where the tokenizer of a language-conditioned policy
    reads it; a list keeps the batch dimension explicit.
    """
    task = item.get("task")
    if isinstance(task, str):
        return [task]
    if isinstance(task, (list, tuple)) and task and all(isinstance(t, str) for t in task):
        return list(task)
    return None


def policy_obs_queues(policy) -> dict | None:
    """The policy's observation queues, or None for a policy that has none.

    Diffusion and multi_task_dit build ``_queues`` in ``reset()`` and stack the
    observation window from them in ``predict_action_chunk``. GR00T keeps only
    an internal action queue and reads the batch it is given.
    """
    queues = getattr(policy, "_queues", None)
    return queues if isinstance(queues, dict) else None


def stacks_images(policy, image_features, obs_images_key: str) -> bool:
    """Whether the policy expects cameras stacked into one ``obs_images_key`` tensor.

    True exactly when it keeps a queue for that key (diffusion and
    multi_task_dit, when they have image features). Call after ``reset()``,
    which is where those queues are created.
    """
    queues = policy_obs_queues(policy)
    return bool(image_features) and queues is not None and obs_images_key in queues


def build_batch_frame(
    item: dict,
    t: int,
    n_obs: int,
    *,
    skip_key: str,
    to_batch_tensor,
    pre=None,
    action_key: str = "action",
    image_keys=(),
    obs_images_key: str = "observation.images",
    stack_images=None,
    camera_keys=(),
) -> dict:
    """One timestep (the t-th of the obs window) as a batched policy input.

    Takes every ``observation*`` key (minus ``skip_key``, the recorded action),
    adds the task string BEFORE preprocessing so a tokenizer step sees it,
    drops any ``action`` the preprocessor injects, and stacks the cameras only
    when ``stack_images`` is given. Tensor conversion is injected so this stays
    testable without torch.

    ``camera_keys`` frames that arrive as uint8 are divided by 255. lerobot >=
    0.5 builds its datasets with ``return_uint8=True`` and its training loop
    does exactly this before the preprocessor (lerobot_train.py); without it
    the policy is fed images 255 times brighter than anything it was trained
    on, and every image-conditioned policy looks as if it ignored its input.
    """
    frame = {}
    for k, v in item.items():
        if k == skip_key or not k.startswith("observation"):
            continue
        vt = v[t] if (hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == n_obs) else v
        frame[k] = to_batch_tensor(vt)
        if k in camera_keys and is_uint8(vt):
            frame[k] = frame[k] / 255.0
    task = task_for_frame(item)
    if task is not None:
        frame["task"] = task
    if pre is not None:
        frame = pre(frame)
    frame.pop(action_key, None)
    if stack_images is not None and image_keys:
        frame = dict(frame)
        frame[obs_images_key] = stack_images([frame[k] for k in image_keys])
    return frame


def is_uint8(value) -> bool:
    """True for a uint8 tensor/array (torch reports 'torch.uint8', numpy 'uint8')."""
    return str(getattr(value, "dtype", "")).endswith("uint8")


def current_step_offset(action_delta_indices) -> int | None:
    """Index of the current timestep (delta 0) in a policy's action window.

    ``predict_action_chunk`` returns actions from the current step on: diffusion
    and multi_task_dit slice their window from ``n_obs_steps - 1``, which is
    exactly where delta 0 sits in ``range(1 - n_obs_steps, ...)``; GR00T's
    window is ``range(0, ...)``. The recorded window must be cut at the same
    place, or every prediction is compared with the step before it. None when
    the window has no delta 0 (alignment unknown).
    """
    try:
        indices = [int(i) for i in action_delta_indices]
    except TypeError:
        return None
    return indices.index(0) if 0 in indices else None


def _describe_episodes(column) -> str:
    episodes = np.unique(column)
    if episodes.size == 0:
        return "no episodes"
    first, last = int(episodes[0]), int(episodes[-1])
    if episodes.size == last - first + 1:
        return f"episodes {first}..{last}"
    shown = ", ".join(str(int(e)) for e in episodes[:20])
    return f"{episodes.size} episodes: {shown}{', ...' if episodes.size > 20 else ''}"


def episode_row_range(episode_column, episode: int) -> tuple[int, int]:
    """Rows ``[lo, hi)`` of ``episode``, from the dataset's per-row episode index.

    ``episode_column[i]`` is the episode of ``dataset[i]``. Row POSITIONS are
    used, not the metadata's global frame indices: the two differ for an
    episode-filtered dataset. Raises ValueError when the episode is not in the
    dataset (naming the ones that are) or its rows are not one block. It never
    substitutes another range: evaluating different frames than the ones asked
    for is worse than stopping.
    """
    column = np.asarray(episode_column).reshape(-1)
    rows = np.flatnonzero(column == int(episode))
    if rows.size == 0:
        raise ValueError(
            f"episode {episode} is not in this dataset (it has {_describe_episodes(column)})"
        )
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    if hi - lo != rows.size:
        raise ValueError(
            f"episode {episode} is not one contiguous block of rows "
            f"({rows.size} rows spread over {lo}..{hi - 1})"
        )
    return lo, hi


def dataset_episode_column(ds) -> np.ndarray:
    """Episode index of every row of ``ds``, in row order, without decoding frames.

    Every LeRobot dataset has an ``episode_index`` column, so this works on
    v3.0 datasets (lerobot >= 0.4), where ``meta.episode_data_index`` no longer
    exists. The Arrow column is read directly when it lines up with the
    dataset's rows (as lerobot's own reader does); otherwise the HF dataset is
    asked for the column, which honours a row selection.
    """
    hf = getattr(ds, "hf_dataset", None)
    if hf is None:
        raise ValueError("the dataset exposes no hf_dataset, so its episodes cannot be located")
    try:
        column = np.asarray(hf.data.column("episode_index").to_numpy())
    except Exception:
        column = None
    if column is None or len(column) != len(ds):
        column = np.asarray(list(hf.with_format(None)["episode_index"]))
    if len(column) != len(ds):
        raise ValueError(
            f"episode_index has {len(column)} rows but the dataset has {len(ds)}; "
            "cannot locate episodes"
        )
    return column.reshape(-1)


def disable_eval_augmentation(train_config) -> bool:
    """Switch off the training run's image augmentation; True if it was on.

    The evaluation dataset is built from the checkpoint's own training config,
    so a run trained with ``--dataset.image_transforms.enable=true`` would
    otherwise feed randomly jittered frames to the policy here.
    """
    transforms = getattr(getattr(train_config, "dataset", None), "image_transforms", None)
    if transforms is not None and getattr(transforms, "enable", False):
        transforms.enable = False
        return True
    return False


def current_pose9(item: dict, n_obs: int):
    """The robot's pose at observation time as a 9-vector, from the item's state.

    Only needed for ABSOLUTE comparisons (relative items are expressed in the
    current frame, which is the identity). Uses ``observation.state.cartesian``
    when present, else the first 9 dims of ``observation.state`` (crisp_gym's
    [xyz, rot6d, gripper, ...] layout), at the latest observation step.
    """
    for key in ("observation.state.cartesian", "observation.state"):
        value = item.get(key)
        if value is None:
            continue
        arr = np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == n_obs:
            arr = arr[-1]
        if arr.ndim == 1 and arr.shape[0] >= 9:
            return arr[:9]
    return None


def chunk_metrics(pred, rec, to_mat, current_mat=None) -> dict[str, list[float]]:
    """Per-step comparison of one predicted chunk with the recorded one.

    ``pred`` and ``rec`` are rows of ``[x, y, z, rot6d(6), ..., gripper]``,
    already ALIGNED (row k of each is the same timestep). ``current_mat`` is
    the 4x4 pose at observation time; None means the poses are relative to it
    (the identity), as RelativePoseDataset produces.

    Returns per-step lists: pos_err / rot_err / grip_err (prediction vs
    recording), rec_motion / pred_motion (distance from the current pose: the
    error a "stand still" prediction would have, and how far the prediction
    moves), rec_rot (recorded rotation from the current pose).
    """
    current = np.eye(4) if current_mat is None else np.asarray(current_mat, dtype=float)
    out = {k: [] for k in ("pos_err", "rot_err", "grip_err", "rec_motion", "pred_motion", "rec_rot")}
    for k in range(min(len(pred), len(rec))):
        Tp = to_mat(np.asarray(pred[k][:9], dtype=float))
        Tr = to_mat(np.asarray(rec[k][:9], dtype=float))
        out["pos_err"].append(float(np.linalg.norm(Tp[:3, 3] - Tr[:3, 3])))
        out["rot_err"].append(_geodesic_deg(Tp[:3, :3], Tr[:3, :3]))
        out["grip_err"].append(abs(float(pred[k][-1]) - float(rec[k][-1])))
        out["rec_motion"].append(float(np.linalg.norm(Tr[:3, 3] - current[:3, 3])))
        out["pred_motion"].append(float(np.linalg.norm(Tp[:3, 3] - current[:3, 3])))
        out["rec_rot"].append(_geodesic_deg(current[:3, :3], Tr[:3, :3]))
    return out


def sampling_diagnostics(samples, rec, swapped_image=None, swapped_state=None) -> dict[str, list[float]]:
    """Per-step position statistics (metres) of K sampled chunks of ONE frame.

    ``samples`` is ``[K, H, D]``: K chunks sampled for the same observation with
    different noise. ``rec`` is the recorded chunk, aligned with them.
    ``swapped_image`` / ``swapped_state`` are chunks sampled with the SAME noise
    as ``samples[0]`` but with that input taken from another frame, so their
    distance to ``samples[0]`` is the effect of the input alone.

      single_err : mean over samples of |sample - recording|
      mean_err   : |mean of the samples - recording|  (what is left without noise)
      spread     : mean over samples of |sample - mean of the samples|  (the noise)
      image_swap / state_swap : |swapped - samples[0]|  (how much that input matters)
    """
    samples = np.asarray(samples, dtype=float)
    rec = np.asarray(rec, dtype=float)
    steps = min(samples.shape[1], len(rec))
    pos = samples[:, :steps, :3]
    target = rec[:steps, :3]
    centre = pos.mean(axis=0)
    out = {
        "single_err": np.linalg.norm(pos - target[None], axis=-1).mean(axis=0).tolist(),
        "mean_err": np.linalg.norm(centre - target, axis=-1).tolist(),
        "spread": np.linalg.norm(pos - centre[None], axis=-1).mean(axis=0).tolist(),
    }
    for name, swapped in (("image_swap", swapped_image), ("state_swap", swapped_state)):
        if swapped is not None:
            other = np.asarray(swapped, dtype=float)[:steps, :3]
            n = min(len(other), steps)
            out[name] = np.linalg.norm(other[:n] - pos[0, :n], axis=-1).tolist()
    return out


def spread_evenly(values, count: int) -> list:
    """Up to ``count`` of ``values``, evenly spaced over the whole list, in order."""
    values = list(values)
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return values
    picks = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[i] for i in dict.fromkeys(picks.tolist())]


# --diagnose: frames per training-style batch, and noise draws averaged per loss.
DIAGNOSE_BATCH = 8
DIAGNOSE_LOSS_REPEATS = 4


# err/move bands for the verdict. Rules of thumb, not calibrated thresholds:
# the per-step table is the result; the verdict only names its region.
ERR_OVER_MOTION_GOOD = 0.3
ERR_OVER_MOTION_POOR = 0.7


def motion_verdict(err_over_motion: float) -> str:
    """Name the err/move region: GOOD, PARTIAL, POOR, or UNKNOWN (no motion)."""
    if not np.isfinite(err_over_motion):
        return "UNKNOWN"
    if err_over_motion <= ERR_OVER_MOTION_GOOD:
        return "GOOD"
    if err_over_motion <= ERR_OVER_MOTION_POOR:
        return "PARTIAL"
    return "POOR"


def ratio_of_means(numerators, denominators) -> float:
    """mean(num) / mean(den); inf when the denominator mean is ~0."""
    den = float(np.mean(denominators)) if len(denominators) else 0.0
    if den <= 1e-9:
        return float("inf")
    return float(np.mean(numerators)) / den


def _find_action_repr(pretrained_path):
    """Load action_repr.json stamped by the absolute training launchers, if any.

    Walks up from the checkpoint dir (usually <out>/checkpoints/<step>/
    pretrained_model) to the training output root, like the deploy policy's
    find_action_repr. Returns the parsed dict or None.
    """
    import json
    from pathlib import Path

    p = Path(pretrained_path).resolve()
    for cand in (p, *list(p.parents)[:4]):
        f = cand / "action_repr.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
    return None


# ── point tracks (checkpoints trained on a build_track_dataset.py copy) ──────
# Nothing below runs for a checkpoint without track targets.

# A predicted visibility at or above this counts as "visible". The recorded flag
# is 0 or 1, so the midpoint is the natural cut.
TRACK_VISIBLE_THRESHOLD = 0.5
_TRACK_LAYOUT_KEYS = ("points", "action_names", "position_dims", "visibility_dims", "image_size")


def find_track_layout(pretrained_path) -> dict | None:
    """The checkpoint's point-track layout, or None for a model without tracks.

    lerobot_relative_pose.py records it as ``track_targets`` in pose_repr.json
    at the training output root, found by walking up from the checkpoint like
    ``_find_action_repr``. None when that file is missing, has no
    ``track_targets``, or the record lacks a field this script needs.
    """
    import json
    from pathlib import Path

    p = Path(pretrained_path).resolve()
    for cand in (p, *list(p.parents)[:4]):
        f = cand / "pose_repr.json"
        if not f.exists():
            continue
        try:
            record = json.loads(f.read_text()).get("track_targets")
        except Exception:
            return None
        if not isinstance(record, dict) or any(k not in record for k in _TRACK_LAYOUT_KEYS):
            return None
        record = dict(record)
        record.setdefault("row_shift", 1)
        return record
    return None


def dataset_column(ds, name: str) -> np.ndarray:
    """Column ``name`` of every row of ``ds`` as floats, in row order, without decoding frames.

    Same lookup as ``dataset_episode_column``: the Arrow column when it lines
    up with the dataset's rows, else the HF dataset's own column.
    """
    hf = getattr(ds, "hf_dataset", None)
    if hf is None:
        raise ValueError(f"the dataset exposes no hf_dataset, so its {name!r} column cannot be read")
    try:
        column = hf.data.column(name).to_pylist()
    except Exception:
        column = None
    if column is None or len(column) != len(ds):
        column = list(hf.with_format(None)[name])
    if len(column) != len(ds):
        raise ValueError(f"{name} has {len(column)} rows but the dataset has {len(ds)}")
    return np.asarray(column, dtype=float)


def track_pixels(rows, layout) -> tuple[np.ndarray, np.ndarray]:
    """Action rows -> point positions [..., N, 2] in camera pixels and recorded visibility [..., N].

    Track dims hold x / (W - 1) and y / (H - 1) of the camera frame
    (``layout["image_size"]`` is [H, W]) and the visibility.
    """
    rows = np.asarray(rows, dtype=float)
    H, W = (int(v) for v in layout["image_size"])
    xy = np.stack(
        [rows[..., layout["position_dims"][p]] for p in layout["points"]], axis=-2
    ) * np.array([W - 1, H - 1], dtype=float)
    vis = np.stack([rows[..., layout["visibility_dims"][p]] for p in layout["points"]], axis=-1)
    return xy, vis


def track_now(episode_actions, frame: int, layout):
    """Where each point is in the image at ``frame`` (episode-relative), or (None, None).

    build_track_dataset.py stores the point positions of image ``t + row_shift``
    in action row ``t``, so the current image's positions sit ``row_shift``
    rows back. The first ``row_shift`` images of an episode are in no row.
    Returns positions [N, 2] in pixels and visible [N] (bool).
    """
    row = int(frame) - int(layout.get("row_shift", 1))
    if row < 0 or row >= len(episode_actions):
        return None, None
    xy, vis = track_pixels(episode_actions[row], layout)
    return xy, vis >= TRACK_VISIBLE_THRESHOLD


def track_chunk_metrics(pred, rec, now_xy, layout) -> dict[str, np.ndarray]:
    """Per point and step ([N, K] arrays) comparison of one chunk's tracks with the recording.

    ``pred`` and ``rec`` are ALIGNED action chunks (row k of each is the same
    timestep). ``now_xy`` is where the points are in the current image (None
    when unknown).

      err    : |predicted - recorded| position, pixels
      move   : |recorded - now| = the error of predicting that the point stays put
      agree  : 1 when the predicted visibility (>= 0.5) matches the recorded flag
      seen   : 1 when the tracker saw the point (the recorded flag)

    err and move are NaN where the tracker did not see the point at that step
    or the current position is unknown, so both are averaged over the same
    entries; agree and seen are counted at every step.
    """
    K = min(len(pred), len(rec))
    p_xy, p_vis = track_pixels(np.asarray(pred, dtype=float)[:K], layout)  # [K, N, 2], [K, N]
    r_xy, r_vis = track_pixels(np.asarray(rec, dtype=float)[:K], layout)
    seen = r_vis >= TRACK_VISIBLE_THRESHOLD
    counted = seen & (now_xy is not None)
    err = np.linalg.norm(p_xy - r_xy, axis=-1)
    move = (
        np.linalg.norm(r_xy - np.asarray(now_xy, dtype=float)[None], axis=-1)
        if now_xy is not None
        else np.full_like(err, np.nan)
    )
    return {
        "err": np.where(counted, err, np.nan).T,
        "move": np.where(counted, move, np.nan).T,
        "agree": ((p_vis >= TRACK_VISIBLE_THRESHOLD) == seen).astype(float).T,
        "seen": seen.astype(float).T,
    }


def stack_track_metrics(frames: list[dict]) -> dict[str, np.ndarray]:
    """Per-frame track metrics -> [F, N, K] arrays, NaN-padded to the longest chunk."""
    K = max(f["err"].shape[1] for f in frames)
    out = {}
    for name in ("err", "move", "agree", "seen"):
        out[name] = np.stack([
            np.pad(f[name], ((0, 0), (0, K - f[name].shape[1])), constant_values=np.nan)
            for f in frames
        ])
    return out


def track_step_summary(stacked: dict, step: int, point: int | None = None) -> dict[str, float]:
    """Means at one step, over all frames and either every point or one."""
    sel = (slice(None), slice(None) if point is None else [point], step)
    err, move = stacked["err"][sel].ravel(), stacked["move"][sel].ravel()
    ok = np.isfinite(err) & np.isfinite(move)

    def _mean(values):
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    return {
        "err": _mean(err[ok]),
        "move": _mean(move[ok]),
        "ratio": ratio_of_means(err[ok], move[ok]) if ok.any() else float("nan"),
        "agree": _mean(stacked["agree"][sel].ravel()),
        "seen": _mean(stacked["seen"][sel].ravel()),
        "counted": int(ok.sum()),
    }


def _fmt(value: float, spec: str, width: int) -> str:
    return f"{value:>{width}{spec}}" if np.isfinite(value) else f"{'-':>{width}}"


def log_track_report(stacked: dict, layout: dict, exec_steps: int) -> None:
    """Print the track section of the report (see the module docstring)."""
    points = list(layout["points"])
    H, W = (int(v) for v in layout["image_size"])
    K = stacked["err"].shape[2]
    logger.info(
        f"\n[openloop] point tracks: {len(points)} points ({', '.join(points)}), "
        f"in pixels of the {W}x{H} camera frame"
    )
    logger.info("  err  = predicted vs recorded (tracker) position. move = how far the recorded")
    logger.info("  point moved from where it is in the current image = the error of predicting")
    logger.info("  that it stays put; err/move reads like the pose table's. Both are counted")
    logger.info("  where the tracker saw the point at that step. vis agree = the predicted")
    logger.info(f"  visibility (>= {TRACK_VISIBLE_THRESHOLD}) matches the tracker's; seen = how often the tracker")
    logger.info("  saw the point, which is what always answering 'visible' would agree.")
    header = (
        f"    {'step':>4} | {'err':>8} | {'move':>8} | {'err/move':>8} | {'vis agree':>9} | {'seen':>6}"
    )
    logger.info(header)
    logger.info(f"    {'':>4} | {'(px)':>8} | {'(px)':>8} | {'':>8} | {'':>9} | {'':>6}")
    for k in range(K):
        s = track_step_summary(stacked, k)
        mark = "  <- re-observe" if k == exec_steps - 1 else ""
        logger.info(
            f"    {k:>4} | {_fmt(s['err'], '.2f', 8)} | {_fmt(s['move'], '.2f', 8)} | "
            f"{_fmt(s['ratio'], '.2f', 8)} | {_fmt(100 * s['agree'], '.1f', 8)}% | "
            f"{_fmt(100 * s['seen'], '.1f', 5)}%{mark}"
        )
    k = min(exec_steps, K) - 1
    width = max(8, *(len(p) for p in points))
    logger.info(f"\n  per point at step {k} (re-observe after {k + 1} executed steps):")
    logger.info(
        f"    {'point':<{width}} | {'err':>8} | {'move':>8} | {'err/move':>8} | "
        f"{'vis agree':>9} | {'seen':>6} | {'counted':>7}"
    )
    for i, name in enumerate(points):
        s = track_step_summary(stacked, k, point=i)
        logger.info(
            f"    {name:<{width}} | {_fmt(s['err'], '.2f', 8)} | {_fmt(s['move'], '.2f', 8)} | "
            f"{_fmt(s['ratio'], '.2f', 8)} | {_fmt(100 * s['agree'], '.1f', 8)}% | "
            f"{_fmt(100 * s['seen'], '.1f', 5)}% | {s['counted']:>7}"
        )
    shift = int(layout.get("row_shift", 1))
    logger.info(
        "  counted = (frame, step) pairs in err and move. The first "
        f"{'frame' if shift == 1 else f'{shift} frames'} of each episode\n"
        "  has no recorded position for the current image, so it is left out of them."
    )


# ── --track-sheets: drawing ───────────────────────────────────────────────────
# Colours are BGR (OpenCV). One per point, then repeating.
TRACK_COLOURS = [
    (40, 160, 255), (255, 170, 40), (90, 210, 90), (230, 90, 230),
    (40, 230, 240), (240, 240, 60), (90, 90, 255), (200, 200, 200),
]
TRACK_HIDDEN_COLOUR = (150, 150, 150)
TRACK_TILE_MIN_WIDTH = 480


def frame_rgb_uint8(value) -> np.ndarray:
    """A dataset camera frame ([n_obs,] C, H, W or H, W, C; uint8 or float in [0, 1]) as HWC uint8.

    With an observation window the latest frame (the current one) is used.
    """
    arr = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"expected an image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if not is_uint8(arr):
        arr = arr.astype(np.float64)
        if arr.size and arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


def _dashed(cv2, img, a, b, colour, thickness, dash=7, gap=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return
    step = (b - a) / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        p, q = a + step * t, a + step * e
        cv2.line(img, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))),
                 colour, thickness, cv2.LINE_AA)
        t = e + gap


def draw_track_tile(image_rgb, pred, rec, now_xy, now_vis, layout, exec_steps: int, title: str):
    """One frame with each point's recorded and predicted path over the chunk (BGR uint8).

      solid line, dots    recorded path (the tracker), where it saw the point
      dashed, circles     predicted path (lighter tint, a hollow circle per step),
                          where the model says the point is visible
      grey x              a step the model predicts the point hidden
      ring                where the point is in this image (when seen)
      squares             the re-observe step (filled: recorded, hollow: predicted)

    Positions are fractions of the camera frame, so they are drawn at the same
    place whatever size the image is shown at.
    """
    import cv2

    img = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    h0, w0 = img.shape[:2]
    if w0 < TRACK_TILE_MIN_WIDTH:
        s = TRACK_TILE_MIN_WIDTH / w0
        img = cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    H, W = (int(v) for v in layout["image_size"])
    to_tile = np.array([(w - 1) / max(W - 1, 1), (h - 1) / max(H - 1, 1)])

    def pt(xy):
        q = np.asarray(xy, dtype=float) * to_tile
        return int(round(q[0])), int(round(q[1]))

    K = min(len(pred), len(rec))
    p_xy, p_vis = track_pixels(np.asarray(pred, dtype=float)[:K], layout)
    r_xy, r_vis = track_pixels(np.asarray(rec, dtype=float)[:K], layout)
    reobs = min(exec_steps, K) - 1
    for i, name in enumerate(layout["points"]):
        colour = TRACK_COLOURS[i % len(TRACK_COLOURS)]
        seen = r_vis[:, i] >= TRACK_VISIBLE_THRESHOLD
        shown = p_vis[:, i] >= TRACK_VISIBLE_THRESHOLD
        # recorded: solid, broken where the tracker lost the point; dark under-stroke for contrast
        for k in range(1, K):
            if seen[k - 1] and seen[k]:
                cv2.line(img, pt(r_xy[k - 1, i]), pt(r_xy[k, i]), (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(img, pt(r_xy[k - 1, i]), pt(r_xy[k, i]), colour, 2, cv2.LINE_AA)
        for k in np.flatnonzero(seen):
            cv2.circle(img, pt(r_xy[k, i]), 2, colour, -1, cv2.LINE_AA)
        # predicted: a lighter dashed line with a hollow circle per step where
        # predicted visible (still readable on top of the recorded line), a grey
        # x where predicted hidden
        light = tuple(int(c + (255 - c) * 0.45) for c in colour)
        for k in range(1, K):
            if shown[k - 1] and shown[k]:
                _dashed(cv2, img, pt(p_xy[k - 1, i]), pt(p_xy[k, i]), light, 1, dash=5, gap=4)
        for k in np.flatnonzero(shown):
            # outline outside the ring, so a recorded dot under it stays visible
            cv2.circle(img, pt(p_xy[k, i]), 5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img, pt(p_xy[k, i]), 4, light, 1, cv2.LINE_AA)
        for k in np.flatnonzero(~shown):
            x, y = pt(p_xy[k, i])
            cv2.line(img, (x - 3, y - 3), (x + 3, y + 3), TRACK_HIDDEN_COLOUR, 1, cv2.LINE_AA)
            cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), TRACK_HIDDEN_COLOUR, 1, cv2.LINE_AA)
        if 0 <= reobs < K:
            if seen[reobs]:
                x, y = pt(r_xy[reobs, i])
                cv2.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), colour, -1)
                cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 0), 1)
            x, y = pt(p_xy[reobs, i])
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5),
                          colour if shown[reobs] else TRACK_HIDDEN_COLOUR, 2)
        if now_xy is not None and now_vis is not None and now_vis[i]:
            cv2.circle(img, pt(now_xy[i]), 7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(img, pt(now_xy[i]), 7, colour, 2, cv2.LINE_AA)

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    bar = 20
    canvas = np.full((h + 3 * bar, w, 3), 32, np.uint8)
    canvas[bar:bar + h] = img
    cv2.putText(canvas, title, (6, 14), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    # legend line 1: the marks
    y = bar + h + 14
    x = 6
    cv2.line(canvas, (x, y - 4), (x + 18, y - 4), (230, 230, 230), 2, cv2.LINE_AA)
    x += 22
    cv2.putText(canvas, "recorded", (x, y), font, fs, (230, 230, 230), 1, cv2.LINE_AA)
    x += 74
    _dashed(cv2, canvas, (x, y - 4), (x + 18, y - 4), (230, 230, 230), 1, dash=4, gap=3)
    cv2.circle(canvas, (x + 9, y - 4), 4, (230, 230, 230), 1, cv2.LINE_AA)
    x += 22
    cv2.putText(canvas, "predicted", (x, y), font, fs, (230, 230, 230), 1, cv2.LINE_AA)
    x += 80
    cv2.circle(canvas, (x + 6, y - 4), 6, (230, 230, 230), 2, cv2.LINE_AA)
    x += 16
    cv2.putText(canvas, "now", (x, y), font, fs, (230, 230, 230), 1, cv2.LINE_AA)
    x += 36
    cv2.rectangle(canvas, (x, y - 9), (x + 9, y), (230, 230, 230), 2)
    x += 14
    cv2.putText(canvas, f"step {reobs}", (x, y), font, fs, (230, 230, 230), 1, cv2.LINE_AA)
    x += 62
    cv2.line(canvas, (x, y - 8), (x + 7, y - 1), TRACK_HIDDEN_COLOUR, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - 1), (x + 7, y - 8), TRACK_HIDDEN_COLOUR, 1, cv2.LINE_AA)
    x += 11
    cv2.putText(canvas, "pred. hidden", (x, y), font, fs, (230, 230, 230), 1, cv2.LINE_AA)
    # legend line 2: the points
    y += bar
    x = 6
    for i, name in enumerate(layout["points"]):
        colour = TRACK_COLOURS[i % len(TRACK_COLOURS)]
        cv2.putText(canvas, name, (x, y), font, fs, colour, 1, cv2.LINE_AA)
        x += cv2.getTextSize(name, font, fs, 1)[0][0] + 14
    return canvas


def track_sheet(tiles: list[np.ndarray], columns: int = 4) -> np.ndarray:
    """Tiles of one episode in a grid (all resized to the first tile's size)."""
    import cv2

    h, w = tiles[0].shape[:2]
    tiles = [t if t.shape[:2] == (h, w) else cv2.resize(t, (w, h)) for t in tiles]
    cols = max(1, min(columns, len(tiles)))
    rows = -(-len(tiles) // cols)
    sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for n, tile in enumerate(tiles):
        r, c = divmod(n, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return sheet


def write_track_sheets(out_dir, frames: list[dict], layout: dict, exec_steps: int,
                       per_sheet: int, video_fps: float) -> list[str]:
    """Draw every stored frame; one PNG sheet and one MP4 per episode. Returns the files written.

    ``frames`` entries: episode, frame, row, image (PNG bytes, BGR), pred, rec,
    now_xy, now_vis.
    """
    import cv2
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    episodes = list(dict.fromkeys(f["episode"] for f in frames))
    for ep in episodes:
        tiles = []
        for f in (f for f in frames if f["episode"] == ep):
            bgr = cv2.imdecode(np.frombuffer(f["image"], np.uint8), cv2.IMREAD_COLOR)
            tiles.append(draw_track_tile(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f["pred"], f["rec"], f["now_xy"],
                f["now_vis"], layout, exec_steps,
                f"episode {ep}  frame {f['frame']}  (row {f['row']})",
            ))
        png = out / f"episode_{int(ep):03d}_tracks.png"
        cv2.imwrite(str(png), track_sheet(spread_evenly(tiles, per_sheet)))
        written.append(str(png))
        mp4 = out / f"episode_{int(ep):03d}_tracks.mp4"
        h, w = tiles[0].shape[:2]
        writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(video_fps), (w, h))
        if not writer.isOpened():
            logger.warning(f"[openloop] could not open a video writer for {mp4}; PNG sheets only")
            continue
        for tile in tiles:
            writer.write(tile if tile.shape[:2] == (h, w) else cv2.resize(tile, (w, h)))
        writer.release()
        written.append(str(mp4))
    return written


def dataset_track_stamp(ds) -> dict | None:
    """The evaluated dataset's meta/track_targets.json (build_track_dataset.py), or None."""
    import json
    from pathlib import Path

    root = getattr(ds, "root", None)
    f = Path(root) / "meta" / "track_targets.json" if root is not None else None
    if f is None or not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def track_layout_problem(layout: dict, model_dim, dataset_dim, dataset_stamp) -> str | None:
    """Why the track report cannot run on this checkpoint + dataset, or None when it can."""
    n = len(layout["action_names"])
    if model_dim is not None and int(model_dim) != n:
        return (f"the checkpoint predicts {model_dim} action dims but its pose_repr.json "
                f"describes {n}")
    if dataset_dim is not None and int(dataset_dim) != n:
        return (f"the dataset's action has {dataset_dim} dims, the checkpoint's {n}: point "
                "--repo-id at the track copy (build_track_dataset.py --out) to compare the tracks")
    if dataset_stamp is None:
        return "the dataset has no meta/track_targets.json, so it is not a track copy"
    if list(dataset_stamp.get("action_names", [])) != list(layout["action_names"]):
        return "the dataset's track layout (meta/track_targets.json) differs from the checkpoint's"
    return None


def main():
    p = argparse.ArgumentParser(description="Open-loop action-prediction error vs the demos")
    p.add_argument("--path", required=True, help="checkpoint pretrained_model dir")
    p.add_argument("--repo-id", required=True, help="dataset repo id or local path")
    p.add_argument("--root", default=None, help="dataset root (if repo-id is a plain name)")
    p.add_argument("--episodes", type=int, nargs="*", default=[0])
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument(
        "--exec-steps", type=int, default=None,
        help="Steps the robot executes per chunk before re-observing (your "
        "deploy n_action_steps). The verdict reads err/move at this step. "
        "Default: the checkpoint's own n_action_steps.",
    )
    p.add_argument(
        "--action-repr", choices=["auto", "relative", "absolute"], default="auto",
        help="How the checkpoint's action is expressed. 'auto' reads "
        "action_repr.json next to --path (absolute marker => absolute, else "
        "relative). RELATIVE wraps the dataset with RelativePoseDataset (obs + "
        "action re-expressed vs the current frame); ABSOLUTE compares the raw "
        "on-disk poses directly — use it for train_absolute_next_pose.py "
        "checkpoints.",
    )
    p.add_argument(
        "--diagnose", action="store_true",
        help="After the report, find out WHY the numbers are what they are: (1) "
        "whether this script feeds the model exactly what a training batch does, "
        "(2) the model's training loss on these frames, with its own observations "
        "and with images / state taken from another frame, (3) several samples "
        "per frame: sampling noise vs systematic error, and how much the output "
        "changes when an input is swapped.",
    )
    p.add_argument("--diagnose-frames", type=int, default=32,
                   help="frames used by --diagnose, spread over the evaluated ones")
    p.add_argument("--diagnose-samples", type=int, default=4,
                   help="chunks sampled per frame by --diagnose")
    p.add_argument(
        "--track-sheets", default=None, metavar="DIR",
        help="Checkpoints trained on a point-track copy only: draw each evaluated "
        "frame with every point's recorded and predicted path over the chunk. "
        "Writes one PNG sheet and one MP4 per episode into DIR. Needs opencv.",
    )
    p.add_argument("--track-sheet-frames", type=int, default=8,
                   help="frames per episode on a --track-sheets PNG, spread over the "
                   "evaluated ones (the MP4 has all of them)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.utils import populate_queues

    try:
        from lerobot.constants import ACTION, OBS_IMAGES
    except ImportError:
        ACTION, OBS_IMAGES = "action", "observation.images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[openloop] loading {args.path} on {device}")

    train_config = TrainPipelineConfig.from_pretrained(args.path)
    # Point the training config at the dataset the user gave (keeps the policy's
    # delta_timestamps so make_dataset builds the correct n_obs_steps window).
    try:
        train_config.dataset.repo_id = args.repo_id
        if args.root:
            train_config.dataset.root = args.root
        elif os.path.isdir(args.repo_id):
            train_config.dataset.root = args.repo_id
    except Exception as e:
        logger.warning(f"[openloop] could not override dataset path on cfg: {e}")

    policy_cls = get_policy_class(train_config.policy.type)
    policy = policy_cls.from_pretrained(args.path)
    if args.num_inference_steps is not None and hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = args.num_inference_steps
    policy.to(device).eval()

    pre = post = None
    try:
        from lerobot.policies.factory import make_pre_post_processors
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.path)
    except ImportError:
        pass

    # Build the dataset with the policy's temporal window (delta_timestamps),
    # then wrap with the training relative-pose conversion. Augmentation off:
    # the dataset comes from the TRAINING config, which may have it enabled.
    if disable_eval_augmentation(train_config):
        logger.info("[openloop] the training run used image augmentation; switched OFF for evaluation")
    from lerobot.datasets.factory import make_dataset
    base_ds = make_dataset(train_config)
    ds = base_ds if not hasattr(base_ds, "datasets") else base_ds  # single dataset expected

    # RELATIVE (lerobot_relative_pose.py checkpoints): wrap so obs + action are
    # re-expressed vs the current frame, exactly as at training. ABSOLUTE
    # (train_absolute_next_pose.py): the model saw/predicts raw absolute poses,
    # so DON'T wrap — comparing wrapped (relative) targets to an absolute model
    # would be meaningless. Auto-detect from action_repr.json next to --path.
    is_absolute = args.action_repr == "absolute"
    if args.action_repr == "auto":
        ar = _find_action_repr(args.path)
        is_absolute = bool(ar and ar.get("action", {}).get("pose_repr") == "absolute")
    logger.info(
        "[openloop] action_repr=%s -> %s comparison",
        args.action_repr, "ABSOLUTE (no relative wrap)" if is_absolute else "RELATIVE",
    )
    wrapped = ds if is_absolute else lrp.RelativePoseDataset(ds)

    image_features = list(getattr(policy.config, "image_features", []) or [])
    n_obs = int(getattr(policy.config, "n_obs_steps", 1))
    # Cameras as the dataset names them (what lerobot's training loop iterates
    # over when it rescales uint8 frames), plus the ones the policy consumes.
    camera_keys = list(dict.fromkeys(
        [*(getattr(getattr(ds, "meta", None), "camera_keys", None) or []), *image_features]
    ))

    # Point tracks: only for a checkpoint whose pose_repr.json lists them, and
    # only when the dataset is the track copy it was trained on. Everything
    # track-related stays None otherwise, and the run is exactly as before.
    track_layout = find_track_layout(args.path)
    track_actions = None  # the dataset's action column: the points' current positions
    track_camera = None
    if track_layout is not None:
        out_feat = (getattr(policy.config, "output_features", None) or {}).get(ACTION)
        model_dim = (getattr(out_feat, "shape", None) or [None])[0] if out_feat is not None else None
        ds_feat = (getattr(getattr(ds, "meta", None), "features", None) or {}).get(ACTION) or {}
        dataset_dim = (ds_feat.get("shape") or [None])[0] if isinstance(ds_feat, dict) else None
        problem = track_layout_problem(track_layout, model_dim, dataset_dim, dataset_track_stamp(ds))
        if problem is None:
            try:
                track_actions = dataset_column(ds, ACTION)
            except ValueError as e:
                problem = f"cannot read the dataset's actions ({e})"
        if problem is not None:
            logger.warning(f"[openloop] point tracks: no track report, {problem}")
        else:
            logger.info(
                f"[openloop] point tracks: {len(track_layout['points'])} points "
                f"({', '.join(track_layout['points'])}); a track section follows the pose report"
            )
    if args.track_sheets:
        if track_actions is None:
            logger.warning(
                "[openloop] --track-sheets ignored: "
                + ("no point tracks to draw (see above)" if track_layout is not None else
                   "this checkpoint was not trained with point tracks (its pose_repr.json lists none)")
            )
        else:
            try:
                import cv2  # noqa: F401
            except ImportError as e:
                raise SystemExit("[openloop] --track-sheets needs opencv (import cv2 failed)") from e
            wanted = track_layout.get("camera")
            track_camera = wanted if wanted in camera_keys else (camera_keys[0] if camera_keys else None)
            if track_camera is None:
                logger.warning("[openloop] --track-sheets ignored: the dataset has no camera")
            elif wanted and wanted != track_camera:
                logger.warning(f"[openloop] the tracks were clicked on {wanted}, which this dataset "
                               f"does not have; drawing on {track_camera}")
    track_frames: list[dict] = []
    sheet_frames: list[dict] = []

    # Align the recorded window with the prediction (see the module docstring).
    delta_indices = getattr(policy.config, "action_delta_indices", None)
    offset = current_step_offset(delta_indices)
    if offset is None:
        offset = 0
        logger.warning(
            "[openloop] the policy's action window has no current step (delta 0): "
            "cannot align the recording with the prediction; comparing from row 0"
        )
    else:
        logger.info(
            "[openloop] recorded actions start at delta %+d; predictions start at the "
            "current step -> comparing from recorded row %d",
            int(list(delta_indices)[0]), offset,
        )

    def _to_tensor(vt):
        vt = torch.as_tensor(np.asarray(vt.numpy() if hasattr(vt, "numpy") else vt))
        return vt.unsqueeze(0).to(device).float()

    def _stack_cameras(images):
        return torch.stack(images, dim=-4)

    task_logged = False

    def _predict(idx: int) -> np.ndarray:
        return _predict_item(wrapped[idx])

    def _predict_item(item: dict, seed: int | None = None) -> np.ndarray:
        """One chunk for this item's observations. ``seed`` fixes the sampling noise."""
        nonlocal task_logged
        policy.reset()
        if pre is not None:
            pre.reset()
        if not task_logged:
            task = task_for_frame(item)
            if task is not None:
                logger.info(f"[openloop] task string fed to the policy: {task[0]!r}")
            else:
                logger.warning(
                    "[openloop] dataset items carry no task string; a language-"
                    "conditioned policy (multi_task_dit, GR00T) will fail or be "
                    "evaluated without its instruction"
                )
            scaled = [k for k in camera_keys if k in item and is_uint8(item[k])]
            if scaled:
                logger.info(
                    f"[openloop] camera frames arrive as uint8 ({', '.join(scaled)}): "
                    "divided by 255, as lerobot's training loop does"
                )
            task_logged = True
        queues = policy_obs_queues(policy)  # after reset(): that is where they are built
        stack = _stack_cameras if stacks_images(policy, image_features, OBS_IMAGES) else None
        last = None
        for t in range(n_obs):
            last = build_batch_frame(
                item, t, n_obs,
                skip_key=lrp.POSE_ACTION_KEY,
                to_batch_tensor=_to_tensor,
                pre=pre,
                action_key=ACTION,
                image_keys=image_features,
                obs_images_key=OBS_IMAGES,
                stack_images=stack,
                camera_keys=camera_keys,
            )
            if queues is not None:
                policy._queues = populate_queues(policy._queues, last)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.inference_mode():
            act = policy.predict_action_chunk(last)
            if post is not None:
                act = post(act)
        return act.squeeze(0).to("cpu").numpy()

    # Rows of each requested episode, resolved for ALL of them before any
    # inference. An episode that cannot be located stops the run: silently
    # evaluating other frames (the old fallback was the whole dataset from row
    # 0) reports numbers for episodes nobody asked about.
    try:
        episode_column = dataset_episode_column(ds)
        episode_ranges = [(ep, *episode_row_range(episode_column, ep)) for ep in args.episodes]
    except ValueError as e:
        raise SystemExit(f"[openloop] cannot select --episodes: {e}") from e
    for ep, lo, hi in episode_ranges:
        logger.info(f"[openloop] episode {ep}: dataset rows {lo}..{hi - 1} ({hi - lo} frames)")

    rows: dict[str, list[list[float]]] = {}
    baseline_ok = True
    n_done = 0
    frames_per_episode: dict[int, int] = {}
    evaluated_rows: list[int] = []
    for ep, lo, hi in episode_ranges:
        frames_per_episode.setdefault(ep, 0)
        for idx in range(lo, hi, args.stride):
            if n_done >= args.max_frames:
                break
            item = wrapped[idx]
            if lrp.POSE_ACTION_KEY not in item:
                continue
            rec = item[lrp.POSE_ACTION_KEY]
            rec = rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec)
            rec = rec[offset:]  # row k is now the same timestep as pred[k]
            try:
                pred = _predict(idx)
            except Exception as e:
                logger.warning(f"[openloop] idx {idx}: predict failed: {e}")
                continue
            # Relative items are expressed in the current frame (identity);
            # absolute ones need the observation-time pose for the baseline.
            current_mat = None
            if is_absolute:
                pose = current_pose9(item, n_obs)
                if pose is None:
                    baseline_ok = False
                else:
                    current_mat = lrp.pose9d_to_mat(pose)
            # Both chunks share the observation-time base, so the position-error
            # norm at step k IS the open-loop drift after k+1 executed steps.
            metrics = chunk_metrics(pred, rec, lrp.pose9d_to_mat, current_mat)
            if not metrics["pos_err"]:
                continue
            for name, values in metrics.items():
                rows.setdefault(name, []).append(values)
            n_done += 1
            frames_per_episode[ep] += 1
            evaluated_rows.append(idx)
            if track_actions is not None:
                now_xy, now_vis = track_now(track_actions[lo:hi], idx - lo, track_layout)
                track_frames.append(track_chunk_metrics(pred, rec, now_xy, track_layout))
                if track_camera is not None and track_camera in item:
                    import cv2

                    bgr = cv2.cvtColor(frame_rgb_uint8(item[track_camera]), cv2.COLOR_RGB2BGR)
                    sheet_frames.append({
                        "episode": ep, "frame": idx - lo, "row": idx,
                        "image": cv2.imencode(".png", bgr)[1].tobytes(),
                        "pred": np.asarray(pred), "rec": np.asarray(rec),
                        "now_xy": now_xy, "now_vis": now_vis,
                    })

    if not rows:
        logger.error("[openloop] no frames evaluated — check --repo-id/--root/--episodes")
        return

    logger.info(
        "\n[openloop] frames evaluated per episode: "
        + ", ".join(f"{ep}: {n}" for ep, n in frames_per_episode.items())
    )
    skipped = [ep for ep, n in frames_per_episode.items() if n == 0]
    if skipped and n_done >= args.max_frames:
        logger.warning(
            f"[openloop] --max-frames {args.max_frames} was reached before episode(s) "
            f"{skipped}: they are NOT in the numbers below. Raise --max-frames or --stride."
        )

    def _col(name, k):
        return np.array([r[k] for r in rows[name] if len(r) > k])

    H = max(len(r) for r in rows["pos_err"])
    exec_steps = args.exec_steps or int(getattr(policy.config, "n_action_steps", 0) or 0) or H
    exec_steps = max(1, min(exec_steps, H))

    logger.info(f"\n[openloop] evaluated {len(rows['pos_err'])} frames  (1 chunk inferred per frame)")
    logger.info(
        "  recorded motion over the chunk (from the current pose): mean %.1f mm, %.2f deg"
        % (_col("rec_motion", H - 1).mean() * 1000, _col("rec_rot", H - 1).mean())
    )
    if not baseline_ok:
        logger.warning(
            "  absolute comparison: some items had no 9-D pose in observation.state, "
            "so the move columns are measured from the world origin there"
        )
    logger.info("\n  per-step OPEN-LOOP comparison (mean over frames). Row k = after k+1 steps")
    logger.info("  executed without re-observing. rec move = how far the demo moved from the")
    logger.info("  current pose = the error of predicting 'stand still'. err/move = error")
    logger.info("  relative to that: 0 reproduces the demo, 1 is no better than standing still.")
    logger.info(
        f"    {'step':>4} | {'pos err':>8} | {'rec move':>8} | {'pred move':>9} | {'err/move':>8} "
        f"| {'rot err':>7} | {'rec rot':>7} | {'grip err':>8}"
    )
    logger.info(
        f"    {'':>4} | {'(mm)':>8} | {'(mm)':>8} | {'(mm)':>9} | {'':>8} "
        f"| {'(deg)':>7} | {'(deg)':>7} | {'':>8}"
    )
    for k in range(H):
        err, move, pmove = _col("pos_err", k), _col("rec_motion", k), _col("pred_motion", k)
        mark = "  <- re-observe" if k == exec_steps - 1 else ""
        logger.info(
            f"    {k:>4} | {err.mean()*1000:>8.2f} | {move.mean()*1000:>8.2f} | "
            f"{pmove.mean()*1000:>9.2f} | {ratio_of_means(err, move):>8.2f} | "
            f"{_col('rot_err', k).mean():>7.3f} | {_col('rec_rot', k).mean():>7.3f} | "
            f"{_col('grip_err', k).mean():>8.4f}{mark}"
        )

    k = exec_steps - 1
    err, move, pmove = _col("pos_err", k), _col("rec_motion", k), _col("pred_motion", k)
    ratio = ratio_of_means(err, move)
    speed = ratio_of_means(pmove, move)
    verdict = motion_verdict(ratio)
    logger.info(
        f"\n[openloop] at step {k} (re-observe after {exec_steps} executed steps): "
        f"error {err.mean()*1000:.1f} mm vs demo motion {move.mean()*1000:.1f} mm "
        f"-> err/move {ratio:.2f}; predicted motion = {speed:.2f}x the demo's"
    )
    logger.info(
        f"[openloop] verdict: {verdict}  (err/move {ratio:.2f}; GOOD <= "
        f"{ERR_OVER_MOTION_GOOD}, POOR > {ERR_OVER_MOTION_POOR}, rough guides)\n"
        "  GOOD     -> the chunk follows the demo; robot drift is a DEPLOY problem.\n"
        "  PARTIAL  -> usable for a few steps; lower n_action_steps (--exec-steps\n"
        "              shows the err/move at any step count).\n"
        "  POOR     -> little better than standing still at this step count.\n"
        "  predicted/demo motion well below 1 -> the policy moves too little (slow or\n"
        "  idling); near 1 with a high err/move -> right amount, wrong direction or timing.\n"
        "  Compare with an earlier checkpoint and with episodes NOT used in training\n"
        "  before reading this as overfitting."
    )

    if track_frames:
        log_track_report(stack_track_metrics(track_frames), track_layout, exec_steps)
    if sheet_frames:
        first = sheet_frames[0]
        reobs = min(exec_steps, len(first["pred"]), len(first["rec"])) - 1
        fps = float(getattr(getattr(ds, "meta", None), "fps", None) or getattr(ds, "fps", None) or 15)
        written = write_track_sheets(
            args.track_sheets, sheet_frames, track_layout, exec_steps,
            per_sheet=args.track_sheet_frames, video_fps=max(1.0, fps / max(1, args.stride)),
        )
        logger.info(
            f"\n[openloop] track sheets ({track_camera}): {len(written)} files in {args.track_sheets}\n"
            "  solid = recorded (tracker), dashed with circles = predicted, ring = where the\n"
            "  point is now, "
            f"squares = step {reobs} (re-observe), "
            "grey x = a step predicted hidden.\n"
            "  The MP4 has every evaluated frame, at real-time speed for --stride."
        )

    if not args.diagnose:
        return

    # ── --diagnose ───────────────────────────────────────────────────────────
    # Everything below reuses ONE fetch of each frame, so the two input paths
    # and all sampling variants see the same item (the wrapper draws fresh
    # start-pose noise on every fetch).
    OBS_STATE = "observation.state"
    picks = spread_evenly(evaluated_rows, args.diagnose_frames)
    items = [wrapped[i] for i in picks]
    logger.info(
        f"\n[diagnose] {len(items)} frames, spread evenly over the {len(evaluated_rows)} evaluated ones"
    )

    # 1) + 2) need the checkpoint's preprocessor and a training-style batch.
    if pre is None:
        logger.warning("[diagnose] no preprocessor for this lerobot version: parts 1 and 2 skipped")
    else:
        from torch.utils.data import default_collate

        n_batches = max(1, -(-len(items) // DIAGNOSE_BATCH))
        # items[j::n] mixes far-apart frames in each batch, so rolling a batch by
        # half pairs every frame with observations from a different moment.
        batches = [items[j::n_batches] for j in range(n_batches)]
        max_diff: dict[str, float] = {}
        # Every input the policy declares (state, cameras, environment state, ...).
        compared_keys = list(dict.fromkeys(
            [OBS_STATE, *image_features, *(getattr(policy.config, "input_features", None) or [])]
        ))
        variants = [("the frame's own observations", [])]
        if image_features:
            variants.append(("images from another frame", list(image_features)))
        variants.append(("state from another frame", [OBS_STATE]))
        if image_features:
            variants.append(("images and state from another frame", [*image_features, OBS_STATE]))
        losses: dict[str, list[float]] = {name: [] for name, _ in variants}
        loss_error = None
        for chunk in batches:
            pre.reset()
            raw = default_collate(chunk)
            for cam in camera_keys:  # lerobot_train.py, verbatim: uint8 frames -> float in [0, 1]
                if cam in raw and is_uint8(raw[cam]):
                    raw[cam] = raw[cam].to(dtype=torch.float32) / 255.0
            proc = pre(raw)
            # 1) the same frames, the way _predict_item feeds them
            for i, item in enumerate(chunk):
                for t in range(n_obs):
                    pre.reset()
                    frame = build_batch_frame(
                        item, t, n_obs,
                        skip_key=lrp.POSE_ACTION_KEY, to_batch_tensor=_to_tensor, pre=pre,
                        action_key=ACTION, image_keys=image_features, obs_images_key=OBS_IMAGES,
                        camera_keys=camera_keys,
                    )
                    for key in compared_keys:
                        if key not in frame or key not in proc:
                            continue
                        mine = frame[key].reshape(-1)
                        theirs = proc[key][i]
                        theirs = theirs.reshape(n_obs, -1)[t] if n_obs > 1 else theirs.reshape(-1)
                        diff = float((mine.float() - theirs.float()).abs().max())
                        max_diff[key] = max(max_diff.get(key, 0.0), diff)
            # 2) the training loss, with the right and with the wrong observations
            if loss_error is None:
                try:
                    with torch.no_grad():
                        for name, swap_keys in variants:
                            if swap_keys and len(chunk) < 2:
                                continue
                            batch = dict(proc)
                            for key in swap_keys:
                                if key in batch:
                                    batch[key] = batch[key].roll(max(1, len(chunk) // 2), 0)
                            for _ in range(DIAGNOSE_LOSS_REPEATS):
                                out = policy.forward(dict(batch))
                                loss = out[0] if isinstance(out, (tuple, list)) else out
                                losses[name].append(float(loss))
                except Exception as e:  # a policy whose forward needs more than a batch
                    loss_error = e

        logger.info("\n[diagnose] 1) Does this script feed the model what a training batch does?")
        logger.info("    Each frame is prepared twice from the same dataset item: as a training")
        logger.info("    batch (collate -> preprocessor) and the way this script feeds the policy")
        logger.info("    (one observation step at a time). Largest difference per input:")
        for key, diff in max_diff.items():
            logger.info(f"      {key:<36} {diff:.3e}")
        worst = max(max_diff.values()) if max_diff else float("nan")
        if not max_diff:
            logger.warning("    -> nothing could be compared (no shared observation keys)")
        elif worst <= 1e-5:
            logger.info("    -> identical: the model receives the same tensors as in training.")
        else:
            logger.warning(
                "    -> DIFFERENT: this script's input path does not match training. "
                "Its numbers above are not trustworthy until that is fixed."
            )

        logger.info(
            f"\n[diagnose] 2) Training loss on these frames (policy.forward, eval mode, "
            f"mean of {DIAGNOSE_LOSS_REPEATS} noise draws per batch)"
        )
        first_note = "   <- compare with the loss in the training log"
        if track_layout is not None and track_layout.get("loss_patch"):
            logger.info(
                f"    (trained with the track loss weight {track_layout.get('loss_weight')}: the loss"
                " here weights every action dim alike and counts hidden points' positions, so it"
                " is not the number in the training log; comparing the rows below with each other"
                " is still valid)"
            )
            first_note = "   <- unweighted: not the training log's number (see above)"
        if loss_error is not None:
            logger.warning(f"    could not compute it for this policy: {loss_error}")
        else:
            base = float(np.mean(losses[variants[0][0]])) if losses[variants[0][0]] else float("nan")
            for name, _ in variants:
                if not losses[name]:
                    continue
                value = float(np.mean(losses[name]))
                note = first_note if name == variants[0][0] else (
                    f"   ({value / base:.1f}x)" if base > 0 else "")
                logger.info(f"      {name:<38} {value:.5f}{note}")
            logger.info("    Near 1.0x: the loss does not depend on that input, so the model is not")
            logger.info("    using it. Clearly above 1.0x: the model relies on it.")

    # 3) several samples per frame, and the same sample with one input swapped
    k_samples = max(1, args.diagnose_samples)
    stats: dict[str, list[list[float]]] = {}
    state_keys = [k for k in items[0] if k.startswith(OBS_STATE)] if items else []
    for i, item in enumerate(items):
        other = items[(i + max(1, len(items) // 2)) % len(items)]
        rec = item[lrp.POSE_ACTION_KEY]
        rec = (rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec))[offset:]
        samples = np.stack([_predict_item(item, seed=s) for s in range(k_samples)])
        swapped_image = swapped_state = None
        if other is not item:
            if image_features:
                swapped_image = _predict_item({**item, **{k: other[k] for k in image_features}}, seed=0)
            swapped_state = _predict_item({**item, **{k: other[k] for k in state_keys}}, seed=0)
        one = sampling_diagnostics(samples, rec, swapped_image, swapped_state)
        current_mat = None
        if is_absolute:
            pose = current_pose9(item, n_obs)
            current_mat = lrp.pose9d_to_mat(pose) if pose is not None else None
        one["demo_motion"] = chunk_metrics(samples[0], rec, lrp.pose9d_to_mat, current_mat)["rec_motion"]
        for name, values in one.items():
            stats.setdefault(name, []).append(values)

    def _mm(name, step):
        values = [r[step] for r in stats.get(name, []) if len(r) > step]
        return f"{np.mean(values) * 1000:>9.1f}" if values else f"{'-':>9}"

    last = exec_steps - 1
    logger.info(
        f"\n[diagnose] 3) Sampled chunks: {k_samples} samples per frame with fixed seeds. Position, mm"
    )
    logger.info(f"      {'':<44} {'step 0':>9} {'step ' + str(last):>9}")
    for label, name in (
        ("demo motion from the current pose", "demo_motion"),
        ("error of one sample", "single_err"),
        (f"error of the mean of {k_samples} samples", "mean_err"),
        ("spread between samples (sampling noise)", "spread"),
        ("change when images come from another frame", "image_swap"),
        ("change when state comes from another frame", "state_swap"),
    ):
        logger.info(f"      {label:<44} {_mm(name, 0)} {_mm(name, last)}")
    logger.info("    Mean error well below one sample's, with a spread about as large: sampling")
    logger.info("    noise, not a wrong prediction. Mean error about the same: the prediction is")
    logger.info("    wrong on average. A 'change' no larger than the spread: that input barely")
    logger.info("    moves the output.")


if __name__ == "__main__":
    main()
