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

Policies: diffusion, and language-conditioned ones (multi_task_dit, GR00T).
Each frame carries the dataset's own task string into the preprocessor, as in
training — those policies tokenize it, and multi_task_dit fails with a shape
error without it. Observation queues and the stacked `observation.images`
tensor are used only when the policy keeps them (diffusion, multi_task_dit);
GR00T has neither and takes the single latest frame directly.

Usage (GPU PC, training env, script next to lerobot_relative_pose.py):
    python3 check_policy_openloop.py \
        --path .../checkpoints/100000/pretrained_model \
        --repo-id /abs/path/to/dataset \
        --episodes 0 1 2 --stride 5 --max-frames 200
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
) -> dict:
    """One timestep (the t-th of the obs window) as a batched policy input.

    Takes every ``observation*`` key (minus ``skip_key``, the recorded action),
    adds the task string BEFORE preprocessing so a tokenizer step sees it,
    drops any ``action`` the preprocessor injects, and stacks the cameras only
    when ``stack_images`` is given. Tensor conversion is injected so this stays
    testable without torch.
    """
    frame = {}
    for k, v in item.items():
        if k == skip_key or not k.startswith("observation"):
            continue
        vt = v[t] if (hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == n_obs) else v
        frame[k] = to_batch_tensor(vt)
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
        nonlocal task_logged
        policy.reset()
        if pre is not None:
            pre.reset()
        item = wrapped[idx]
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
            )
            if queues is not None:
                policy._queues = populate_queues(policy._queues, last)
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


if __name__ == "__main__":
    main()
