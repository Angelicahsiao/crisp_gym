"""Open-loop policy check: does the trained model reproduce the recorded actions?

STANDALONE — runs on the GPU/training PC with only lerobot + torch + numpy and
a LOCAL copy of `lerobot_relative_pose.py` sitting in the SAME folder (no
crisp_gym import, matching how the training scripts are deployed there).

It feeds the policy the EXACT training-time observations (via the same
RelativePoseDataset wrapper + the policy's delta_timestamps window) and compares
the predicted action to the RECORDED action, frame by frame. No robot.

Decision rule for real-robot drift:
  * LOW error  -> policy reproduces the demos. A drifting rollout is a DEPLOY
                  problem (images OOD, control rate << training fps, timing).
  * HIGH error -> policy did not learn the task; retrain (no deploy tweak helps).

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
    # then wrap with the training relative-pose conversion.
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

    idx_from = getattr(getattr(ds, "meta", None), "episode_data_index", None)
    pos_err, rot_err, grip_err, motion = [], [], [], []
    per_step_pos, per_step_rot, per_step_grip = [], [], []
    n_done = 0
    for ep in args.episodes:
        try:
            lo = int(idx_from["from"][ep]); hi = int(idx_from["to"][ep])
        except Exception:
            lo, hi = 0, len(ds)
        for idx in range(lo, hi, args.stride):
            if n_done >= args.max_frames:
                break
            item = wrapped[idx]
            if lrp.POSE_ACTION_KEY not in item:
                continue
            rec = item[lrp.POSE_ACTION_KEY]
            rec = rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec)
            try:
                pred = _predict(idx)
            except Exception as e:
                logger.warning(f"[openloop] idx {idx}: predict failed: {e}")
                continue
            n_steps = min(len(rec), len(pred))
            # Per-step errors across the WHOLE chunk. Both pred and rec are
            # relative to the SAME base (current frame), so the position-error
            # norm is base-invariant — it IS the absolute open-loop error the
            # robot would have at step k if it executed the chunk without
            # re-observing. Error at step k therefore = the OPEN-LOOP DRIFT
            # after k+1 executed steps.
            fp, fr, fg = [], [], []
            for k in range(n_steps):
                Tp = lrp.pose9d_to_mat(pred[k][:9]); Tr = lrp.pose9d_to_mat(rec[k][:9])
                fp.append(float(np.linalg.norm(Tp[:3, 3] - Tr[:3, 3])))
                fr.append(_geodesic_deg(Tp[:3, :3], Tr[:3, :3]))
                fg.append(abs(float(pred[k][-1]) - float(rec[k][-1])))
            per_step_pos.append(fp); per_step_rot.append(fr); per_step_grip.append(fg)
            # step-0 (single-step) summary kept for the headline numbers.
            pos_err.append(fp[0]); rot_err.append(fr[0]); grip_err.append(fg[0])
            motion.append(float(np.linalg.norm(rec[-1][:3] - rec[0][:3])))
            n_done += 1

    if not pos_err:
        logger.error("[openloop] no frames evaluated — check --repo-id/--root/--episodes")
        return

    pos = np.array(pos_err); rot = np.array(rot_err); grp = np.array(grip_err); mot = np.array(motion)
    logger.info(f"\n[openloop] evaluated {len(pos)} frames  (1 chunk inferred per frame)")
    logger.info(f"  recorded chunk motion span (m): mean {mot.mean():.4f}  median {np.median(mot):.4f}")

    # Ragged chunks -> pad to the max length for a per-step table.
    H = max(len(f) for f in per_step_pos)
    def _col(rows, k):
        return np.array([r[k] for r in rows if len(r) > k])
    logger.info("\n  per-step OPEN-LOOP error (mean over frames) — error at step k")
    logger.info("  = the drift the robot would have after executing k+1 steps")
    logger.info("  without re-observing:")
    logger.info(f"    {'step':>4} | {'pos err (mm)':>12} | {'rot err (deg)':>13} | {'grip err':>8}")
    for k in range(H):
        pk = _col(per_step_pos, k); rk = _col(per_step_rot, k); gk = _col(per_step_grip, k)
        logger.info(f"    {k:>4} | {pk.mean()*1000:>12.2f} | {rk.mean():>13.3f} | {gk.mean():>8.4f}")

    logger.info(f"\n  step-0 (single-step) — pos {pos.mean()*1000:.1f}mm  rot {rot.mean():.2f}deg  grip {grp.mean():.4f}")
    good = pos.mean() < 0.005 and rot.mean() < 2.0
    logger.info(
        f"\n[openloop] verdict: {'POLICY GOOD' if good else 'POLICY WEAK'} "
        f"(step-0 pos {pos.mean()*1000:.1f}mm, rot {rot.mean():.2f}deg)\n"
        "  GOOD  (step-0 pos < ~5mm, rot < ~2deg) -> policy reproduces the demos.\n"
        "  Read the per-step table: if error GROWS a lot by the last executed\n"
        "  step (your deploy n_action_steps), that growth IS the open-loop drift\n"
        "  per chunk — LOWER n_action_steps to re-observe before it accumulates.\n"
        "  If even step-0 is large -> policy weak, retrain."
    )


if __name__ == "__main__":
    main()
