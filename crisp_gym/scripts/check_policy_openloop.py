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


def main():
    p = argparse.ArgumentParser(description="Open-loop action-prediction error vs the demos")
    p.add_argument("--path", required=True, help="checkpoint pretrained_model dir")
    p.add_argument("--repo-id", required=True, help="dataset repo id or local path")
    p.add_argument("--root", default=None, help="dataset root (if repo-id is a plain name)")
    p.add_argument("--episodes", type=int, nargs="*", default=[0])
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--num-inference-steps", type=int, default=None)
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
    wrapped = lrp.RelativePoseDataset(ds)

    image_features = list(getattr(policy.config, "image_features", []) or [])
    n_obs = int(getattr(policy.config, "n_obs_steps", 1))

    def _to_batch_frame(item: dict, t: int) -> dict:
        """One timestep (t-th of the obs window) as a batched policy input."""
        frame = {}
        for k, v in item.items():
            if k == lrp.POSE_ACTION_KEY or not k.startswith("observation"):
                continue
            vt = v[t] if (hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == n_obs) else v
            vt = torch.as_tensor(np.asarray(vt.numpy() if hasattr(vt, "numpy") else vt))
            frame[k] = vt.unsqueeze(0).to(device).float()
        if pre is not None:
            frame = pre(frame)
        frame.pop(ACTION, None)
        if image_features:
            frame = dict(frame)
            frame[OBS_IMAGES] = torch.stack([frame[k] for k in image_features], dim=-4)
        return frame

    def _predict(idx: int) -> np.ndarray:
        policy.reset()
        if pre is not None:
            pre.reset()
        item = wrapped[idx]
        last = None
        for t in range(n_obs):
            last = _to_batch_frame(item, t)
            policy._queues = populate_queues(policy._queues, last)
        with torch.inference_mode():
            act = policy.predict_action_chunk(last)
            if post is not None:
                act = post(act)
        return act.squeeze(0).to("cpu").numpy()

    idx_from = getattr(getattr(ds, "meta", None), "episode_data_index", None)
    pos_err, rot_err, grip_err, motion = [], [], [], []
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
            rec0 = rec[0]
            try:
                pred0 = _predict(idx)[0]
            except Exception as e:
                logger.warning(f"[openloop] idx {idx}: predict failed: {e}")
                continue
            Tp = lrp.pose9d_to_mat(pred0[:9]); Tr = lrp.pose9d_to_mat(rec0[:9])
            pos_err.append(float(np.linalg.norm(Tp[:3, 3] - Tr[:3, 3])))
            rot_err.append(_geodesic_deg(Tp[:3, :3], Tr[:3, :3]))
            grip_err.append(abs(float(pred0[-1]) - float(rec0[-1])))
            motion.append(float(np.linalg.norm(Tr[:3, 3])))  # recorded step size
            n_done += 1

    if not pos_err:
        logger.error("[openloop] no frames evaluated — check --repo-id/--root/--episodes")
        return

    pos = np.array(pos_err); rot = np.array(rot_err); grp = np.array(grip_err); mot = np.array(motion)
    logger.info(f"\n[openloop] evaluated {len(pos)} frames")
    logger.info(f"  recorded per-step motion (m): mean {mot.mean():.4f}  median {np.median(mot):.4f}")
    logger.info(f"  action position error   (m): mean {pos.mean():.4f}  median {np.median(pos):.4f}  p90 {np.percentile(pos,90):.4f}  max {pos.max():.4f}")
    logger.info(f"  action rotation error (deg): mean {rot.mean():.2f}   median {np.median(rot):.2f}   p90 {np.percentile(rot,90):.2f}")
    logger.info(f"  action gripper error  [0,1]: mean {grp.mean():.4f}  median {np.median(grp):.4f}  max {grp.max():.4f}")
    ratio = pos.mean() / max(mot.mean(), 1e-9)
    logger.info(
        f"\n[openloop] position-error / recorded-motion ratio = {ratio:.2f}\n"
        "  << 1  -> policy REPRODUCES the demos; drift is a DEPLOY problem\n"
        "          (images OOD, control rate << training fps, obs timing).\n"
        "  ~1 or > -> policy did NOT learn the task; retrain / debug training."
    )


if __name__ == "__main__":
    main()
