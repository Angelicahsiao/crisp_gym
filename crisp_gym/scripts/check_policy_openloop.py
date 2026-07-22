"""Open-loop policy check: does the trained model reproduce the recorded actions?

Runs on the GPU/training PC (lerobot + torch + the checkpoint). It feeds the
policy the EXACT training-time observations (via RelativePoseDataset, the same
wrapper used for training) and compares the predicted action to the RECORDED
action, frame by frame. No robot, no cameras-live — pure offline eval.

Why this is the decisive test for deployment drift:
  * LOW error  -> the policy reproduces the demonstrations. The model is fine;
                  a drifting real-robot rollout is therefore a DEPLOY-side
                  problem (camera images out-of-distribution, control rate far
                  below the training fps, obs timing), not the policy.
  * HIGH error -> the policy did not learn the task (undertrained, or a
                  training-pipeline bug). No deployment tweak will fix it —
                  retrain / debug training.

It reuses the training conversion verbatim, so it is faithful to what the model
saw. Compares three things per frame:
  - position error of the predicted vs recorded action (meters),
  - rotation error (degrees, geodesic on SO(3)),
  - gripper error.

Usage (on the GPU PC, in the training env):
    python -m crisp_gym.scripts.check_policy_openloop \
        --path outputs/train/<run>/checkpoints/<step>/pretrained_model \
        --repo-id my_org/my_dataset \
        --episodes 0 1 2 --stride 5 --max-frames 200

If --repo-id resolves to a local LeRobotDataset (HF cache or --root), video
frames are decoded for the image inputs exactly as in training.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle (deg) of the relative rotation Ra^T Rb."""
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    p = argparse.ArgumentParser(description="Open-loop action-prediction error on the training data")
    p.add_argument("--path", required=True, help="checkpoint pretrained_model dir")
    p.add_argument("--repo-id", required=True, help="dataset repo id (or local name)")
    p.add_argument("--root", default=None, help="dataset root (if not in HF cache)")
    p.add_argument("--episodes", type=int, nargs="*", default=[0],
                   help="episode indices to evaluate")
    p.add_argument("--stride", type=int, default=5, help="evaluate every Nth frame")
    p.add_argument("--max-frames", type=int, default=200, help="cap total frames")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="override diffusion inference steps (speed)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import get_policy_class

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F811

    from crisp_gym.scripts.lerobot_relative_pose import (
        BASE_KEY,
        POSE_ACTION_KEY,
        RelativePoseDataset,
        pose9d_to_mat,
        rot6d_to_mat,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[openloop] loading {args.path} on {device}")

    train_config = TrainPipelineConfig.from_pretrained(args.path)
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

    from crisp_gym.util.lerobot_features import numpy_obs_to_torch  # ROS-free
    try:
        from lerobot.constants import OBS_IMAGES, ACTION
    except ImportError:
        OBS_IMAGES, ACTION = "observation.images", "action"
    from lerobot.policies.utils import populate_queues

    image_features = list(getattr(policy.config, "image_features", []) or [])
    n_obs = int(getattr(policy.config, "n_obs_steps", 1))

    ds = LeRobotDataset(args.repo_id, root=args.root) if args.root else LeRobotDataset(args.repo_id)
    wrapped = RelativePoseDataset(ds)  # training-time conversion (with wrt-start)

    def _predict(idx: int) -> np.ndarray:
        """One chunk from the model for dataset index idx (training-time input)."""
        policy.reset()
        if pre is not None:
            pre.reset()
        # Build the obs window the same way select_action would, but from the
        # training-converted items so it byte-matches training.
        # RelativePoseDataset returns a single temporally-stacked item already
        # (lerobot delta_timestamps), so we feed it directly.
        item = wrapped[idx]
        batch = {k: v for k, v in item.items() if k != POSE_ACTION_KEY}
        obs = numpy_obs_to_torch({k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
                                  for k, v in batch.items()
                                  if k.startswith("observation")})
        if pre is not None:
            obs = pre(obs)
        obs.pop(ACTION, None)
        if image_features:
            obs = dict(obs)
            obs[OBS_IMAGES] = torch.stack([obs[k] for k in image_features], dim=-4)
        policy._queues = populate_queues(policy._queues, obs)
        with torch.inference_mode():
            act = policy.predict_action_chunk(obs)
            if post is not None:
                act = post(act)
        return act.squeeze(0).to("cpu").numpy()

    pos_err, rot_err, grip_err = [], [], []
    n_done = 0
    from_ep = getattr(ds.meta, "episode_data_index", None)
    for ep in args.episodes:
        # frame range for this episode
        try:
            lo = int(from_ep["from"][ep]); hi = int(from_ep["to"][ep])
        except Exception:
            lo, hi = 0, len(ds)
        for idx in range(lo, hi, args.stride):
            if n_done >= args.max_frames:
                break
            item = wrapped[idx]
            if POSE_ACTION_KEY not in item:
                continue
            rec = item[POSE_ACTION_KEY]
            rec = rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec)
            rec0 = rec[0]  # first action step (relative, model space)
            try:
                pred = _predict(idx)[0]
            except Exception as e:
                logger.warning(f"[openloop] idx {idx}: predict failed: {e}")
                continue
            # both are relative [pos3, rot6d6, gripper]; compare in model space
            Tp = pose9d_to_mat(pred[:9]); Tr = pose9d_to_mat(rec0[:9])
            pos_err.append(float(np.linalg.norm(Tp[:3, 3] - Tr[:3, 3])))
            rot_err.append(_geodesic_deg(Tp[:3, :3], Tr[:3, :3]))
            grip_err.append(abs(float(pred[-1]) - float(rec0[-1])))
            n_done += 1

    if not pos_err:
        logger.error("[openloop] no frames evaluated — check --repo-id/--root/--episodes")
        return

    pos = np.array(pos_err); rot = np.array(rot_err); grp = np.array(grip_err)
    logger.info(f"\n[openloop] evaluated {len(pos)} frames")
    logger.info(f"  action position error  (m):   mean {pos.mean():.4f}  median {np.median(pos):.4f}  p90 {np.percentile(pos,90):.4f}  max {pos.max():.4f}")
    logger.info(f"  action rotation error  (deg): mean {rot.mean():.2f}   median {np.median(rot):.2f}   p90 {np.percentile(rot,90):.2f}   max {rot.max():.2f}")
    logger.info(f"  action gripper error   [0,1]: mean {grp.mean():.4f}  median {np.median(grp):.4f}  max {grp.max():.4f}")
    logger.info(
        "\n[openloop] interpretation:\n"
        "  * per-step position error << the per-step motion scale (~1-2 cm at 15 fps)\n"
        "    and rotation error a few deg => the policy REPRODUCES the demos.\n"
        "    A drifting real rollout is then a DEPLOY problem (images OOD, control\n"
        "    rate << training fps, obs timing) — not the policy.\n"
        "  * errors on the order of the motion itself (or larger) => the policy did\n"
        "    NOT learn the task; retrain / debug training (no deploy tweak helps)."
    )


if __name__ == "__main__":
    main()
