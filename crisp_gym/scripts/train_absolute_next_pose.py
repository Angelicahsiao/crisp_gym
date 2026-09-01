"""Train on the dataset AS RECORDED — absolute next-TCP-pose action, no transform.

This is the baseline launcher for the UMI robot datasets: it runs lerobot-train
with NO change to observations or actions. The `action` column is consumed
exactly as it sits on disk —

    action = [x, y, z, rot6d(6), gripper]   (10D, ABSOLUTE next TCP pose)

i.e. the policy learns to predict the next absolute Cartesian pose (lookahead=1
at record time), NOT a relative/delta pose. It is the counterpart to
lerobot_relative_pose.py (which re-expresses every pose relative to the current
TCP frame at dataloader level); use THIS script when you want the plain
absolute-pose behavioral-cloning baseline.

Because nothing is transformed, this is almost `lerobot-train` verbatim. The one
thing it adds is a provenance stamp (action_repr.json) next to the checkpoints,
so the serving side can tell an absolute-next-pose checkpoint from a relative
one instead of guessing. Training is never blocked if the stamp fails.

Runs on the GPU PC (lerobot >= 0.5 / 0.6.x). ROS-free, crisp-import-free — keep
it that way so it stays runnable anywhere lerobot is installed.

Usage:

    python train_absolute_next_pose.py \\
        --dataset.repo_id=delta/vive \\
        --policy.type=diffusion \\
        --output_dir=outputs/train/vive_absolute \\
        ... (any other lerobot-train args)

Any native lerobot-train argument works unchanged (the script only patches the
dataset factory to stamp provenance, then calls lerobot_train.main()).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _installed_lerobot_version() -> str:
    """lerobot version importable here; 'unknown' rather than raising."""
    try:
        from importlib.metadata import version

        return version("lerobot")
    except Exception:
        try:
            import lerobot

            return getattr(lerobot, "__version__", "unknown")
        except Exception:
            return "unknown"


def stamp_action_repr(cfg, dataset) -> None:
    """Write action_repr.json next to the checkpoints (provenance only).

    Records that this checkpoint outputs ABSOLUTE next-TCP poses, so the deploy
    side does not mistake it for a relative-pose checkpoint (which composes
    T_cmd = T_current @ T_rel). Deliberately NOT named pose_repr.json: that file
    is the relative-pipeline's contract, and an absolute checkpoint must not
    masquerade as one. Never fails training.
    """
    try:
        meta = getattr(dataset, "meta", None)
        features = getattr(meta, "features", {}) or {}
        action_feat = features.get("action", {})
        info = {
            "action": {
                "source": "next_tcp_pose",
                "pose_repr": "absolute",
                "layout": "[x, y, z, rot6d(6), gripper]",
                "names": action_feat.get("names"),
                "note": "policy predicts the ABSOLUTE next TCP pose; no relative "
                        "conversion. Do NOT deploy with RelativeLerobotPolicy, "
                        "which expects a relative-pose checkpoint.",
            },
            "observation": {
                "pose_repr": "absolute",
                "note": "observation.state consumed as recorded (rot6d absolute).",
            },
            "fps": getattr(meta, "fps", None),
            "lerobot_version_target": _installed_lerobot_version(),
        }
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "action_repr.json").write_text(json.dumps(info, indent=2))
        logger.info(f"Stamped {out_dir / 'action_repr.json'}")
    except Exception as e:  # provenance must never kill a training run
        logger.warning(f"Could not stamp action_repr.json: {e}")


def main():
    """Run lerobot-train unchanged, stamping absolute-action provenance.

    The dataset factory is monkeypatched only to stamp provenance and is
    returned untouched — observations and actions reach the policy exactly as
    recorded. The factory name differs by lerobot version (>=0.5:
    make_train_eval_datasets; 0.4.x: make_dataset); both are handled so the
    script runs against either.
    """
    import lerobot.scripts.lerobot_train as lerobot_train

    if hasattr(lerobot_train, "make_train_eval_datasets"):  # lerobot >= 0.5
        original_pair = lerobot_train.make_train_eval_datasets

        def make_train_eval_stamped(cfg):
            dataset, eval_dataset = original_pair(cfg)
            stamp_action_repr(cfg, dataset)
            return dataset, eval_dataset

        lerobot_train.make_train_eval_datasets = make_train_eval_stamped
        logger.info("Patched make_train_eval_datasets (provenance stamp only, no transform).")
    elif hasattr(lerobot_train, "make_dataset"):  # lerobot 0.4.x
        original_single = lerobot_train.make_dataset

        def make_dataset_stamped(cfg):
            dataset = original_single(cfg)
            stamp_action_repr(cfg, dataset)
            return dataset

        lerobot_train.make_dataset = make_dataset_stamped
        logger.info("Patched make_dataset (provenance stamp only, no transform).")
    else:
        # Unlike the relative wrapper, running unpatched here is still CORRECT
        # (no transform is the whole point) — so warn and proceed rather than
        # refuse. Only the provenance stamp is lost.
        logger.warning(
            "lerobot_train exposes neither make_train_eval_datasets nor "
            "make_dataset — training proceeds with no provenance stamp."
        )

    logging.basicConfig(level=logging.INFO)
    lerobot_train.main()


if __name__ == "__main__":
    main()
