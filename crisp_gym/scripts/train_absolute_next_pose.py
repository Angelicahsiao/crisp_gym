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


# ── --epochs support (identical in all three training launchers) ─────────────
# These launchers must stay runnable where only lerobot is installed: a GPU PC
# with one script copied over, or a repo clone that was never pip-installed
# (scripts/train_groot_server.sh's layout). So the epoch scheduler is not a hard
# `import crisp_gym`: the package is tried first, then epoch_steps.py is loaded
# by file path from next to this script, then from the repo layout
# (crisp_gym/util/). epoch_steps.py uses only the standard library. Duplicated
# on purpose, since a shared helper would need exactly the import it avoids;
# tests/test_launcher_epochs.py checks the three copies stay identical.


def _load_epoch_steps():
    """Return the epoch_steps module, or None when it cannot be found."""
    try:
        from crisp_gym.util import epoch_steps

        return epoch_steps
    except ImportError:
        pass

    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for candidate in (here / "epoch_steps.py", here.parent / "util" / "epoch_steps.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("crisp_gym_epoch_steps", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    return None


def _setup_epochs(argv):
    """Strip --epochs from argv.

    Returns:
        (schedule, epochs, remaining_argv): `schedule` is resolve_epoch_steps,
        or None when epoch_steps.py cannot be found; `epochs` is None when the
        flag is absent.

    Raises:
        RuntimeError: --epochs was given but epoch_steps.py cannot be found.
            Falling back to the --steps schedule would look normal and be wrong.
    """
    epoch_steps = _load_epoch_steps()
    if epoch_steps is None:
        if any(a == "--epochs" or a.startswith("--epochs=") for a in argv):
            raise RuntimeError(
                "--epochs needs epoch_steps.py, which was found neither as the "
                "crisp_gym.util.epoch_steps package nor next to this script. "
                "Copy crisp_gym/util/epoch_steps.py into the same folder as this "
                "launcher, or drop --epochs and pass --steps."
            )
        logger.warning(
            "epoch_steps.py not found: --epochs is unavailable and this run's "
            "epoch count will not be logged. Training proceeds on --steps."
        )
        return None, None, list(argv)
    epochs, remaining = epoch_steps.pop_epochs_flag(argv)
    return epoch_steps.resolve_epoch_steps, epochs, remaining


# ── end --epochs support ─────────────────────────────────────────────────────


def main():
    """Run lerobot-train unchanged, stamping absolute-action provenance.

    The dataset factory is monkeypatched only to stamp provenance and is
    returned untouched — observations and actions reach the policy exactly as
    recorded. The factory name differs by lerobot version (>=0.5:
    make_train_eval_datasets; 0.4.x: make_dataset); both are handled so the
    script runs against either.

    CLI: --epochs N schedules by EPOCHS instead of lerobot's --steps, landing a
    checkpoint on the epoch target and training 2 checkpoint intervals past it
    (see crisp_gym/util/epoch_steps.py). OUR flag, stripped before draccus parses.
    """
    import sys

    import lerobot.scripts.lerobot_train as lerobot_train

    schedule_epochs, epochs, remaining = _setup_epochs(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    if hasattr(lerobot_train, "make_train_eval_datasets"):  # lerobot >= 0.5
        original_pair = lerobot_train.make_train_eval_datasets

        def make_train_eval_stamped(cfg):
            dataset, eval_dataset = original_pair(cfg)
            if schedule_epochs:
                schedule_epochs(cfg, dataset, epochs)
            stamp_action_repr(cfg, dataset)
            return dataset, eval_dataset

        lerobot_train.make_train_eval_datasets = make_train_eval_stamped
        logger.info("Patched make_train_eval_datasets (provenance stamp only, no transform).")
    elif hasattr(lerobot_train, "make_dataset"):  # lerobot 0.4.x
        original_single = lerobot_train.make_dataset

        def make_dataset_stamped(cfg):
            dataset = original_single(cfg)
            if schedule_epochs:
                schedule_epochs(cfg, dataset, epochs)
            stamp_action_repr(cfg, dataset)
            return dataset

        lerobot_train.make_dataset = make_dataset_stamped
        logger.info("Patched make_dataset (provenance stamp only, no transform).")
    else:
        # Unlike the relative wrapper, running unpatched here is still CORRECT
        # (no transform is the whole point) — so warn and proceed rather than
        # refuse. Only the provenance stamp is lost.
        #
        # ...UNLESS --epochs was passed: without the hook there is nowhere to
        # read len(dataset), so the epoch target would be silently dropped and
        # the run would use the default --steps schedule while LOOKING correct.
        # That is exactly the failure this flag exists to prevent, so refuse.
        if epochs is not None:
            raise RuntimeError(
                "--epochs was requested but lerobot_train exposes neither "
                "'make_train_eval_datasets' (>=0.5) nor 'make_dataset' (0.4.x), so "
                "the dataset length cannot be read and the epoch target cannot be "
                "applied. Refusing to fall back to the --steps schedule silently. "
                "Either drop --epochs and pass --steps explicitly, or update the "
                "factory name in this script."
            )
        logger.warning(
            "lerobot_train exposes neither make_train_eval_datasets nor "
            "make_dataset — training proceeds with no provenance stamp."
        )

    logging.basicConfig(level=logging.INFO)
    lerobot_train.main()


if __name__ == "__main__":
    main()
