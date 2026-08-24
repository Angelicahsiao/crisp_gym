"""Train with the arm action REPLACED by extra.target_cartesian (commanded pose).

Same dataset, same observations, same gripper action — but the ARM part of the
action is swapped from the recorded `next_tcp_pose` (where the arm ENDED UP) to
`extra.target_cartesian` (the pose that was COMMANDED to the Cartesian impedance
controller). The policy then learns to reproduce the command stream rather than
the achieved trajectory. Everything else is identical to
train_absolute_next_pose.py.

    recorded action : [ next_tcp_pose(9) , gripper ]     (arm = achieved next pose)
    swapped action  : [ target_cartesian(9) , gripper ]  (arm = commanded pose)

The gripper channel (action dim 9) is kept from the recorded action untouched —
only the 9 arm dims (pos3 + rot6d6) are replaced. Observations are NOT changed.
Both stay ABSOLUTE (no relative conversion; for that, compose this idea with
lerobot_relative_pose.py — not done here).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY MEANINGFUL FOR CARTESIAN-DRIVEN DATA (streamed pose -> CIC).

crisp_py writes robot._target_pose only from set_target()/move_to(). On a
JOINT-driven dataset (FACTR leader -> JIC), set_target_joint() never touches it,
so extra.target_cartesian is the measured pose re-seeded after each home() and
then CONSTANT for the whole episode. Training on a constant "action" is
worthless, and it looks like normal data — so this script REFUSES to run when
target_cartesian does not vary within episodes (see _assert_target_varies).
Record the arm in Cartesian space (dric_dual_rscam_franka_umi) if you want this.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Runs on the GPU PC (lerobot >= 0.5 / 0.6.x). ROS-free, crisp-import-free.

Usage:

    python train_action_from_target_cartesian.py \\
        --dataset.repo_id=delta/vive \\
        --policy.type=diffusion \\
        --output_dir=outputs/train/vive_target_cmd \\
        ... (any other lerobot-train args)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

ACTION_KEY = "action"
TARGET_KEY = "extra.target_cartesian"
ARM_DIMS = 9          # pos(3) + rot6d(6); action dim 9 is the gripper, kept as-is
TARGET_DIMS = 9


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class TargetCartesianActionDataset:
    """Wrap a LeRobotDataset so `action`'s arm dims come from extra.target_cartesian.

    lerobot windows the `action` column over the policy's action horizon using
    self.delta_indices. extra.target_cartesian is not a policy feature, so it is
    NOT windowed by default; we inject it into delta_indices (and
    delta_timestamps) with the SAME offsets as `action`, so the wrapper receives
    it stacked and padded identically and the swap is exact frame-for-frame.
    """

    def __init__(self, dataset):
        self._dataset = dataset

        self._assert_features(dataset)
        self._assert_target_varies(dataset)
        self._inject_windowing(dataset)

    # Delegate .meta, .num_frames, .hf_dataset, etc.
    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._dataset)

    @staticmethod
    def _to_np(v) -> np.ndarray:
        return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    # ── preflight checks ──
    @staticmethod
    def _assert_features(dataset) -> None:
        features = getattr(getattr(dataset, "meta", None), "features", None) or {}
        for key, dims in ((ACTION_KEY, ARM_DIMS + 1), (TARGET_KEY, TARGET_DIMS)):
            if key not in features:
                raise KeyError(
                    f"dataset has no '{key}' feature. This script needs both "
                    f"'{ACTION_KEY}' and '{TARGET_KEY}'; the latter is recorded "
                    "by umi_robot_full_record.yaml (source robot.target_pose). "
                    f"features present: {sorted(features)}"
                )
        adim = int(np.prod(features[ACTION_KEY]["shape"]))
        tdim = int(np.prod(features[TARGET_KEY]["shape"]))
        if adim != ARM_DIMS + 1:
            raise ValueError(
                f"'{ACTION_KEY}' is {adim}D; expected {ARM_DIMS + 1} "
                f"([pos3, rot6d6, gripper]). This script only handles the rot6d "
                "UMI action layout."
            )
        if tdim != TARGET_DIMS:
            raise ValueError(
                f"'{TARGET_KEY}' is {tdim}D; expected {TARGET_DIMS} "
                "([pos3, rot6d6]). Record it with representation: rotation_6d."
            )

    def _assert_target_varies(self, dataset, atol: float = 1e-4) -> None:
        """Refuse the FACTR/JIC dead-column case (target_cartesian is constant).

        Reads the low-dim column straight from hf_dataset (no video decode) and
        checks the per-episode spread. If NO episode moves the target by more
        than `atol` in any dim, the column carries no command — training on it is
        meaningless — so we stop with a pointed message rather than produce a
        plausible-looking useless checkpoint.
        """
        hf = getattr(dataset, "hf_dataset", None)
        if hf is None:
            logger.warning(
                "No hf_dataset to preflight %s variance; skipping the dead-column "
                "check. If this is FACTR/JIC data the action will be constant.",
                TARGET_KEY,
            )
            return
        try:
            tc = np.asarray(hf[TARGET_KEY], dtype=np.float64)          # (N, 9)
            ep = np.asarray(hf["episode_index"], dtype=np.int64)       # (N,)
        except Exception as e:
            logger.warning("Could not read %s for the variance check: %s", TARGET_KEY, e)
            return

        max_spread = 0.0
        for e in np.unique(ep):
            seg = tc[ep == e]
            if seg.shape[0] > 1:
                max_spread = max(max_spread, float(np.ptp(seg, axis=0).max()))
        if max_spread <= atol:
            raise ValueError(
                f"'{TARGET_KEY}' is CONSTANT within every episode "
                f"(max per-episode spread {max_spread:.2e} <= {atol:.0e}). This "
                "is the FACTR/JIC signature: robot._target_pose is only written "
                "by Cartesian control (set_target), so on joint-driven data it "
                "is the post-home pose frozen for the episode. Training would "
                "learn a constant. Use a Cartesian-driven dataset "
                "(dric_dual_rscam_franka_umi), or train the recorded action with "
                "train_absolute_next_pose.py instead."
            )
        logger.info(
            "%s varies (max per-episode spread %.3g) — OK to use as action.",
            TARGET_KEY, max_spread,
        )

    def _inject_windowing(self, dataset) -> None:
        """Make lerobot window TARGET_KEY exactly like ACTION_KEY."""
        di = getattr(dataset, "delta_indices", None)
        dt = getattr(dataset, "delta_timestamps", None)
        self._windowed = False
        if di is not None and ACTION_KEY in di:
            di[TARGET_KEY] = di[ACTION_KEY]
            self._windowed = True
        if dt is not None and ACTION_KEY in dt:
            dt[TARGET_KEY] = dt[ACTION_KEY]
        if not self._windowed:
            # No action horizon configured (single-step policy): action arrives
            # as a single frame and so does TARGET_KEY — the swap still works,
            # frame-for-frame. Nothing to inject.
            logger.info(
                "No action horizon in delta_indices; swapping single-frame "
                "actions (policy has no action chunk)."
            )

    # ── the swap ──
    def convert_item(self, item: dict) -> dict:
        if ACTION_KEY not in item or TARGET_KEY not in item:
            raise KeyError(
                f"item is missing '{ACTION_KEY}' or '{TARGET_KEY}'. The windowing "
                "injection did not take — lerobot may have precomputed its query "
                "differently in this version. Add extra.target_cartesian to the "
                "policy's delta_timestamps, or preprocess the dataset offline."
            )
        a = self._to_np(item[ACTION_KEY]).astype(np.float64)
        tc = self._to_np(item[TARGET_KEY]).astype(np.float64)
        if a.shape[:-1] != tc.shape[:-1]:
            raise ValueError(
                f"'{ACTION_KEY}' window {a.shape} and '{TARGET_KEY}' window "
                f"{tc.shape} disagree — target was not windowed like action, so "
                "the horizons are not aligned. See _inject_windowing()."
            )
        a[..., :ARM_DIMS] = tc[..., :ARM_DIMS]     # arm := commanded pose; gripper kept
        item[ACTION_KEY] = torch.from_numpy(a.astype(np.float32))
        return item

    def __getitem__(self, idx: int) -> dict:
        return self.convert_item(self._dataset[idx])


# ── Stats recomputation ───────────────────────────────────────────────────────

def recompute_action_stats(wrapped: TargetCartesianActionDataset, num_samples: int = 2000) -> None:
    """Overwrite meta.stats['action'] with stats of the SWAPPED action.

    lerobot's normalizer uses dataset-wide stats computed on the raw parquet
    `action` (the achieved next pose). After the swap the action distribution is
    different, so we resample and overwrite — exactly as lerobot_relative_pose.py
    does for its relative values, including the no-video guard below.
    """
    n = len(wrapped)
    indices = np.linspace(0, n - 1, min(num_samples, n)).astype(int)

    # CRITICAL (see lerobot_relative_pose.py): do NOT decode video in this
    # main-process pass. A torchcodec decoder opened here is forked into the
    # DataLoader workers broken, and every worker dies at step 0. action is
    # low-dim, so no-op the video query while sampling.
    ds = wrapped._dataset
    _orig_query_videos = getattr(ds, "_query_videos", None)
    if _orig_query_videos is not None:
        ds._query_videos = lambda *a, **k: {}
    try:
        collected: list[np.ndarray] = []
        for i in indices:
            v = wrapped[int(i)][ACTION_KEY]
            v = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            collected.append(v.reshape(-1, v.shape[-1]))
    finally:
        if _orig_query_videos is not None:
            ds._query_videos = _orig_query_videos

    data = np.concatenate(collected, axis=0)
    wrapped.meta.stats[ACTION_KEY] = {
        "mean": torch.from_numpy(data.mean(axis=0).astype(np.float32)),
        "std": torch.from_numpy((data.std(axis=0) + 1e-8).astype(np.float32)),
        "min": torch.from_numpy(data.min(axis=0).astype(np.float32)),
        "max": torch.from_numpy(data.max(axis=0).astype(np.float32)),
    }
    logger.info(f"Recomputed action stats over {len(data)} frames (arm := {TARGET_KEY}).")


# ── Provenance ────────────────────────────────────────────────────────────────

def _installed_lerobot_version() -> str:
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
    """Write action_repr.json: the arm action is the COMMANDED pose, not achieved."""
    try:
        meta = getattr(dataset, "meta", None)
        info = {
            "action": {
                "source": TARGET_KEY,
                "pose_repr": "absolute",
                "layout": "[x, y, z, rot6d(6), gripper]",
                "note": "ARM dims (0-8) replaced with extra.target_cartesian "
                        "(pose COMMANDED to the CIC); gripper (dim 9) kept from "
                        "the recorded next_tcp_pose action. Observations "
                        "unchanged/absolute. Deploy as an absolute-pose policy; "
                        "NOT a relative (RelativeLerobotPolicy) checkpoint.",
            },
            "fps": getattr(meta, "fps", None),
            "lerobot_version_target": _installed_lerobot_version(),
        }
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "action_repr.json").write_text(json.dumps(info, indent=2))
        logger.info(f"Stamped {out_dir / 'action_repr.json'}")
    except Exception as e:
        logger.warning(f"Could not stamp action_repr.json: {e}")


# ── Training launcher ─────────────────────────────────────────────────────────

def main():
    """Run lerobot-train with the target-cartesian action swap injected."""
    import lerobot.scripts.lerobot_train as lerobot_train

    def _wrap(dataset, cfg):
        wrapped = TargetCartesianActionDataset(dataset)
        logger.info("Wrapped dataset: action arm dims := extra.target_cartesian.")
        recompute_action_stats(wrapped)
        stamp_action_repr(cfg, wrapped)
        return wrapped

    if hasattr(lerobot_train, "make_train_eval_datasets"):  # lerobot >= 0.5
        original_pair = lerobot_train.make_train_eval_datasets

        def make_train_eval_swapped(cfg):
            dataset, eval_dataset = original_pair(cfg)
            wrapped = _wrap(dataset, cfg)
            if eval_dataset is not None:
                # Swap the eval split too, else a command-trained policy is
                # scored against achieved-pose targets — plausible metrics, no
                # error. Stats/stamp stay train-only.
                eval_dataset = TargetCartesianActionDataset(eval_dataset)
                logger.info("Wrapped eval dataset with the same action swap.")
            return wrapped, eval_dataset

        lerobot_train.make_train_eval_datasets = make_train_eval_swapped
        logger.info("Patched make_train_eval_datasets (lerobot >= 0.5 layout).")
    elif hasattr(lerobot_train, "make_dataset"):  # lerobot 0.4.x
        original_single = lerobot_train.make_dataset

        def make_dataset_swapped(cfg):
            return _wrap(original_single(cfg), cfg)

        lerobot_train.make_dataset = make_dataset_swapped
        logger.info("Patched make_dataset (lerobot 0.4.x layout).")
    else:
        raise RuntimeError(
            "lerobot.scripts.lerobot_train exposes neither 'make_train_eval_datasets' "
            "(>=0.5) nor 'make_dataset' (0.4.x) — the action swap cannot be injected. "
            "Without it training would run on the RECORDED action (achieved pose), "
            "silently ignoring the swap. Find the dataset factory lerobot_train now "
            "calls and patch that name."
        )

    logging.basicConfig(level=logging.INFO)
    lerobot_train.main()


if __name__ == "__main__":
    main()
