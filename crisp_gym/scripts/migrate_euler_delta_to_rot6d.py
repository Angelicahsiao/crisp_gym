"""Migrate a legacy (Euler + delta-command) LeRobot dataset to the UMI
absolute-on-disk rot6d schema so it can be trained with
``lerobot_relative_pose.py``.

Why this exists
---------------
Datasets collected with the OLD recording scripts store:
  * ``observation.state.cartesian`` = ``[x, y, z, roll, pitch, yaw]``  (Euler, 6d)
  * ``action``                      = a **delta pose command** (already relative,
                                       produced by the teleop ``stream_fn``)

``lerobot_relative_pose.py`` (the rot6d + relative-at-train wrapper) requires
the UMI convention instead:
  * ``observation.state.cartesian`` = ``[x, y, z, rot6d(6)]``            (9d)
  * ``action``                      = the **absolute** ``next_tcp_pose``
                                       (``pos(3) + rot6d(6) [+ gripper]``); the
                                       trainer relativises it itself.

Feeding the old data directly would (a) misread Euler angles as rot6d rows and
(b) double-relativise the already-delta actions. This script rewrites the data
once so the existing trainer works unchanged.

What it does (per the agreed plan — action reconstruction "from measured obs")
------------------------------------------------------------------------------
For every frame it:
  1. converts ``observation.state.cartesian`` Euler(6) -> rot6d(9)
     (scipy ``from_euler("xyz")`` -> first two rows of R, the UMI convention
     used everywhere else in this repo — see crisp_py ``utils/geometry.py``);
  2. rebuilds the concatenated ``observation.state`` from its sub-features;
  3. **discards the old delta action** and sets
     ``action[t] = absolute measured TCP at t+1`` (``next_tcp_pose``,
     lookahead 1) taken from the converted cartesian pose, with the gripper
     from ``observation.state.gripper`` at ``t+1``. The last frame of each
     episode repeats its own pose (zero relative motion after conversion).

Images / videos, sensors and any other columns are copied through unchanged.

Assumptions (validated at start; the script aborts with a clear message if not)
------------------------------------------------------------------------------
  * ``observation.state.cartesian`` exists and its last 3 dims are Euler xyz
    (radians) — matching ``Pose.to_pos_euler_array`` in crisp_py.
  * a gripper scalar is available as ``observation.state.gripper`` (used for the
    reconstructed action's gripper channel). If absent, pass
    ``--no-action-gripper`` to emit a pose-only (9d) action.

Run it in the lerobot environment (``pixi shell -e <rosdistro>-lerobot``); no
ROS is needed.

Example
-------
    python crisp_gym/scripts/migrate_euler_delta_to_rot6d.py \\
        --input  my_org/old_euler_delta_demo \\
        --output my_org/old_euler_delta_demo_rot6d \\
        --dry-run          # inspect the planned schema change first, then drop it

Then train exactly as in USAGE.md §8:
    python lerobot_relative_pose.py \\
        --dataset.repo_id=my_org/old_euler_delta_demo_rot6d \\
        --policy.type=diffusion --output_dir=outputs/train/umi ...
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger("migrate_euler_delta_to_rot6d")

CARTESIAN_KEY = "observation.state.cartesian"
GRIPPER_KEY = "observation.state.gripper"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# rot6d dim names, consistent with util/lerobot_features.py
ROT6D_NAMES = [f"rot6d_{i}" for i in range(6)]
CART_NAMES = ["x", "y", "z"] + ROT6D_NAMES

# Columns LeRobot manages itself: they appear in info.json / dataset[i] but must
# NOT be passed to create()/add_frame() (add_frame regenerates them).
INTERNAL_KEYS = {"index", "episode_index", "frame_index", "task_index", "timestamp"}


def _is_internal(key: str) -> bool:
    return key in INTERNAL_KEYS or key.startswith("next.")


# ── pose conversion ───────────────────────────────────────────────────────────

def euler_pose_to_rot6d(pose6: np.ndarray) -> np.ndarray:
    """``[x,y,z, roll,pitch,yaw]`` (xyz radians) -> ``[x,y,z, rot6d(6)]``.

    rot6d = first two ROWS of the rotation matrix, flattened row-major — the
    UMI/pytorch3d convention used across this repo.
    """
    pose6 = np.asarray(pose6, dtype=np.float64)
    pos = pose6[:3]
    mat = Rotation.from_euler("xyz", pose6[3:6]).as_matrix()
    rot6d = mat[:2, :].flatten()
    return np.concatenate([pos, rot6d]).astype(np.float32)


# ── feature (info.json) rewriting ─────────────────────────────────────────────

def build_target_features(src_features: dict, action_has_gripper: bool) -> dict:
    """Copy the source feature dict, growing cartesian/state/action for rot6d."""
    if CARTESIAN_KEY not in src_features:
        raise KeyError(
            f"'{CARTESIAN_KEY}' not in dataset features {list(src_features)}. "
            "This script expects a cartesian pose state recorded by the old "
            "collection scripts."
        )

    src_cart_dim = int(np.prod(src_features[CARTESIAN_KEY]["shape"]))
    if src_cart_dim != 6:
        raise ValueError(
            f"'{CARTESIAN_KEY}' has dim {src_cart_dim}, expected 6 "
            "([x,y,z,roll,pitch,yaw]). If it is already 9 the data is likely "
            "already rot6d — no migration needed."
        )

    # Drop LeRobot-managed columns; create()/add_frame() add them back.
    feats = {k: dict(v) for k, v in src_features.items() if not _is_internal(k)}

    # cartesian 6 -> 9
    feats[CARTESIAN_KEY]["shape"] = (9,)
    feats[CARTESIAN_KEY]["names"] = list(CART_NAMES)

    # concatenated observation.state grows by +3 (6 -> 9 in its cartesian slice)
    if STATE_KEY in feats:
        new_names, new_len = _rebuilt_state_names(feats)
        feats[STATE_KEY]["shape"] = (new_len,)
        feats[STATE_KEY]["names"] = new_names

    # action: pos(3) + rot6d(6) [+ gripper]
    act_dim = 9 + (1 if action_has_gripper else 0)
    feats[ACTION_KEY]["shape"] = (act_dim,)
    feats[ACTION_KEY]["names"] = list(CART_NAMES) + (["gripper"] if action_has_gripper else [])
    return feats


def _state_subfeature_order(features: dict) -> list[str]:
    """Sub-features that make up the concatenated ``observation.state`` vector,
    in their dict order (matches ``concatenate_state_features``)."""
    return [
        k
        for k in features
        if k.startswith("observation.state.") and k != STATE_KEY
    ]


def _rebuilt_state_names(features: dict) -> tuple[list[str], int]:
    names: list[str] = []
    for k in _state_subfeature_order(features):
        sub = features[k]
        sub_names = sub.get("names") or [
            f"{k.split('.')[-1]}_{i}" for i in range(int(np.prod(sub["shape"])))
        ]
        names += list(sub_names)
    return names, len(names)


# ── main migration ────────────────────────────────────────────────────────────

def _to_np(v):
    import torch

    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _episode_bounds(dataset) -> list[tuple[int, int]]:
    """Return [(from, to), ...] global-frame ranges per episode."""
    edi = getattr(dataset, "episode_data_index", None)
    if edi is not None and "from" in edi and "to" in edi:
        froms = _to_np(edi["from"]).astype(int).tolist()
        tos = _to_np(edi["to"]).astype(int).tolist()
        return list(zip(froms, tos))
    # Fallback: scan the episode_index column.
    epi = _to_np(dataset.hf_dataset["episode_index"]).astype(int)
    bounds, start = [], 0
    for i in range(1, len(epi) + 1):
        if i == len(epi) or epi[i] != epi[start]:
            bounds.append((start, i))
            start = i
    return bounds


def migrate(args) -> int:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        logger.error(
            "lerobot is required. Run inside 'pixi shell -e <rosdistro>-lerobot'."
        )
        return 2

    from inspect import signature

    logger.info("Loading source dataset: %s", args.input)
    src = LeRobotDataset(repo_id=args.input)
    src_features = src.meta.features
    action_has_gripper = not args.no_action_gripper and GRIPPER_KEY in src_features
    if not args.no_action_gripper and GRIPPER_KEY not in src_features:
        logger.warning(
            "'%s' not found; reconstructed action will be pose-only (9d). "
            "Pass --no-action-gripper to silence this.",
            GRIPPER_KEY,
        )

    target_features = build_target_features(src_features, action_has_gripper)

    logger.info("Planned feature changes:")
    for key in (CARTESIAN_KEY, STATE_KEY, ACTION_KEY):
        if key in src_features:
            logger.info(
                "  %-28s %s -> %s",
                key,
                tuple(src_features[key]["shape"]),
                tuple(target_features[key]["shape"]),
            )
    logger.info(
        "  action gripper channel: %s", "yes" if action_has_gripper else "no"
    )

    bounds = _episode_bounds(src)
    logger.info("Episodes: %d, frames: %d", len(bounds), len(src))

    if args.dry_run:
        logger.info("--dry-run: no data written.")
        return 0

    fps = getattr(src.meta, "fps", None) or args.fps
    robot_type = getattr(src.meta, "robot_type", None) or "unknown"

    dst = LeRobotDataset.create(
        repo_id=args.output,
        fps=fps,
        robot_type=robot_type,
        features=target_features,
        use_videos=True,
    )
    add_frame_has_task = "task" in signature(LeRobotDataset.add_frame).parameters
    state_order = _state_subfeature_order(target_features)

    # copy-through keys = everything except the ones we recompute.
    recomputed = {CARTESIAN_KEY, STATE_KEY, ACTION_KEY, "task"}
    passthrough = [
        k for k in target_features if k not in recomputed and not k.startswith("timestamp")
    ]

    for ep_idx, (start, stop) in enumerate(bounds):
        # Pre-convert the whole episode's cartesian + gripper so the action
        # (next_tcp_pose) can look one frame ahead.
        cart9 = []
        grip = []
        for t in range(start, stop):
            item = src[t]
            cart9.append(euler_pose_to_rot6d(_to_np(item[CARTESIAN_KEY])))
            if action_has_gripper:
                grip.append(np.asarray(_to_np(item[GRIPPER_KEY]), dtype=np.float32).reshape(-1))
        cart9 = np.stack(cart9)

        task = args.task
        for local, t in enumerate(range(start, stop)):
            item = src[t]
            frame = {}

            # recomputed observation.state.cartesian
            frame[CARTESIAN_KEY] = cart9[local].astype(np.float32)

            # passthrough features (images, gripper, sensors, joints, ...)
            for k in passthrough:
                if k in item:
                    val = _to_np(item[k])
                    if k.startswith("observation.state"):
                        # dataset[i] can return e.g. gripper as a 0-d scalar;
                        # coerce back to the feature's declared shape.
                        val = val.astype(np.float32).reshape(target_features[k]["shape"])
                    elif k.startswith("observation.images"):
                        val = _as_uint8_hwc(val)
                    frame[k] = val

            # rebuilt concatenated observation.state
            if STATE_KEY in target_features:
                parts = []
                for sk in state_order:
                    parts.append(
                        frame[sk] if sk in frame else _to_np(item[sk]).astype(np.float32)
                    )
                frame[STATE_KEY] = np.concatenate(
                    [p.reshape(-1) for p in parts]
                ).astype(np.float32)

            # action = next_tcp_pose (absolute pose at t+1); last frame repeats
            nxt = min(local + 1, len(cart9) - 1)
            action = cart9[nxt].astype(np.float32)
            if action_has_gripper:
                action = np.concatenate([action, grip[nxt].reshape(-1)]).astype(np.float32)
            frame[ACTION_KEY] = action

            # task string, if the source carries one per frame
            if "task" in item and item["task"] is not None:
                task = item["task"]

            if add_frame_has_task:
                dst.add_frame(frame, task=task)
            else:
                frame["task"] = task
                dst.add_frame(frame)

        dst.save_episode()
        logger.info("  migrated episode %d/%d (%d frames)", ep_idx + 1, len(bounds), stop - start)

    logger.info("Done. New dataset: %s", args.output)
    return 0


def _as_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """LeRobot returns decoded video frames as float CHW in [0,1]; add_frame
    wants uint8 HWC (what the camera produced at record time). Convert if
    needed, otherwise pass through."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    # CHW float -> HWC uint8
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[-1]:
        a = np.moveaxis(a, 0, -1)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)
    return a


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a legacy Euler + delta-command LeRobot dataset to "
        "the UMI absolute rot6d schema for lerobot_relative_pose.py training."
    )
    parser.add_argument("--input", required=True, help="Source repo_id (legacy dataset).")
    parser.add_argument("--output", required=True, help="Destination repo_id (rot6d dataset).")
    parser.add_argument(
        "--task", default="", help="Task string to stamp if the source has none per-frame."
    )
    parser.add_argument(
        "--no-action-gripper",
        action="store_true",
        help="Emit a pose-only (9d) action instead of pos+rot6d+gripper (10d).",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="FPS fallback if source meta lacks it."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned schema change without writing.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    return migrate(args)


if __name__ == "__main__":
    sys.exit(main())
