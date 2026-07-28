"""Verify the relative-pose representation of a migrated / recorded rot6d
dataset — the numeric proof that train-time and deploy-time frames agree.

Run in the lerobot environment, next to lerobot_relative_pose.py:

    python check_relative_pose.py /path/to/dataset_rot6d

It wraps the dataset with RelativePoseDataset (no start-pose noise) and, over a
random sample of frames, checks:

  1. IDENTITY   — the current observed TCP frame expressed relative to itself is
                  the identity pose [0,0,0, 1,0,0, 0,1,0]. If not ~0, base-frame
                  selection or rot6d encode/decode is wrong.
  2. ROUND-TRIP — composing the relative action back onto the absolute base pose
                  (T_cmd = T_current ∘ T_rel, the exact deployment math)
                  reproduces the on-disk absolute next_tcp_pose. ~1e-6 means
                  train-time and deploy-time frames match.
  3. MAGNITUDE  — relative action translations are small (one control step of
                  motion, ~mm–cm). Huge values mean absolute poses leaked in.
  4. GRIPPER    — the gripper channel passes through the conversion unchanged.

Prints PASS if identity and round-trip are within tolerance.
"""

from __future__ import annotations

import sys

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_relative_pose import (  # self-contained; ships next to this file
    RelativePoseDataset,
    mat_to_pose9d,
    pose9d_to_mat,
)

IDENTITY_9D = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python check_relative_pose.py <dataset_repo_id_or_path> [num_samples]")
        return 2
    repo = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    ds = LeRobotDataset(repo_id=repo)
    wrapped = RelativePoseDataset(ds, start_pose_noise_scale=0.0)  # no aug noise

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(num_samples, len(ds)), replace=False)

    max_id_err = max_rt_err = 0.0
    rel_pos = []
    for i in idxs:
        i = int(i)
        raw, w = ds[i], wrapped[i]
        raw_cart = np.asarray(raw["observation.state.cartesian"], np.float64).reshape(-1)[:9]
        w_cart = np.asarray(w["observation.state.cartesian"], np.float64).reshape(-1)[:9]
        raw_act = np.asarray(raw["action"], np.float64).reshape(-1)
        w_act = np.asarray(w["action"], np.float64).reshape(-1)

        # (1) current obs frame relative to itself -> identity
        max_id_err = max(max_id_err, np.abs(w_cart - IDENTITY_9D).max())
        # (2) round-trip: base ∘ T_rel(action) recovers the absolute action pose
        recon = mat_to_pose9d(pose9d_to_mat(raw_cart) @ pose9d_to_mat(w_act[:9]))
        max_rt_err = max(max_rt_err, np.abs(recon - raw_act[:9]).max())
        # (3) relative action translation size (one step of motion)
        rel_pos.append(float(np.linalg.norm(w_act[:3])))
        # (4) gripper channel untouched by the relative conversion
        if w_act.shape[0] > 9:
            assert np.isclose(w_act[9], raw_act[9]), "gripper changed by rel conversion!"

    print(f"samples checked          : {len(idxs)}")
    print(f"(1) identity err   max   : {max_id_err:.2e}   [expect ~0]")
    print(f"(2) round-trip err max   : {max_rt_err:.2e}   [expect <1e-5]")
    print(
        f"(3) rel action |Δpos| m  : mean {np.mean(rel_pos):.4f}  "
        f"p50 {np.median(rel_pos):.4f}  max {np.max(rel_pos):.4f}"
    )
    ok = max_id_err < 1e-4 and max_rt_err < 1e-4
    print("PASS" if ok else "FAIL — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
