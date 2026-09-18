"""Tests for scripts/groot_preflight.py.

The checks that matter are the ones with no runtime symptom. A dataset whose
`action` feature carries no per-dimension `names` trains fine and silently
ignores relative_exclude_joints, so the gripper becomes a delta. A dataset that
crisp_gym already converted to relative trains fine and learns deltas of deltas.
Both are pinned here against fixtures shaped like the real umi_robot_full
contract: 10-dim rot6d action, "gripper" as the last name, one wide camera.

Run: python -m pytest tests/test_groot_preflight.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "groot_preflight", REPO / "crisp_gym" / "scripts" / "groot_preflight.py"
)
gp = importlib.util.module_from_spec(_spec)
sys.modules["groot_preflight"] = gp
_spec.loader.exec_module(gp)

POSE_NAMES = ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)]
ACTION_NAMES = POSE_NAMES + ["gripper"]
IDENTITY6 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def make_dataset(root: Path, *, episodes=4, length=120, relative=False,
                 names=ACTION_NAMES, fps=15, task="open the power switch",
                 nan=False, rot_names=None) -> Path:
    """A v3.0-shaped dataset mirroring umi_robot_full_record.yaml's contract."""
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for ep in range(episodes):
        # an absolute TCP path wandering the workspace ~0.45 m from base
        base = np.array([0.45, 0.0, 0.30])
        for t in range(length):
            drift = np.array([0.05 * np.sin(t / 20), 0.05 * np.cos(t / 20), 0.02 * np.sin(t / 9)])
            pos, nxt = base + drift, base + drift + rng.normal(0, 2e-3, 3)
            rot = np.array(IDENTITY6) + rng.normal(0, 0.25, 6)
            grip = float(rng.random())
            state = np.concatenate([pos, rot, [grip]])
            action = (np.concatenate([nxt - pos, np.array(IDENTITY6) + rng.normal(0, 5e-3, 6), [grip]])
                      if relative else np.concatenate([nxt, rot + rng.normal(0, 1e-3, 6), [grip]]))
            if nan and ep == 0 and t == 3:
                action = action.copy(); action[0] = np.nan
            rows.append({"episode_index": ep, "frame_index": t,
                         "index": ep * length + t, "task_index": 0,
                         "observation.state": state.tolist(), "action": action.tolist()})
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame([{"episode_index": e, "tasks": [task], "length": length}
                  for e in range(episodes)]).to_parquet(
        root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task": [task], "task_index": [0]}).set_index("task").to_parquet(
        root / "meta/tasks.parquet")

    pose_names = rot_names or POSE_NAMES
    action_feat = {"dtype": "float32", "shape": [len(names or ACTION_NAMES)]}
    if names is not None:
        action_feat["names"] = names
    json.dump({
        "robot_type": "franka", "codebase_version": "v3.0", "fps": fps,
        "total_episodes": episodes, "total_frames": episodes * length, "total_tasks": 1,
        "features": {
            "action": action_feat,
            "observation.state": {"dtype": "float32", "shape": [10],
                                  "names": pose_names + ["gripper"]},
            "observation.images.oakw_cam": {"dtype": "video", "shape": [800, 1280, 3],
                                            "names": ["height", "width", "channels"]},
        },
    }, open(root / "meta/info.json", "w"), indent=4)
    return root


def run(root: Path, **kw) -> tuple[int, list]:
    """Invoke main() on a dataset, returning (exit code, recorded results)."""
    gp._results.clear()
    argv = ["groot_preflight", str(root)] + [str(x) for pair in kw.items() for x in pair]
    old = sys.argv
    sys.argv = argv
    try:
        gp.main()
    except SystemExit as exc:
        return int(exc.code), list(gp._results)
    finally:
        sys.argv = old
    return 0, list(gp._results)


def status_of(results, prefix):
    return next((s for c, s, _ in results if c.startswith(prefix)), None)


# ── the happy path: your actual contract ──────────────────────────────────────


def test_umi_robot_full_shaped_dataset_is_ready(tmp_path):
    """rot6d, 10-dim, gripper-named, absolute — the contract this repo records."""
    code, res = run(make_dataset(tmp_path / "ds"))
    assert code == 0, [r for r in res if r[1] == "FAIL"]
    assert status_of(res, "G1") == "PASS"
    assert status_of(res, "G2 absolute") == "PASS"
    assert status_of(res, "G3 pose layout") == "PASS"


def test_gripper_dim_is_identified_as_the_excluded_one(tmp_path):
    _, res = run(make_dataset(tmp_path / "ds"))
    msg = next(m for c, _, m in res if c.startswith("G1"))
    assert "gripper" in msg and "dim 9" in msg


# ── G1: the silent no-op ──────────────────────────────────────────────────────


def test_missing_action_names_fails(tmp_path):
    """No names -> _infer_n1_7_action_groups returns [] -> exclude is ignored."""
    code, res = run(make_dataset(tmp_path / "ds", names=None))
    assert code == 1
    assert status_of(res, "G1") == "FAIL"


def test_names_without_a_gripper_dim_fails(tmp_path):
    names = POSE_NAMES + ["aux"]
    code, res = run(make_dataset(tmp_path / "ds", names=names))
    assert code == 1
    assert status_of(res, "G1") == "FAIL"


def test_name_count_mismatch_fails(tmp_path):
    code, res = run(make_dataset(tmp_path / "ds", names=ACTION_NAMES[:-1]))
    assert code == 1
    assert status_of(res, "G1") == "FAIL"


# ── G2: the double-conversion trap ────────────────────────────────────────────


def test_relative_converted_dataset_fails(tmp_path):
    """The one that trains happily and learns deltas of deltas."""
    code, res = run(make_dataset(tmp_path / "ds", relative=True))
    assert code == 1
    assert status_of(res, "G2 absolute") == "FAIL"
    assert status_of(res, "G2 rot6d") == "FAIL"


def test_absolute_dataset_passes_the_next_state_identity(tmp_path):
    """next_tcp_pose lookahead 1 means action[t] tracks state[t+1]."""
    _, res = run(make_dataset(tmp_path / "ds"))
    assert status_of(res, "G2 action ==") in {"PASS", "WARN"}


# ── G3 / G6 / G7 ──────────────────────────────────────────────────────────────


def test_euler_poses_fail_the_layout_check(tmp_path):
    names = ["x", "y", "z", "roll", "pitch", "yaw", "a", "b", "c", "gripper"]
    code, res = run(make_dataset(tmp_path / "ds", names=names))
    assert code == 1
    assert status_of(res, "G3 pose layout") == "FAIL"


def test_episodes_shorter_than_the_chunk_fail(tmp_path):
    code, res = run(make_dataset(tmp_path / "ds", length=20), **{"--chunk-size": 40})
    assert code == 1
    assert status_of(res, "G6 episode vs chunk") == "FAIL"


def test_chunk_period_is_reported_from_fps(tmp_path):
    _, res = run(make_dataset(tmp_path / "ds", fps=15), **{"--chunk-size": 40})
    msg = next(m for c, _, m in res if c.startswith("G6 chunk period"))
    assert "2667 ms" in msg or "2666 ms" in msg


def test_non_finite_values_fail(tmp_path):
    code, res = run(make_dataset(tmp_path / "ds", nan=True))
    assert code == 1
    assert status_of(res, "G7 action") == "FAIL"


def test_placeholder_task_string_warns(tmp_path):
    _, res = run(make_dataset(tmp_path / "ds", task="Finish the task."))
    assert status_of(res, "G5 task text") == "WARN"


def test_camera_key_and_resolution_reported(tmp_path):
    _, res = run(make_dataset(tmp_path / "ds"))
    msg = next(m for c, _, m in res if c.startswith("G4"))
    assert "oakw_cam" in msg
