"""Test the offline action-swap core (swap_arm + feature guard), numpy-only.

swap_action_offline.py rewrites the `action` column on disk so its arm dims come
from extra.target_cartesian, keeping the gripper. The parquet/stats plumbing is
reused (and already tested) from migrate_euler_delta_to_rot6d.py; here we pin the
pure logic: the arm is replaced, the gripper survives, the input is not mutated,
and the feature guard rejects a missing target or a wrong-width action.

pandas + the migrate import are stubbed so this runs anywhere numpy is present.

Run:  python tests/test_swap_action_offline.py   (or via pytest)
"""

import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    stub = types.ModuleType("migrate_euler_delta_to_rot6d")
    stub.data_parquet_files = lambda *a, **k: []
    stub.load_info = lambda *a, **k: {}
    stub.update_stats_files = lambda *a, **k: None
    sys.modules["migrate_euler_delta_to_rot6d"] = stub
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    return SourceFileLoader(
        "swap_offline", str(REPO / "crisp_gym" / "scripts" / "swap_action_offline.py")
    ).load_module()


def test_swap_arm_replaces_arm_keeps_gripper_and_does_not_mutate():
    m = _load()
    act = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0.3],
                    [-2, -2, -2, -2, -2, -2, -2, -2, -2, 0.7]], dtype=np.float32)
    tc = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.float32)
    out = m.swap_arm(act, tc)
    assert out.shape == (2, 10)
    assert np.allclose(out[:, :9], tc), "arm dims must become target_cartesian"
    assert np.allclose(out[:, 9], [0.3, 0.7]), "gripper (dim 9) must be preserved"
    assert np.allclose(act[:, :9], [[-1] * 9, [-2] * 9]), "input act must not be mutated"


def test_features_guard_rejects_missing_target():
    m = _load()
    try:
        m._assert_features({"features": {"action": {"shape": [10]}}})
        raise AssertionError("missing target not rejected")
    except KeyError as e:
        assert "extra.target_cartesian" in str(e)


def test_features_guard_rejects_wrong_action_dim():
    m = _load()
    try:
        m._assert_features({"features": {
            "action": {"shape": [7]},
            "extra.target_cartesian": {"shape": [9]},
        }})
        raise AssertionError("wrong action dim not rejected")
    except ValueError as e:
        assert "10D" in str(e)


def test_features_guard_accepts_valid():
    m = _load()
    m._assert_features({"features": {
        "action": {"shape": [10]},
        "extra.target_cartesian": {"shape": [9]},
    }})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} offline-swap tests passed.")
