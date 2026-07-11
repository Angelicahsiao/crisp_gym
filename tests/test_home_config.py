"""Tests for robot-aware home-config selection (config/home.py).

Regression for: HomeConfig enum poses are Franka (7-joint); sending one to a
6-joint UR made the controller silently reject the trajectory, so the robot
never homed between episodes (observed with FACTR teleop recording).

Runs under pytest or directly:  python tests/test_home_config.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crisp_gym.config.home import HomeConfig, randomized_home_for  # noqa: E402


def test_preferred_used_when_joint_count_matches():
    fallback = [0.0] * 7
    out = randomized_home_for(7, fallback, preferred=HomeConfig.OPEN_POSE, noise=0.0)
    np.testing.assert_allclose(out, HomeConfig.OPEN_POSE.value)
    assert len(out) == 7


def test_fallback_used_when_preferred_mismatches():
    """The UR7e case: 6 joints, Franka HomeConfig(7) must NOT be sent."""
    fallback = [0.1, -1.5, 1.5, -1.5, -1.5, 0.0]  # UR-style 6-joint home
    out = randomized_home_for(6, fallback, preferred=HomeConfig.OPEN_POSE, noise=0.0)
    np.testing.assert_allclose(out, fallback)
    assert len(out) == 6


def test_no_preferred_uses_fallback_with_noise():
    fallback = [0.0] * 6
    out = randomized_home_for(6, fallback, preferred=None, noise=0.05)
    assert len(out) == 6
    assert all(abs(v) <= 0.05 for v in out)


def test_mismatched_fallback_raises():
    try:
        randomized_home_for(6, [0.0] * 7, preferred=None)
        raise AssertionError("mismatched fallback accepted")
    except ValueError as e:
        assert "6" in str(e) and "7" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} home-config tests passed.")
