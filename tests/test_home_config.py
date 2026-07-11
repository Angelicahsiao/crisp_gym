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

from crisp_gym.config.home import HomeConfig, home_for_env, randomized_home_for  # noqa: E402


class _FakeEnv:
    """Duck-typed env: .config.named_home_configs + .robot.nq/.config.home_config."""

    def __init__(self, nq, named=None, robot_home=None):
        import types

        self.config = types.SimpleNamespace(named_home_configs=named or {})
        self.robot = types.SimpleNamespace(
            nq=nq,
            config=types.SimpleNamespace(home_config=robot_home or [0.0] * nq),
        )


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


def test_home_for_env_prefers_named_config():
    """The consistent per-robot way: env YAML named_home_configs wins."""
    ur_open = [0.2, -1.2, 1.2, -1.6, -1.6, 0.1]
    env = _FakeEnv(nq=6, named={"open_pose": ur_open})
    out = home_for_env(env, "open_pose", noise=0.0)
    np.testing.assert_allclose(out, ur_open)


def test_home_for_env_falls_back_to_enum_on_franka():
    env = _FakeEnv(nq=7)
    out = home_for_env(env, "open_pose", noise=0.0)
    np.testing.assert_allclose(out, HomeConfig.OPEN_POSE.value)


def test_home_for_env_falls_back_to_robot_home_on_ur():
    robot_home = [0.1, -1.5, 1.5, -1.5, -1.5, 0.0]
    env = _FakeEnv(nq=6, robot_home=robot_home)
    out = home_for_env(env, "open_pose", noise=0.0)  # Franka enum skipped (7 != 6)
    np.testing.assert_allclose(out, robot_home)


def test_home_for_env_unknown_name_uses_robot_home():
    robot_home = [0.0] * 6
    env = _FakeEnv(nq=6, robot_home=robot_home)
    out = home_for_env(env, "between_episodes", noise=0.0)
    np.testing.assert_allclose(out, robot_home)


def test_home_for_env_rejects_wrong_size_named_entry():
    env = _FakeEnv(nq=6, named={"open_pose": [0.0] * 7})
    try:
        home_for_env(env, "open_pose")
        raise AssertionError("wrong-size named entry accepted")
    except ValueError as e:
        assert "open_pose" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} home-config tests passed.")
