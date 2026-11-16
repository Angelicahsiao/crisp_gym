"""The workspace box and the per-step target limit (crisp_gym/util/safety.py).

Both run on every Cartesian command a policy or a teleop device sends, so the
properties pinned here are the safety-relevant ones: a bound of 0.0 is a bound
(it used to be dropped), a step inside the limit is untouched, a step outside
it keeps its direction and is cut to exactly the limit, and translation and
rotation are limited independently.
"""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("safety", REPO / "crisp_gym" / "util" / "safety.py")
safety = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(safety)


def _angle_deg(a, b):
    return float(np.degrees(np.linalg.norm((a * b.inv()).as_rotvec())))


def test_zero_is_a_bound_and_none_is_not():
    box = safety.workspace_box(min_z=0.0, max_x=0.0)
    assert box["lower"].tolist() == [-np.inf, -np.inf, 0.0]
    assert box["upper"].tolist() == [0.0, np.inf, np.inf]
    clipped = np.clip([0.3, 0.1, -0.05], box["lower"], box["upper"])
    assert clipped.tolist() == [0.0, 0.1, 0.0]  # the old `or` code let both through
    free = safety.workspace_box()
    assert np.all(np.isinf(free["lower"])) and np.all(np.isinf(free["upper"]))


def test_below_the_base_is_allowed_unless_asked():
    """A mobile-base or table-edge mount reaches below z = 0: no floor by default."""
    box = safety.workspace_box()
    assert np.clip([0.4, 0.0, -0.3], box["lower"], box["upper"])[2] == -0.3
    box = safety.workspace_box(min_z=-0.5)
    assert np.clip([0.4, 0.0, -0.8], box["lower"], box["upper"])[2] == -0.5


def test_small_steps_pass_untouched():
    R0 = Rotation.from_euler("xyz", [0.1, -0.2, 0.3])
    p0 = np.array([0.4, 0.0, 0.3])
    p1 = p0 + [0.01, -0.01, 0.005]
    R1 = Rotation.from_euler("z", 3, degrees=True) * R0
    p, R, cut_t, cut_r = safety.limit_target_step(p0, R0, p1, R1, 0.03, 5.0)
    assert not cut_t and not cut_r
    assert np.allclose(p, p1) and _angle_deg(R, R1) < 1e-9


def test_long_translation_is_cut_to_the_limit_along_its_direction():
    p0 = np.array([0.4, 0.0, 0.3])
    p1 = p0 + [0.3, 0.4, 0.0]  # 0.5 m in one step
    R0 = Rotation.identity()
    p, R, cut_t, cut_r = safety.limit_target_step(p0, R0, p1, R0, 0.03, 5.0)
    assert cut_t and not cut_r
    assert np.isclose(np.linalg.norm(p - p0), 0.03)
    assert np.allclose((p - p0) / 0.03, [0.6, 0.8, 0.0])  # same direction


def test_long_rotation_is_cut_on_the_shortest_path():
    R0 = Rotation.from_euler("xyz", [0.3, 0.2, -0.1])
    axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    R1 = Rotation.from_rotvec(axis * np.radians(40)) * R0
    p0 = np.zeros(3)
    p, R, cut_t, cut_r = safety.limit_target_step(p0, R0, p0, R1, 0.03, 5.0)
    assert cut_r and not cut_t
    assert np.isclose(_angle_deg(R, R0), 5.0)
    assert np.isclose(_angle_deg(R1, R), 35.0)  # what is left lies on the same path
    rv = (R * R0.inv()).as_rotvec()
    assert np.allclose(rv / np.linalg.norm(rv), axis)


def test_translation_and_rotation_are_independent():
    R0 = Rotation.identity()
    p0 = np.zeros(3)
    p, R, cut_t, cut_r = safety.limit_target_step(
        p0, R0, np.array([0.1, 0, 0]), Rotation.from_euler("z", 2, degrees=True), 0.03, 5.0
    )
    assert cut_t and not cut_r and np.isclose(_angle_deg(R, R0), 2.0)
    p, R, cut_t, cut_r = safety.limit_target_step(
        p0, R0, np.array([0.001, 0, 0]), Rotation.from_euler("z", 30, degrees=True), None, 5.0
    )
    assert cut_r and not cut_t and np.allclose(p, [0.001, 0, 0])


def test_no_limits_means_no_change():
    R0, p0 = Rotation.identity(), np.zeros(3)
    R1, p1 = Rotation.from_euler("x", 170, degrees=True), np.array([5.0, 0, 0])
    p, R, cut_t, cut_r = safety.limit_target_step(p0, R0, p1, R1)
    assert not cut_t and not cut_r and np.allclose(p, p1) and _angle_deg(R, R1) < 1e-9


def test_repeated_steps_reach_a_far_target_at_the_limited_rate():
    """A far target is approached at the limit, step by step, and then reached."""
    p, R = np.zeros(3), Rotation.identity()
    goal_p, goal_R = np.array([0.1, 0.0, 0.0]), Rotation.from_euler("z", 12, degrees=True)
    steps = 0
    while True:
        p, R, cut_t, cut_r = safety.limit_target_step(p, R, goal_p, goal_R, 0.03, 5.0)
        steps += 1
        if not (cut_t or cut_r):
            break
    assert steps == 4  # ceil(0.1 / 0.03) = 4, ceil(12 / 5) = 3
    assert np.allclose(p, goal_p) and _angle_deg(R, goal_R) < 1e-9


# ── the env wiring: the real methods from manipulator_env.py, on a fake env ──
# manipulator_env.py imports rclpy, so (as tests/test_home_gripper_order.py
# does) the methods are exec'd from the shipped source instead of imported.

import ast  # noqa: E402
import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ENV_SOURCE = REPO / "crisp_gym" / "envs" / "manipulator_env.py"


def _env_methods():
    tree = ast.parse(ENV_SOURCE.read_text())
    wanted = {"_limit_target_step", "clip_position_for_safety"}
    bodies = [
        fn
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ManipulatorBaseEnv"
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted
    ]
    assert {fn.name for fn in bodies} == wanted
    module = ast.Module(body=bodies, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "np": np,
        "Rotation": Rotation,
        "limit_target_step": safety.limit_target_step,
        "logger": logging.getLogger("test_safety_limits"),
    }
    exec(compile(module, str(ENV_SOURCE), "exec"), ns)  # noqa: S102
    return ns


def _fake_env(**config):
    defaults = dict(max_target_step_m=None, max_target_step_deg=None, safety_box=None)
    defaults.update(config)
    target = SimpleNamespace(position=np.array([0.4, 0.0, 0.3]), orientation=Rotation.identity())
    env = SimpleNamespace(
        config=SimpleNamespace(**defaults),
        robot=SimpleNamespace(target_pose=target),
        target_step_cuts=0,
    )
    return env


def test_env_limits_against_the_robots_current_target(caplog):
    m = _env_methods()
    env = _fake_env(max_target_step_m=0.03, max_target_step_deg=5.0)
    far = np.array([0.4, 0.5, 0.3])
    with caplog.at_level(logging.WARNING):
        p, R = m["_limit_target_step"](env, far, Rotation.from_euler("z", 90, degrees=True))
        m["_limit_target_step"](env, far, Rotation.identity())
    assert np.allclose(p, [0.4, 0.03, 0.3]) and np.isclose(_angle_deg(R, Rotation.identity()), 5.0)
    assert env.target_step_cuts == 2
    warnings = [r for r in caplog.records if "step limit acted" in r.getMessage()]
    assert len(warnings) == 1  # once per episode, not per step
    assert "500.0 mm and 90.0 deg" in warnings[0].getMessage()


def test_env_without_limits_does_nothing():
    m = _env_methods()
    env = _fake_env()
    far = np.array([5.0, 5.0, 5.0])
    p, R = m["_limit_target_step"](env, far, Rotation.from_euler("x", 120, degrees=True))
    assert np.allclose(p, far) and env.target_step_cuts == 0


def test_env_box_uses_zero_bounds():
    m = _env_methods()
    env = _fake_env(safety_box=safety.workspace_box(min_z=0.0))
    assert m["clip_position_for_safety"](env, np.array([0.4, 0.0, -0.2]))[2] == 0.0


def test_cartesian_step_limits_first_and_clips_into_the_box_last():
    """The box must hold even after the step limit moved the target."""
    tree = ast.parse(ENV_SOURCE.read_text())
    step = next(
        fn
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ManipulatorCartesianEnv"
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "step"
    )
    calls = [
        n.func.attr
        for n in ast.walk(step)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    order = [
        c for c in calls if c in ("_limit_target_step", "clip_position_for_safety", "set_target")
    ]
    lines = {
        n.func.attr: n.lineno
        for n in ast.walk(step)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in order
    }
    assert sorted(lines, key=lines.get) == [
        "_limit_target_step",
        "clip_position_for_safety",
        "set_target",
    ]
    assert calls.count("clip_position_for_safety") == 1  # one place, both branches
