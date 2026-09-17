"""Ordering contract for ManipulatorBaseEnv.home(): arm first, gripper after.

Regression for a real hazard. `gripper.open()` is NON-BLOCKING -- it sets a
target that the 30 Hz publisher walks to in max_delta (0.1) steps, so the
gripper is fully open ~0.3 s later, while `robot.home` is still driving a
`time_to_home` (5 s default) trajectory. home() used to call `gripper.open()`
FIRST, so every homing motion released whatever was grasped over the task area
and carried the empty gripper to home. The two on_end hooks that drove
`robot.home(blocking=False)` and then opened had the same effect.

These tests exec the real `home` / `apply_home_gripper` source out of
manipulator_env.py rather than importing it, because the module pulls in rclpy.
That keeps the assertion on the shipped code: reorder the statements and these
fail.

Run: python -m pytest tests/test_home_gripper_order.py -v
"""

import ast
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crisp_gym.util.gripper_mode import GripperMode, HomeGripper  # noqa: E402

SOURCE = REPO / "crisp_gym" / "envs" / "manipulator_env.py"


def _load_methods() -> dict:
    """Exec ManipulatorBaseEnv.home / .apply_home_gripper as plain functions."""
    tree = ast.parse(SOURCE.read_text())
    wanted = {"home", "apply_home_gripper"}
    bodies = [
        fn
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ManipulatorBaseEnv"
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted
    ]
    assert {fn.name for fn in bodies} == wanted, (
        f"expected {wanted} on ManipulatorBaseEnv, found {[f.name for f in bodies]}"
    )
    module = ast.Module(body=bodies, type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {"GripperMode": GripperMode, "HomeGripper": HomeGripper}
    exec(compile(module, str(SOURCE), "exec"), ns)  # noqa: S102
    return ns


METHODS = _load_methods()


class _FakeEnv:
    """Duck-typed env recording the ORDER of the calls home() makes."""

    def __init__(self, home_gripper=HomeGripper.OPEN, gripper_mode=GripperMode.ABSOLUTE_CONTINUOUS):
        self.calls: list[str] = []
        self.config = types.SimpleNamespace(
            home_gripper=home_gripper, gripper_mode=gripper_mode
        )
        self.robot = types.SimpleNamespace(home=self._record("robot.home"))
        self.gripper = types.SimpleNamespace(
            open=self._record("gripper.open"), close=self._record("gripper.close")
        )

    def _record(self, name):
        def fn(*args, **kwargs):  # noqa: ANN002, ANN003
            self.calls.append(name)

        return fn

    def switch_to_default_controller(self):
        self.calls.append("switch_to_default_controller")

    def apply_home_gripper(self):
        return METHODS["apply_home_gripper"](self)

    def home(self, **kwargs):  # noqa: ANN003
        return METHODS["home"](self, **kwargs)


# ── the regression ────────────────────────────────────────────────────────────


def test_gripper_is_commanded_after_the_arm_not_before():
    """THE test: any gripper call must come after robot.home, never before."""
    env = _FakeEnv()
    env.home(blocking=True)

    assert "robot.home" in env.calls
    arm = env.calls.index("robot.home")
    gripper = [i for i, c in enumerate(env.calls) if c.startswith("gripper.")]
    assert gripper, f"expected a gripper call, got {env.calls}"
    assert min(gripper) > arm, (
        f"gripper commanded before the arm finished homing: {env.calls}"
    )


def test_blocking_home_opens_by_default():
    env = _FakeEnv()
    env.home(blocking=True)
    assert env.calls == ["robot.home", "gripper.open"]


# ── home_gripper dispatch ─────────────────────────────────────────────────────


def test_closed_closes_at_home():
    env = _FakeEnv(home_gripper=HomeGripper.CLOSED)
    env.home(blocking=True)
    assert env.calls == ["robot.home", "gripper.close"]


def test_hold_leaves_the_gripper_alone():
    env = _FakeEnv(home_gripper=HomeGripper.HOLD)
    env.home(blocking=True)
    assert env.calls == ["robot.home"]


@pytest.mark.parametrize("home_gripper", list(HomeGripper))
def test_gripper_mode_none_never_touches_the_gripper(home_gripper):
    """A NONE-mode env has no gripper to command, whatever home_gripper says."""
    env = _FakeEnv(home_gripper=home_gripper, gripper_mode=GripperMode.NONE)
    env.home(blocking=True)
    assert env.calls == ["robot.home"]


def test_unknown_home_gripper_raises():
    env = _FakeEnv(home_gripper="sideways")
    with pytest.raises(ValueError, match="home_gripper"):
        env.home(blocking=True)


# ── non-blocking ──────────────────────────────────────────────────────────────


def test_non_blocking_home_does_not_touch_the_gripper():
    """blocking=False returns mid-trajectory, so there is no safe moment to act."""
    env = _FakeEnv()
    env.home(blocking=False)
    assert env.calls == ["robot.home", "switch_to_default_controller"]
    assert not [c for c in env.calls if c.startswith("gripper.")]


# ── the enum itself ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [("open", HomeGripper.OPEN), ("closed", HomeGripper.CLOSED), ("hold", HomeGripper.HOLD)],
)
def test_yaml_strings_coerce(text, expected):
    """__post_init__ coerces a YAML string through HomeGripper(...)."""
    assert HomeGripper(text) is expected


def test_default_is_open_so_existing_setups_keep_their_end_state():
    """The fix changes WHEN the gripper opens, not the state homing leaves."""
    src = (REPO / "crisp_gym" / "envs" / "manipulator_env_config.py").read_text()
    assert "home_gripper: HomeGripper | str = HomeGripper.OPEN" in src


# ── call sites ────────────────────────────────────────────────────────────────


def _calls(src: str) -> list[ast.Call]:
    """Every Call node in a script (AST, so comments and docstrings cannot match)."""
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]


def _dotted(node: ast.AST) -> str:
    """'env.gripper.open' for an attribute chain, '' for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.parametrize(
    "script",
    ["scripts/record_lerobot_format_leader_follower.py", "scripts/deploy_policy.py"],
)
def test_on_end_hooks_no_longer_open_after_a_non_blocking_home(script):
    """Both hooks drove robot.home(blocking=False) and then opened immediately."""
    calls = _calls((REPO / "crisp_gym" / script).read_text())

    assert not [c for c in calls if _dotted(c.func) == "env.gripper.open"], (
        f"{script} still opens the gripper directly; it should go through env.home()"
    )

    non_blocking_env_homes = [
        c
        for c in calls
        if _dotted(c.func) == "env.robot.home"
        and any(
            kw.arg == "blocking" and getattr(kw.value, "value", None) is False
            for kw in c.keywords
        )
    ]
    assert not non_blocking_env_homes, (
        f"{script} still homes the follower non-blocking; the gripper would settle "
        "while the arm is still travelling"
    )


@pytest.mark.parametrize(
    "script",
    ["scripts/record_lerobot_format_leader_follower.py", "scripts/deploy_policy.py"],
)
def test_on_end_hooks_home_through_the_env(script):
    """env.home() is what applies home_gripper; robot.home() bypasses it."""
    calls = _calls((REPO / "crisp_gym" / script).read_text())
    assert [c for c in calls if _dotted(c.func) == "env.home"], (
        f"{script} never calls env.home(), so config.home_gripper is ignored there"
    )
