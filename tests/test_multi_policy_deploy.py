"""One process, several policies: the per-episode switch and its bookkeeping.

Two deploy_policy.py processes cannot share a robot. crisp_py's Robot stores a
target on set_target() and republishes it from a timer at publish_frequency
(robot.py:173-194, _callback_publish_target_pose returns early only while no
target has EVER been set), so a second process keeps commanding its last target
whether or not its policy is stepping and the arm is pulled between the two.

One process with a list of policies has one env, one Robot, one publisher. The
switch is per EPISODE: the recording loop already re-requests the data_fn every
iteration, and on_end has homed the arm by then, so the handover never happens
mid-motion.

These exec the real code out of its source rather than importing it — the
modules pull rclpy/rich/numpy, which the test environment does not have.

Run: python -m pytest tests/test_multi_policy_deploy.py -v
"""

import argparse
import ast
import io
import sys
import threading
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANAGER = REPO / "crisp_gym" / "record" / "recording_manager.py"
EVALUATOR = REPO / "crisp_gym" / "record" / "evaluate.py"
DEPLOY = REPO / "crisp_gym" / "scripts" / "deploy_policy.py"


def _exec_members(source: Path, class_name: str, names: set, ns: dict) -> dict:
    """Exec the named methods/properties of a class as plain module members."""
    tree = ast.parse(source.read_text())
    bodies = [
        fn
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for fn in cls.body
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in names
    ]
    found = {fn.name for fn in bodies}
    assert found == names, f"expected {names} on {class_name}, found {found}"
    module = ast.Module(body=bodies, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), ns)  # noqa: S102
    return ns


# ── the policy index on RecordingManager ─────────────────────────────────────

class _FakeManager:
    """Duck-typed manager carrying only what the switch logic touches."""

    def __init__(self, policy_count=1, state="is_waiting"):
        self._state_lock = threading.Lock()
        self._state = state
        self.policy_count = policy_count
        self._active_policy = 0

    # the real class guards `state` with the same lock
    @property
    def state(self):
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, value):
        with self._state_lock:
            self._state = value


_NS = _exec_members(
    MANAGER,
    "RecordingManager",
    {"active_policy", "cycle_active_policy"},
    {"threading": threading},
)
_FakeManager.active_policy = _NS["active_policy"]
_FakeManager.cycle_active_policy = _NS["cycle_active_policy"]

_KEY_NS = _exec_members(MANAGER, "KeyboardRecordingManager", {"_handle_key"}, {})
_FakeManager._handle_key = _KEY_NS["_handle_key"]


def test_cycling_wraps_through_every_policy():
    m = _FakeManager(policy_count=3)
    assert m.active_policy == 0
    assert [m.cycle_active_policy() for _ in range(4)] == [1, 2, 0, 1]


def test_single_policy_can_never_leave_zero():
    """Every existing caller leaves policy_count at 1 and must be unaffected."""
    m = _FakeManager(policy_count=1)
    for _ in range(5):
        assert m.cycle_active_policy() == 0
    assert m.active_policy == 0


def test_p_cycles_only_between_episodes():
    """Mid-episode handover is a different, riskier feature: 'p' must do
    nothing while recording or paused, where the arm is not parked."""
    m = _FakeManager(policy_count=2, state="recording")
    m._handle_key("p")
    assert m.active_policy == 0, "switched while recording"

    m = _FakeManager(policy_count=2, state="paused")
    m._handle_key("p")
    assert m.active_policy == 0, "switched while paused"

    m = _FakeManager(policy_count=2, state="is_waiting")
    m._handle_key("p")
    assert m.active_policy == 1


def test_p_does_not_disturb_the_existing_keys():
    m = _FakeManager(policy_count=2, state="is_waiting")
    m._handle_key("p")
    assert m.state == "is_waiting", "cycling must not change the recording state"
    m._handle_key("r")
    assert m.state == "recording"


def test_the_banner_mentions_p_only_with_several_policies():
    src = MANAGER.read_text()
    assert "To cycle the active [b]P[/b]olicy" in src
    assert "self.policy_count > 1" in src, "the <p> line must be conditional"


# ── the evaluator's policy column and the two bugs it sat on ─────────────────

def _load_evaluator():
    ns = {
        "logging": __import__("logging"),
        "time": __import__("time"),
        "contextmanager": __import__("contextlib").contextmanager,
        "Path": Path,
        "prompt": None,  # replaced per test
        "logger": __import__("logging").getLogger("evaluator-test"),
    }
    tree = ast.parse(EVALUATOR.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Evaluator"
    )
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(EVALUATOR), "exec"), ns)  # noqa: S102
    return ns


def _evaluator(tmp_path, answers=("yes", "7")):
    ns = _load_evaluator()
    it = iter(answers)

    def fake_prompt(message="", options=None, default=None):  # noqa: ANN001
        return next(it)

    ns["prompt"] = fake_prompt
    return ns["Evaluator"](tmp_path / "eval" / "results.csv")


def test_a_fresh_file_gets_a_header():
    """The bug: the header was written only when the file ALREADY existed, so
    the first run opened a missing file in append mode and never wrote one."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ev = _evaluator(Path(d))
        with ev.start_eval(overwrite=False, activate=True):
            pass
        head = (Path(d) / "eval" / "results.csv").read_text().splitlines()[0]
        assert head == "episode,policy,success,score,time"


def test_the_policy_name_is_recorded():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ev = _evaluator(Path(d), answers=("yes", "7"))
        with ev.start_eval(activate=True):
            ev.start_timer()
            ev.evaluate(episode=3, policy="groot_se3_lerobot_policy")
        rows = (Path(d) / "eval" / "results.csv").read_text().splitlines()
        assert rows[0] == "episode,policy,success,score,time"
        cols = rows[1].split(",")
        assert cols[0] == "3"
        assert cols[1] == "groot_se3_lerobot_policy"
        assert cols[2] == "yes"
        assert cols[3] == "7"


def test_a_single_policy_run_leaves_the_column_empty():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ev = _evaluator(Path(d))
        with ev.start_eval(activate=True):
            ev.start_timer()
            ev.evaluate(episode=0)
        cols = (Path(d) / "eval" / "results.csv").read_text().splitlines()[1].split(",")
        assert cols[1] == "", f"expected an empty policy column, got {cols[1]!r}"


def test_no_timer_does_not_lose_the_users_answers():
    """The bug: the no-timer path set elapsed_time = None and the row was then
    formatted with :.2f, raising TypeError AFTER the user had already typed
    success and score."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ev = _evaluator(Path(d), answers=("no", "2"))
        with ev.start_eval(activate=True):
            ev.evaluate(episode=1, policy="p")  # start_timer() deliberately NOT called
        cols = (Path(d) / "eval" / "results.csv").read_text().splitlines()[1].split(",")
        assert cols[:4] == ["1", "p", "no", "2"]
        assert cols[4] == "", "a missing time should be blank, not a crash"


def test_deactivated_evaluation_writes_nothing():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ev = _evaluator(Path(d))
        with ev.start_eval(activate=False):
            ev.evaluate(episode=0, policy="p")
        assert not (Path(d) / "eval" / "results.csv").exists()


# ── the paired CLI flags ─────────────────────────────────────────────────────

def _deploy_parser():
    """Rebuild a parser from deploy_policy.py's own --path / --policy-config
    add_argument calls, so the flag shape is tested, not a copy of it."""
    tree = ast.parse(DEPLOY.read_text())
    wanted = {"--path", "--policy-config"}
    parser = argparse.ArgumentParser()
    seen = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        try:
            flag = ast.literal_eval(node.args[0])
        except Exception:  # noqa: BLE001
            continue
        if flag not in wanted:
            continue
        kwargs = {}
        for kw in node.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception:  # noqa: BLE001
                kwargs[kw.arg] = None
        kwargs.pop("type", None)
        kwargs.pop("help", None)
        parser.add_argument(flag, **kwargs)
        seen.add(flag)
    assert seen == wanted, f"could not rebuild {wanted - seen}"
    return parser


def test_both_flags_are_repeatable_and_land_in_lists():
    a = _deploy_parser().parse_args(
        ["--policy-config", "cfgA", "--path", "A",
         "--policy-config", "cfgB", "--path", "B"]
    )
    assert a.policy_configs == ["cfgA", "cfgB"]
    assert a.paths == ["A", "B"]


def test_one_pair_still_works_and_omitting_them_is_still_none():
    a = _deploy_parser().parse_args(["--policy-config", "cfg", "--path", "P"])
    assert a.policy_configs == ["cfg"] and a.paths == ["P"]
    b = _deploy_parser().parse_args([])
    assert b.policy_configs is None and b.paths is None, (
        "None is what the interactive prompt branches on"
    )


def test_a_count_mismatch_is_rejected():
    """zip() would silently drop the extra checkpoint, so the script must
    exit(2) instead."""
    src = DEPLOY.read_text()
    i = src.index("len(args.policy_configs) != len(args.paths)")
    tail = src[i : i + 500]
    assert "exit(2)" in tail, "a mismatch must be a usage error"
    assert "pair up in order" in tail, "the message must say how they pair"


def test_every_policy_is_shut_down_on_both_paths():
    """A leaked worker subprocess keeps a CUDA context and a pipe open."""
    tree = ast.parse(DEPLOY.read_text())
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "shutdown"
            for c in ast.walk(node)
        )
    ]
    assert len(loops) >= 2, (
        f"expected a shutdown loop on the normal AND the except path, found {len(loops)}"
    )


def test_the_data_fn_comes_from_the_active_policy():
    src = DEPLOY.read_text()
    assert 'data_fn=active["policy"].make_data_fn()' in src
    assert "idx = recording_manager.active_policy" in src
    assert "recording_manager.policy_count = len(policies)" in src


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} multi-policy deploy tests passed.")
