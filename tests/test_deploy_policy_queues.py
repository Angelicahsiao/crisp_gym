"""Observation-queue contract for the local deployment worker.

`inference_worker` rebuilds the observation window every request by pushing
each frame through lerobot's `populate_queues`, then predicting on the last
batch. That is a diffusion/ACT/VQ-BeT shape: those policies keep `_queues`,
and `predict_action_chunk` stacks them into the n_obs_steps dimension.

GR00T keeps no observation queue. Its `reset()` builds an ACTION queue only
(`self._action_queue`), and `predict_action_chunk` consumes the batch it is
handed. Touching `policy._queues` therefore hit `nn.Module.__getattr__` and
every rollout died on the first frame with

    AttributeError: 'GrootPolicy' object has no attribute '_queues'

The push is now guarded. With n_obs_steps=1 the window is a single frame, so
the batch that would have been queued is exactly the batch predicted on.

The guarded statement lives deep inside the worker's request loop, so these
assertions are structural (on the shipped AST) plus a check that `hasattr` is
the right discriminator against an nn.Module-style `__getattr__`.

Run: python -m pytest tests/test_deploy_policy_queues.py -v
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SOURCE = REPO / "crisp_gym" / "policy" / "relative_lerobot_policy.py"


def _worker() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text())
    return next(
        fn for fn in tree.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "inference_worker"
    )


def _populate_calls(node: ast.AST) -> list[ast.Call]:
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "populate_queues"
    ]


# ── the regression ────────────────────────────────────────────────────────────

def test_populate_queues_is_guarded():
    """An unguarded push is the GR00T crash."""
    worker = _worker()
    calls = _populate_calls(worker)
    assert calls, "populate_queues vanished from inference_worker"
    guarded = [
        node for node in ast.walk(worker)
        if isinstance(node, ast.If) and _populate_calls(node.test) == []
        and _populate_calls(ast.Module(body=node.body, type_ignores=[]))
    ]
    assert guarded, (
        "populate_queues is not inside an `if` — a policy without observation "
        "queues (GR00T) dies on the first frame"
    )
    tests = [ast.dump(node.test) for node in guarded]
    assert any("_queues" in t and "hasattr" in t for t in tests), (
        f"the guard does not test hasattr(policy, '_queues'): {tests}"
    )


def test_prediction_is_not_inside_the_queue_guard():
    """Skipping the queue push must still reach predict_action_chunk --
    otherwise GR00T would return no chunk instead of crashing."""
    worker = _worker()
    guards = [
        node for node in ast.walk(worker)
        if isinstance(node, ast.If)
        and "_queues" in ast.dump(node.test)
        and "hasattr" in ast.dump(node.test)
    ]
    assert guards, "no hasattr(_queues) guard found"
    for guard in guards:
        inside = [
            n for n in ast.walk(ast.Module(body=guard.body, type_ignores=[]))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "predict_action_chunk"
        ]
        assert not inside, "predict_action_chunk must run for queueless policies too"
    called = {
        n.func.attr for n in ast.walk(worker)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "predict_action_chunk" in called


# ── hasattr really is the right discriminator ─────────────────────────────────

class _ModuleLike:
    """torch.nn.Module raises AttributeError from __getattr__ for an unknown
    name (modeling: torch/nn/modules/module.py). hasattr must read that as
    False rather than propagating -- the whole guard rests on it.
    """

    def __getattr__(self, name):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class _QueuePolicy(_ModuleLike):
    def __init__(self):
        self._queues = {"observation.state": []}


def test_hasattr_is_false_for_a_module_without_queues():
    assert not hasattr(_ModuleLike(), "_queues")


def test_hasattr_is_true_for_an_observation_queue_policy():
    assert hasattr(_QueuePolicy(), "_queues")


def test_a_non_attribute_error_still_propagates():
    """Documents the limit: hasattr only swallows AttributeError, so a policy
    whose __getattr__ fails some other way is not silently treated as
    queueless."""

    class _Angry:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    try:
        hasattr(_Angry(), "_queues")
    except RuntimeError:
        return
    raise AssertionError("expected the RuntimeError to propagate")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} deploy queue-compat tests passed.")
