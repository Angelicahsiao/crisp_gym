"""Tests for epoch-based scheduling (crisp_gym/util/epoch_steps.py).

The behaviour under test is arithmetic plus in-place config mutation, so these
run without lerobot, torch or a dataset: `cfg` is a plain namespace carrying the
fields `resolve_epoch_steps` reads, and the dataset is anything with `__len__`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from crisp_gym.util.epoch_steps import (
    EXTRA_STEPS,
    effective_batch_size,
    epochs_for_steps,
    pop_epochs_flag,
    resolve_epoch_steps,
    steps_for_epochs,
)


class _Dataset:
    """Minimal stand-in: only `len()` is consumed."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


def _cfg(steps=100_000, batch_size=64, **extra):
    return SimpleNamespace(steps=steps, batch_size=batch_size, **extra)


# ── argv flag popping ─────────────────────────────────────────────────────────

def test_pop_epochs_absent_leaves_argv_untouched():
    argv = ["--policy.type=diffusion", "--steps=1000"]
    epochs, remaining = pop_epochs_flag(argv)
    assert epochs is None
    assert remaining == argv


@pytest.mark.parametrize(
    "argv",
    [
        ["--epochs", "120"],
        ["--epochs=120"],
    ],
)
def test_pop_epochs_both_spellings(argv):
    epochs, remaining = pop_epochs_flag([*argv, "--batch_size=64"])
    assert epochs == 120.0
    # The flag AND its value must be gone, or draccus raises on them.
    assert remaining == ["--batch_size=64"]


def test_pop_epochs_preserves_order_of_other_args():
    epochs, remaining = pop_epochs_flag(["--a=1", "--epochs", "10", "--b=2", "--c=3"])
    assert epochs == 10.0
    assert remaining == ["--a=1", "--b=2", "--c=3"]


@pytest.mark.parametrize("bad", [["--epochs"], ["--epochs", "abc"], ["--epochs=0"], ["--epochs=-5"]])
def test_pop_epochs_rejects_bad_values(bad):
    # Loud failure matters: a silently-dropped --epochs trains on the default
    # step schedule and looks entirely normal.
    with pytest.raises(ValueError):
        pop_epochs_flag(bad)


# ── arithmetic ────────────────────────────────────────────────────────────────

def test_epochs_for_steps_matches_the_handheld_vs_robot_example():
    """The motivating case from the module docstring."""
    # 50 episodes @ 15 Hz, batch 64, 100k steps.
    handheld = epochs_for_steps(100_000, 64, 50 * 12 * 15)   # 9,000 samples
    robot = epochs_for_steps(100_000, 64, 50 * 30 * 15)      # 22,500 samples
    assert handheld == pytest.approx(711.1, abs=0.1)
    assert robot == pytest.approx(284.4, abs=0.1)
    # The handheld run is 2.5x longer in epochs at identical --steps.
    assert handheld / robot == pytest.approx(2.5, abs=0.01)


def test_steps_for_epochs_is_inverse_of_epochs_for_steps():
    steps = steps_for_epochs(284.4, 64, 9_000)
    assert epochs_for_steps(steps, 64, 9_000) == pytest.approx(284.4, abs=0.01)


def test_steps_for_epochs_rounds_up_and_floors_at_one():
    # Rounds up so the requested epochs are fully covered, not truncated.
    assert steps_for_epochs(1, 64, 100) == math.ceil(100 / 64) == 2
    # A sub-step request still runs one step rather than zero.
    assert steps_for_epochs(0.001, 64, 100) == 1


def test_epochs_for_steps_empty_dataset_is_zero_not_a_crash():
    assert epochs_for_steps(1000, 64, 0) == 0.0


def test_steps_for_epochs_rejects_empty_dataset():
    with pytest.raises(ValueError):
        steps_for_epochs(10, 64, 0)


# ── effective batch size ──────────────────────────────────────────────────────

def test_effective_batch_defaults_grad_accum_to_one():
    assert effective_batch_size(_cfg(batch_size=64)) == (64, 1)


@pytest.mark.parametrize(
    "field",
    ["gradient_accumulation_steps", "grad_accum_steps", "gradient_accumulate_every"],
)
def test_effective_batch_reads_each_grad_accum_spelling(field):
    assert effective_batch_size(_cfg(batch_size=32, **{field: 4})) == (32, 4)


@pytest.mark.parametrize("bad", [None, 0, -1, "64"])
def test_effective_batch_raises_on_unusable_batch_size(bad):
    # Guessing a batch size would silently produce a wrong step count.
    with pytest.raises(RuntimeError, match="batch_size"):
        effective_batch_size(_cfg(batch_size=bad))


# ── resolve_epoch_steps ───────────────────────────────────────────────────────

def test_resolve_without_epochs_changes_nothing_but_reports():
    cfg = _cfg(steps=100_000, batch_size=64, save_freq=10_000, eval_freq=5_000)
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), None)

    assert cfg.steps == 100_000          # untouched
    assert cfg.save_freq == 10_000       # untouched
    assert cfg.eval_freq == 5_000        # untouched
    assert plan.changed is False
    assert plan.rescaled == {}
    assert plan.extra_steps == 0         # no epoch-derived count to add to
    # ...but the epoch equivalent is still computed and reported.
    assert plan.epochs_after == pytest.approx(711.1, abs=0.1)


def test_resolve_with_epochs_sets_steps():
    cfg = _cfg(steps=100_000, batch_size=64)
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 284.4)

    assert cfg.steps == plan.steps_after == steps_for_epochs(284.4, 64, 9_000) + EXTRA_STEPS
    assert cfg.steps == pytest.approx(40_000, abs=50)   # ~40k, per the analysis
    assert plan.changed is True
    # The extra steps push the epoch count fractionally past the request.
    assert plan.epochs_after == pytest.approx(epochs_for_steps(cfg.steps, 64, 9_000))
    assert plan.epochs_after > 284.4
    assert plan.steps_before == 100_000


def test_epoch_derived_steps_get_exactly_two_extra():
    """The run is scheduled EXTRA_STEPS (2) past the epoch-derived count."""
    assert EXTRA_STEPS == 2
    for epochs, n, batch in [(120, 9_000, 64), (1, 100, 64), (0.001, 100, 64), (284.4, 22_500, 32)]:
        cfg = _cfg(steps=100_000, batch_size=batch)
        plan = resolve_epoch_steps(cfg, _Dataset(n), epochs)
        assert cfg.steps == steps_for_epochs(epochs, batch, n) + 2
        assert plan.extra_steps == 2


def test_extra_steps_also_apply_with_gradient_accumulation():
    cfg = _cfg(steps=100_000, batch_size=64, gradient_accumulation_steps=2)
    resolve_epoch_steps(cfg, _Dataset(9_000), 100)
    # +2 OPTIMIZER steps on top of the effective-batch (128) count.
    assert cfg.steps == steps_for_epochs(100, 128, 9_000) + 2


def test_resolve_accounts_for_gradient_accumulation():
    # Effective batch 128 => half the optimizer steps for the same epochs
    # (compared on the epoch-derived part, before the fixed extra steps).
    cfg_plain = _cfg(steps=1000, batch_size=64)
    cfg_accum = _cfg(steps=1000, batch_size=64, gradient_accumulation_steps=2)

    resolve_epoch_steps(cfg_plain, _Dataset(9_000), 100)
    resolve_epoch_steps(cfg_accum, _Dataset(9_000), 100)

    assert cfg_accum.steps - EXTRA_STEPS == math.ceil((cfg_plain.steps - EXTRA_STEPS) / 2)


def test_resolve_rescales_save_and_eval_but_not_log():
    """The answered design decision: save/eval scale, log_freq stays fixed."""
    cfg = _cfg(steps=100_000, batch_size=64, save_freq=10_000, eval_freq=20_000, log_freq=200)
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 284.4)

    ratio = cfg.steps / 100_000
    assert cfg.save_freq == round(10_000 * ratio)
    assert cfg.eval_freq == round(20_000 * ratio)
    assert cfg.log_freq == 200                      # deliberately untouched
    assert set(plan.rescaled) == {"save_freq", "eval_freq"}


def test_rescaling_preserves_the_number_of_checkpoints():
    """The point of rescaling: a shorter run still saves as many checkpoints."""
    cfg = _cfg(steps=100_000, batch_size=64, save_freq=10_000)
    before = 100_000 / 10_000               # 10 checkpoints
    resolve_epoch_steps(cfg, _Dataset(9_000), 284.4)
    after = cfg.steps / cfg.save_freq
    assert after == pytest.approx(before, rel=0.05)


def test_rescaled_freq_never_drops_below_one_or_exceeds_total_steps():
    # A huge ratio reduction must not yield save_freq=0 (division by zero in
    # lerobot) nor a freq larger than the run itself (no checkpoint at all).
    cfg = _cfg(steps=100_000, batch_size=64, save_freq=90_000, eval_freq=1)
    resolve_epoch_steps(cfg, _Dataset(100), 1)
    assert cfg.steps >= 1
    assert 1 <= cfg.save_freq <= cfg.steps
    assert 1 <= cfg.eval_freq <= cfg.steps


def test_resolve_ignores_absent_and_disabled_freq_fields():
    cfg = _cfg(steps=100_000, batch_size=64, eval_freq=0)   # 0 = disabled
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 100)
    assert cfg.eval_freq == 0            # left disabled, not scaled to 1
    assert plan.rescaled == {}


def test_resolve_raises_on_empty_dataset():
    with pytest.raises(RuntimeError, match="0 samples"):
        resolve_epoch_steps(_cfg(), _Dataset(0), 120)


@pytest.mark.parametrize("bad_steps", [None, 0, -1])
def test_resolve_raises_when_steps_field_is_unusable(bad_steps):
    with pytest.raises(RuntimeError, match="steps"):
        resolve_epoch_steps(_cfg(steps=bad_steps), _Dataset(9_000), 120)


# ── provenance ────────────────────────────────────────────────────────────────

def test_plan_as_dict_is_json_serializable_and_records_the_epochs():
    import json

    cfg = _cfg(steps=100_000, batch_size=64, save_freq=10_000)
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 120)
    d = plan.as_dict()

    json.dumps(d)   # must not raise — it is stamped into pose_repr.json
    assert d["epochs_requested"] == 120
    # Recorded epochs reflect the FINAL step total, extra steps included.
    assert d["epochs"] == pytest.approx(cfg.steps * 64 / 9_000, abs=0.001)
    assert d["epochs"] == pytest.approx(120, abs=0.05)
    assert d["num_samples"] == 9_000
    assert d["effective_batch"] == 64
    assert d["steps"] == cfg.steps
    assert d["extra_steps"] == EXTRA_STEPS
    assert d["rescaled_intervals"]["save_freq"]["from"] == 10_000


def test_describe_mentions_the_arithmetic():
    cfg = _cfg(steps=100_000, batch_size=64)
    text = resolve_epoch_steps(cfg, _Dataset(9_000), 120).describe()
    assert "--epochs 120" in text
    assert "9000" in text and "64" in text
    # The extra steps are visible in the logged arithmetic, not hidden.
    assert f"+ {EXTRA_STEPS} = {cfg.steps}" in text


def test_describe_without_epochs_explains_the_comparison_hazard():
    text = resolve_epoch_steps(_cfg(), _Dataset(9_000), None).describe()
    assert "--epochs NOT given" in text
    assert "epochs" in text
