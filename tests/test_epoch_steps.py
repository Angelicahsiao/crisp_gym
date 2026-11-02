"""Tests for epoch-based scheduling (crisp_gym/util/epoch_steps.py).

The behaviour under test is arithmetic plus in-place config mutation, so these
run without lerobot, torch or a dataset: `cfg` is a plain namespace carrying the
fields `resolve_epoch_steps` reads, and the dataset is anything with `__len__`.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from crisp_gym.util.epoch_steps import (
    EXTRA_CHECKPOINTS,
    checkpoint_schedule,
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


def _cfg(steps=100_000, batch_size=64, save_freq=20_000, **extra):
    """lerobot TrainPipelineConfig defaults: steps 100k, save_freq 20k."""
    return SimpleNamespace(steps=steps, batch_size=batch_size, save_freq=save_freq, **extra)


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


# ── checkpoint_schedule (pure) ────────────────────────────────────────────────

def test_checkpoint_schedule_exact_division():
    save, target, total, n = checkpoint_schedule(16_875, 5, 2)
    assert (save, target, total, n) == (3_375, 16_875, 23_625, 5)


def test_checkpoint_schedule_rounds_target_up_by_less_than_n():
    save, target, total, n = checkpoint_schedule(16_876, 5, 2)
    assert target % save == 0                 # lands ON a checkpoint
    assert 16_876 <= target < 16_876 + n      # covers the epochs; overshoot < n
    assert total == target + 2 * save


def test_checkpoint_schedule_caps_checkpoints_at_target_steps():
    # 2 steps cannot hold 5 checkpoints: save_freq floors at 1.
    save, target, total, n = checkpoint_schedule(2, 5, 2)
    assert (save, target, total, n) == (1, 2, 4, 2)


def test_checkpoint_schedule_zero_extra_is_target_only():
    save, target, total, _ = checkpoint_schedule(16_875, 5, 0)
    assert total == target == 16_875


@pytest.mark.parametrize("bad", [(0, 5, 2), (-1, 5, 2), (100, 5, -1)])
def test_checkpoint_schedule_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        checkpoint_schedule(*bad)


# ── resolve_epoch_steps: without --epochs ─────────────────────────────────────

def test_resolve_without_epochs_changes_nothing_but_reports():
    cfg = _cfg(steps=100_000, batch_size=64, save_freq=10_000, eval_freq=5_000)
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), None)

    assert cfg.steps == 100_000          # untouched
    assert cfg.save_freq == 10_000       # untouched
    assert cfg.eval_freq == 5_000        # untouched
    assert plan.changed is False
    assert plan.rescaled == {}
    assert plan.extra_checkpoints == 0   # no epoch target to extend past
    assert plan.checkpoint_steps == []
    # ...but the epoch equivalent is still computed and reported.
    assert plan.epochs_after == pytest.approx(711.1, abs=0.1)


def test_resolve_without_epochs_tolerates_missing_save_freq():
    # Reporting only: nothing is rescaled, so save_freq is not required.
    plan = resolve_epoch_steps(_cfg(save_freq=None), _Dataset(9_000), None)
    assert plan.save_freq is None


# ── resolve_epoch_steps: the checkpoint schedule ──────────────────────────────

def test_worked_example_lands_on_target_then_two_more_intervals():
    """The example from the module docstring / the agreed design."""
    cfg = _cfg()                                      # 100k steps, save 20k -> 5 ckpts
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 120)

    assert plan.target_steps_min == 16_875            # ceil(120 x 9000 / 64)
    assert cfg.save_freq == 3_375                     # 24 epochs per checkpoint
    assert plan.target_step == 16_875                 # a checkpoint ON the target
    assert cfg.steps == 23_625                        # target + 2 x 3,375
    assert plan.checkpoint_steps == [3_375 * i for i in range(1, 8)]
    assert plan.epochs_target == pytest.approx(120.0)
    assert plan.epochs_after == pytest.approx(168.0)


def test_two_more_checkpoints_than_the_incoming_schedule():
    for steps, save in [(100_000, 20_000), (100_000, 10_000), (50_000, 5_000), (30_000, 30_000)]:
        cfg = _cfg(steps=steps, save_freq=save)
        plan = resolve_epoch_steps(cfg, _Dataset(9_000), 120)
        assert len(plan.checkpoint_steps) == round(steps / save) + EXTRA_CHECKPOINTS


def test_exactly_two_checkpoints_land_after_the_target():
    for epochs, n, batch in [(120, 9_000, 64), (284.4, 22_500, 32), (1, 100, 64), (37.3, 5_000, 16)]:
        cfg = _cfg(batch_size=batch)
        plan = resolve_epoch_steps(cfg, _Dataset(n), epochs)
        after = [s for s in plan.checkpoint_steps if s > plan.target_step]
        assert len(after) == EXTRA_CHECKPOINTS == 2
        assert plan.target_step in plan.checkpoint_steps
        assert cfg.steps == plan.target_step + 2 * cfg.save_freq
        assert cfg.steps % cfg.save_freq == 0          # final step is a checkpoint
        # The requested epochs are always covered by the target checkpoint.
        assert plan.epochs_target >= epochs


def test_target_rounds_up_when_not_divisible():
    # 9,001 samples -> ceil(120 x 9001 / 64) = 16,877, not a multiple of 5.
    cfg = _cfg()
    plan = resolve_epoch_steps(cfg, _Dataset(9_001), 120)
    assert plan.target_steps_min == 16_877
    assert plan.target_step % cfg.save_freq == 0
    assert 16_877 <= plan.target_step < 16_877 + 5
    assert "rounded up from 16877" in plan.describe()


def test_resolve_accounts_for_gradient_accumulation():
    # Effective batch 128 => half the steps per epoch, so half the save_freq.
    cfg_plain = _cfg()
    cfg_accum = _cfg(gradient_accumulation_steps=2)
    p_plain = resolve_epoch_steps(cfg_plain, _Dataset(9_000), 100)
    p_accum = resolve_epoch_steps(cfg_accum, _Dataset(9_000), 100)

    assert p_accum.target_steps_min == steps_for_epochs(100, 128, 9_000)
    assert p_accum.target_steps_min == math.ceil(p_plain.target_steps_min / 2)
    # Same checkpoint count and same epochs per checkpoint either way.
    assert len(p_accum.checkpoint_steps) == len(p_plain.checkpoint_steps)
    assert p_accum.epochs_target == pytest.approx(p_plain.epochs_target, abs=0.05)


def test_eval_follows_save_and_log_is_untouched():
    cfg = _cfg(save_freq=20_000, eval_freq=20_000, log_freq=200)
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 120)

    assert cfg.eval_freq == cfg.save_freq == 3_375   # evals still align with ckpts
    assert cfg.log_freq == 200                       # deliberately untouched
    assert set(plan.rescaled) == {"save_freq", "eval_freq"}


def test_eval_scales_by_the_save_factor():
    cfg = _cfg(save_freq=20_000, eval_freq=10_000)    # two evals per checkpoint
    resolve_epoch_steps(cfg, _Dataset(9_000), 120)
    assert cfg.eval_freq == round(10_000 * 3_375 / 20_000)


def test_disabled_eval_stays_disabled():
    cfg = _cfg(eval_freq=0)                           # 0 = disabled
    plan = resolve_epoch_steps(cfg, _Dataset(9_000), 100)
    assert cfg.eval_freq == 0                         # not scaled to 1
    assert "eval_freq" not in plan.rescaled


def test_rescaled_freqs_stay_within_bounds():
    # A huge reduction must not give a 0 freq (division by zero in lerobot) nor
    # a freq larger than the run itself.
    cfg = _cfg(steps=100_000, save_freq=90_000, eval_freq=1)
    resolve_epoch_steps(cfg, _Dataset(100), 1)
    assert cfg.steps >= 1
    assert 1 <= cfg.save_freq <= cfg.steps
    assert 1 <= cfg.eval_freq <= cfg.steps


# ── resolve_epoch_steps: refusals ─────────────────────────────────────────────

def test_resolve_raises_on_empty_dataset():
    with pytest.raises(RuntimeError, match="0 samples"):
        resolve_epoch_steps(_cfg(), _Dataset(0), 120)


@pytest.mark.parametrize("bad_steps", [None, 0, -1])
def test_resolve_raises_when_steps_field_is_unusable(bad_steps):
    with pytest.raises(RuntimeError, match="steps"):
        resolve_epoch_steps(_cfg(steps=bad_steps), _Dataset(9_000), 120)


@pytest.mark.parametrize("bad_save", [None, 0, -1, "20000"])
def test_epochs_needs_a_usable_save_freq(bad_save):
    # The extra checkpoints cannot exist without a checkpoint interval.
    with pytest.raises(RuntimeError, match="save_freq"):
        resolve_epoch_steps(_cfg(save_freq=bad_save), _Dataset(9_000), 120)


def test_epochs_refuses_when_checkpointing_is_disabled():
    with pytest.raises(RuntimeError, match="save_checkpoint"):
        resolve_epoch_steps(_cfg(save_checkpoint=False), _Dataset(9_000), 120)


# ── provenance + logging ──────────────────────────────────────────────────────

def test_plan_as_dict_is_json_serializable_and_records_the_schedule():
    cfg = _cfg(save_freq=20_000)
    d = resolve_epoch_steps(cfg, _Dataset(9_000), 120).as_dict()

    json.dumps(d)   # must not raise — it is stamped into pose_repr.json
    assert d["epochs_requested"] == 120
    assert d["steps"] == cfg.steps == 23_625
    assert d["target_step"] == 16_875
    assert d["epochs_target"] == pytest.approx(120.0)
    assert d["epochs"] == pytest.approx(168.0)
    assert d["save_freq"] == 3_375
    assert d["n_target_checkpoints"] == 5
    assert d["extra_checkpoints"] == EXTRA_CHECKPOINTS
    assert d["checkpoint_steps"] == [3_375 * i for i in range(1, 8)]
    assert d["num_samples"] == 9_000
    assert d["effective_batch"] == 64
    assert d["rescaled_intervals"]["save_freq"] == {"from": 20_000, "to": 3_375}


def test_describe_shows_target_and_extra_checkpoints():
    text = resolve_epoch_steps(_cfg(), _Dataset(9_000), 120).describe()
    assert "--epochs 120" in text
    assert "ceil(120 x 9000 / 64) = 16875" in text
    assert "+ 2 extra checkpoint intervals -> steps = 23625" in text
    assert text.count("<- epoch target") == 1
    assert text.count("<- extra") == 2


def test_describe_elides_long_tables_but_keeps_target_and_extras():
    # 100 checkpoints up to the target -> 102 rows; the table must elide.
    cfg = _cfg(steps=100_000, save_freq=1_000)
    text = resolve_epoch_steps(cfg, _Dataset(9_000), 120).describe()
    assert "more)" in text
    assert text.count("<- epoch target") == 1
    assert text.count("<- extra") == 2


def test_describe_without_epochs_explains_the_comparison_hazard():
    text = resolve_epoch_steps(_cfg(), _Dataset(9_000), None).describe()
    assert "--epochs NOT given" in text
    assert "epochs" in text
