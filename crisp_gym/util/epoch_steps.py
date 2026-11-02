"""Express a training run's length in EPOCHS instead of optimizer steps.

Why this exists
---------------
lerobot schedules training by optimizer STEPS (``--steps``). That makes two runs
on datasets of different size incomparable, because a step covers a fixed number
of SAMPLES, not a fixed fraction of the dataset::

    epochs = steps * effective_batch / num_samples

A dataset of short episodes has fewer frames, so at a fixed step count it is
traversed MORE times. Concretely, 50 episodes at 15 Hz, batch 64, 100k steps:

    handheld, 12 s/ep -> 180 frames/ep ->  9,000 samples -> 711 epochs
    robot,    30 s/ep -> 450 frames/ep -> 22,500 samples -> 284 epochs

Same ``--steps``, same batch size, and yet the handheld model was trained 2.5x
longer in the only unit that governs memorization. Comparing their losses is
meaningless and the shorter-episode model overfits first. Every diffusion-policy
codebase upstream of this one therefore schedules in epochs, not steps (see the
reference table below).

This module converts an epoch target into the step count lerobot wants and lays
the checkpoints out around it (see "Checkpoint schedule" below). `eval_freq` is
rescaled by the same factor as `save_freq`. `log_freq` is deliberately NOT
rescaled: it controls console/wandb density, which should stay constant so logs
remain readable across runs.

Reference epoch counts (read from upstream configs, not from memory)
--------------------------------------------------------------------
From ``universal_manipulation_interface/diffusion_policy/config/``:

    ============================================  ==========
    config                                        num_epochs
    ============================================  ==========
    train_diffusion_unet_timm_umi_workspace          120   <- UMI handheld, UNet
    train_diffusion_transformer_umi_workspace        200   <- UMI handheld, Transformer
    train_diffusion_unet_umi_bimanual_workspace      100
    train_diffusion_transformer_umi_bimanual_...     100
    train_diffusion_unet_timm_robomimic_workspace    600   <- sim, small datasets
    train_diffusion_unet_image_workspace            8000   <- sim toy tasks (pusht)
    ============================================  ==========

Reading of that table:

* The TRANSFORMER denoiser gets ~1.7x the epochs of the UNet (200 vs 120) on the
  same data. It has weaker inductive bias, so it needs more passes — and it also
  overfits harder on small data. On a few thousand samples prefer the UNet.
* Real-robot/handheld configs sit at 100-200. The four-digit numbers are
  simulation tasks with tiny datasets where an epoch costs almost nothing; do
  NOT copy them for real data.
* Bimanual drops to 100 — more data per episode, so fewer passes.

So for a crisp_gym UNet (lerobot's `diffusion` policy is the conv-UNet family),
``--epochs 120`` is the grounded default. Use 200 only if you switch to a
transformer denoiser. The `MirrorDuo` inverse-scaling rule (``epochs =
50000 / num_demos``) is a different convention with different augmentation and
is not directly comparable.

To make the handheld-vs-robot A/B fair, compute the robot run's epochs from its
own numbers and pass that to the handheld run::

    epochs_robot = steps_robot * batch / num_samples_robot
    python3 ... lerobot_relative_pose.py --epochs <epochs_robot> ...

Checkpoint schedule
-------------------
With ``--epochs``, training runs ``EXTRA_CHECKPOINTS`` (2) checkpoint intervals
PAST the epoch target, with one checkpoint landing exactly on the target:

1. ``target_steps_min = ceil(epochs * num_samples / effective_batch)``.
2. ``n = round(steps_before / save_freq_before)`` — the number of checkpoints
   the incoming schedule would have produced (lerobot default 100k / 20k = 5).
3. ``save_freq`` is capped at ``MAX_SAVE_FREQ`` (20,000): if
   ``ceil(target_steps_min / n)`` would exceed it, ``n`` is raised to
   ``ceil(target_steps_min / MAX_SAVE_FREQ)``. The cap adds checkpoints; it
   never moves the target or drops a checkpoint.
4. ``save_freq = ceil(target_steps_min / n)``; the epoch-target checkpoint is
   at ``target_step = n * save_freq``. That rounds the target UP by fewer than
   ``n`` steps when ``target_steps_min`` is not a multiple of ``n``, so the
   requested epochs are always fully covered.
5. ``steps = target_step + EXTRA_CHECKPOINTS * save_freq`` — so the run saves
   ``n + 2`` checkpoints: ``n`` up to the epoch target, then 2 beyond it.

Example: 9,000 samples, batch 64, ``--epochs 120``, lerobot defaults::

    target 16,875 steps (120 ep); save every 3,375 steps (24 ep)
    checkpoints at 3,375 ... 16,875 (target) | 20,250, 23,625 (extra)
    steps = 23,625 (168 ep)

Example where the cap applies: 60,000 samples, batch 64, ``--epochs 120``::

    target 112,500 steps; 5 checkpoints would be every 22,500 (> 20,000)
    -> 6 checkpoints, every 18,750 steps (20 ep)
    checkpoints at 18,750 ... 112,500 (target) | 131,250, 150,000 (extra)
    steps = 150,000 (160 ep)

The cap applies only with ``--epochs``. Without it, an explicit ``--save_freq``
above 20,000 is left exactly as given, like the rest of the schedule.

The two extra checkpoints show whether training past the target helps or starts
to overfit. Because lerobot's LR schedule spans the whole run (``cfg.steps``),
the target checkpoint is NOT identical to the final checkpoint of a standalone
run that stops at the target: its learning rate has not finished decaying.

Usage
-----
``--epochs N`` is consumed by `pop_epochs_flag` before draccus parses argv, then
applied inside the dataset factory hook (the one place that has both the WRAPPED
dataset — so ``len()`` is post-wrapping — and ``cfg``, before the training loop
reads ``cfg.steps``).

Without ``--epochs`` nothing is rescaled, but the equivalent epoch count is
still logged, so two runs can never again be compared without seeing it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "EXTRA_CHECKPOINTS",
    "MAX_SAVE_FREQ",
    "EpochPlan",
    "pop_epochs_flag",
    "effective_batch_size",
    "epochs_for_steps",
    "steps_for_epochs",
    "checkpoint_schedule",
    "resolve_epoch_steps",
]

# Checkpoint intervals trained PAST the epoch target whenever --epochs is given.
# Not applied without --epochs: there is no epoch target to extend past then,
# and the schedule is left exactly as lerobot received it.
EXTRA_CHECKPOINTS = 2

# Upper bound on save_freq (in optimizer steps) whenever --epochs is given. A
# spacing that would exceed it is tightened by adding checkpoints, so one still
# lands on the epoch target. Not applied without --epochs.
MAX_SAVE_FREQ = 20_000

# Longest checkpoint table printed in full; longer ones elide the middle.
_MAX_TABLE_ROWS = 12


class EpochPlan:
    """The resolved schedule, for logging, tests and provenance stamping.

    Attributes:
        num_samples: Length of the dataset actually handed to the dataloader
            (post-wrapping), i.e. the denominator of the epoch arithmetic.
        batch_size: `cfg.batch_size`.
        grad_accum: Gradient-accumulation factor (1 when unset).
        effective_batch: ``batch_size * grad_accum`` — samples per optimizer step.
        steps_before: `cfg.steps` as it arrived from the CLI/defaults.
        steps_after: `cfg.steps` as written back (== steps_before when epochs
            was None). Includes the extra checkpoint intervals.
        target_steps_min: ``ceil(epochs * num_samples / effective_batch)``, the
            fewest steps covering the requested epochs (None without epochs).
        target_step: Step of the checkpoint that lands on the epoch target,
            ``>= target_steps_min`` (== steps_after without epochs).
        save_freq: `cfg.save_freq` as written back (None when absent and
            epochs was None).
        n_target_checkpoints: Checkpoints up to and including the target.
        n_checkpoints_incoming: Checkpoints the incoming schedule implied
            (``round(steps_before / save_freq_before)``; 0 without epochs).
        max_save_freq: The save_freq cap applied (None without epochs).
        save_freq_uncapped: The spacing the incoming checkpoint count would
            have given (None without epochs). Above `max_save_freq` iff the
            cap fired.
        save_freq_capped: True when the cap raised the checkpoint count.
        extra_checkpoints: Checkpoint intervals trained past the target
            (`EXTRA_CHECKPOINTS` with epochs, else 0).
        epochs_requested: The ``--epochs`` value, or None.
        rescaled: ``{field: (old, new)}`` for each interval actually rescaled.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        batch_size: int,
        grad_accum: int,
        steps_before: int,
        steps_after: int,
        epochs_requested: float | None,
        rescaled: dict[str, tuple[int, int]],
        target_steps_min: int | None = None,
        target_step: int | None = None,
        save_freq: int | None = None,
        n_target_checkpoints: int = 0,
        extra_checkpoints: int = 0,
        n_checkpoints_incoming: int = 0,
        max_save_freq: int | None = None,
        save_freq_uncapped: int | None = None,
        save_freq_capped: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.effective_batch = batch_size * grad_accum
        self.steps_before = steps_before
        self.steps_after = steps_after
        self.epochs_requested = epochs_requested
        self.rescaled = rescaled
        self.target_steps_min = target_steps_min
        self.target_step = steps_after if target_step is None else target_step
        self.save_freq = save_freq
        self.n_target_checkpoints = n_target_checkpoints
        self.extra_checkpoints = extra_checkpoints
        self.n_checkpoints_incoming = n_checkpoints_incoming
        self.max_save_freq = max_save_freq
        self.save_freq_uncapped = save_freq_uncapped
        self.save_freq_capped = save_freq_capped

    def _epochs(self, steps: int) -> float:
        return epochs_for_steps(steps, self.effective_batch, self.num_samples)

    @property
    def epochs_before(self) -> float:
        """Epoch equivalent of `steps_before`."""
        return self._epochs(self.steps_before)

    @property
    def epochs_after(self) -> float:
        """Epoch equivalent of `steps_after` (the whole run)."""
        return self._epochs(self.steps_after)

    @property
    def epochs_target(self) -> float:
        """Epoch equivalent of `target_step`."""
        return self._epochs(self.target_step)

    @property
    def changed(self) -> bool:
        """True when the step count was actually rewritten."""
        return self.steps_after != self.steps_before

    @property
    def checkpoint_steps(self) -> list[int]:
        """Steps at which interval checkpoints are saved (empty without epochs).

        `steps_after` is a whole multiple of `save_freq` by construction, so the
        last entry is the end of the run.
        """
        if self.epochs_requested is None or not self.save_freq:
            return []
        return list(range(self.save_freq, self.steps_after + 1, self.save_freq))

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary for stamping next to a checkpoint."""
        return {
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "effective_batch": self.effective_batch,
            "steps": self.steps_after,
            "steps_before": self.steps_before,
            "epochs": round(self.epochs_after, 3),
            "epochs_before": round(self.epochs_before, 3),
            "epochs_requested": self.epochs_requested,
            "target_step": self.target_step,
            "epochs_target": round(self.epochs_target, 3),
            "save_freq": self.save_freq,
            "max_save_freq": self.max_save_freq,
            "save_freq_capped": self.save_freq_capped,
            "n_checkpoints_incoming": self.n_checkpoints_incoming,
            "n_target_checkpoints": self.n_target_checkpoints,
            "extra_checkpoints": self.extra_checkpoints,
            "checkpoint_steps": self.checkpoint_steps,
            "rescaled_intervals": {k: {"from": v[0], "to": v[1]} for k, v in self.rescaled.items()},
        }

    def describe(self) -> str:
        """Multi-line human summary, including the full arithmetic."""
        lines = [
            "[epochs] dataset samples = %d | batch %d x grad_accum %d = effective %d"
            % (self.num_samples, self.batch_size, self.grad_accum, self.effective_batch),
        ]
        if self.epochs_requested is None:
            lines.append(
                "[epochs] --epochs NOT given: keeping steps = %d, which is %.1f epochs "
                "(steps x effective_batch / samples). Pass --epochs to schedule by "
                "epochs instead; comparing two runs by steps alone is not meaningful "
                "when their datasets differ in size." % (self.steps_after, self.epochs_after)
            )
            return "\n".join(lines)

        lines.append(
            "[epochs] --epochs %g -> epoch target = ceil(%g x %d / %d) = %d steps"
            % (
                self.epochs_requested,
                self.epochs_requested,
                self.num_samples,
                self.effective_batch,
                self.target_steps_min,
            )
        )
        rounded = ""
        if self.target_step != self.target_steps_min:
            rounded = " (rounded up from %d so it lands on a checkpoint)" % self.target_steps_min
        lines.append(
            "[epochs] %d checkpoints up to the target, every %d steps (%.1f epochs); "
            "target checkpoint at step %d (%.2f epochs)%s"
            % (
                self.n_target_checkpoints,
                self.save_freq,
                self._epochs(self.save_freq),
                self.target_step,
                self.epochs_target,
                rounded,
            )
        )
        if self.save_freq_capped:
            lines.append(
                "[epochs] save_freq capped at %d: %d checkpoints up to the target "
                "would be every %d steps, so %d are used instead"
                % (
                    self.max_save_freq,
                    self.n_checkpoints_incoming,
                    self.save_freq_uncapped,
                    self.n_target_checkpoints,
                )
            )
        lines.append(
            "[epochs] + %d extra checkpoint intervals -> steps = %d (%.1f epochs)  "
            "(was %d, i.e. %.1f epochs)"
            % (
                self.extra_checkpoints,
                self.steps_after,
                self.epochs_after,
                self.steps_before,
                self.epochs_before,
            )
        )
        for field, (old, new) in self.rescaled.items():
            lines.append("[epochs] rescaled %s %d -> %d" % (field, old, new))
        lines.append("[epochs] checkpoints (step, epochs):")
        lines.extend(self._checkpoint_table())
        return "\n".join(lines)

    def _checkpoint_table(self) -> list[str]:
        steps = self.checkpoint_steps
        rows = []
        for s in steps:
            tag = ""
            if s == self.target_step:
                tag = "  <- epoch target"
            elif s > self.target_step:
                tag = "  <- extra"
            rows.append("[epochs]   %9d  %8.2f%s" % (s, self._epochs(s), tag))
        if len(rows) <= _MAX_TABLE_ROWS:
            return rows
        # Keep the head and everything from just before the target onward —
        # the target and the extra checkpoints are the rows that matter.
        tail = self.extra_checkpoints + 2
        head = max(1, _MAX_TABLE_ROWS - tail - 1)
        if head + tail >= len(rows):
            return rows
        hidden = len(rows) - head - tail
        return [*rows[:head], "[epochs]   ... (%d more)" % hidden, *rows[-tail:]]


def pop_epochs_flag(argv: list[str]) -> tuple[float | None, list[str]]:
    """Consume ``--epochs N`` (or ``--epochs=N``) from argv.

    This is OUR flag, not lerobot's, so it must be stripped before draccus
    parses the rest or it raises 'unrecognized arguments' — same contract as
    the existing ``--wrt-start`` handling.

    Returns:
        ``(epochs, remaining_argv)``; epochs is None when the flag is absent.

    Raises:
        ValueError: if the flag is present but the value is missing, not a
            number, or not strictly positive. Failing loudly matters here: a
            silently-ignored --epochs would train on the default step schedule
            and look completely normal.
    """
    epochs: float | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--epochs":
            if i + 1 >= len(argv):
                raise ValueError("--epochs requires a value, e.g. --epochs 120")
            epochs = _parse_positive_float(argv[i + 1])
            i += 2
            continue
        if a.startswith("--epochs="):
            epochs = _parse_positive_float(a.split("=", 1)[1])
            i += 1
            continue
        remaining.append(a)
        i += 1
    return epochs, remaining


def _parse_positive_float(raw: str) -> float:
    try:
        val = float(raw)
    except ValueError as e:
        raise ValueError(f"--epochs expects a number, got {raw!r}") from e
    if val <= 0 or not math.isfinite(val):
        raise ValueError(f"--epochs must be a finite positive number, got {val}")
    return val


def effective_batch_size(cfg: Any) -> tuple[int, int]:
    """Return ``(batch_size, grad_accum)`` from a lerobot TrainPipelineConfig.

    `batch_size` is required — without it the epoch arithmetic is impossible, so
    a missing field raises rather than silently assuming a value and producing a
    wrong step count. Gradient accumulation is optional and defaults to 1; the
    field name differs across lerobot versions, so several spellings are tried.
    """
    batch_size = getattr(cfg, "batch_size", None)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise RuntimeError(
            "Cannot schedule by epochs: the training config has no usable "
            f"'batch_size' (found {batch_size!r}). lerobot may have renamed it — "
            "locate the field on TrainPipelineConfig and update "
            "crisp_gym/util/epoch_steps.py:effective_batch_size. Refusing to guess, "
            "because a wrong batch size silently yields a wrong step count."
        )
    grad_accum = 1
    for name in ("gradient_accumulation_steps", "grad_accum_steps", "gradient_accumulate_every"):
        val = getattr(cfg, name, None)
        if isinstance(val, int) and val > 0:
            grad_accum = val
            break
    return batch_size, grad_accum


def epochs_for_steps(steps: int, effective_batch: int, num_samples: int) -> float:
    """``steps * effective_batch / num_samples`` (0.0 when the dataset is empty)."""
    if num_samples <= 0:
        return 0.0
    return steps * effective_batch / num_samples


def steps_for_epochs(epochs: float, effective_batch: int, num_samples: int) -> int:
    """Smallest step count covering `epochs` passes, always at least 1.

    Rounds UP so the requested number of epochs is fully covered rather than
    truncated; the overshoot is below one step.
    """
    if num_samples <= 0:
        raise ValueError("cannot convert epochs to steps for an empty dataset")
    return max(1, math.ceil(epochs * num_samples / effective_batch))


def checkpoint_schedule(
    target_steps_min: int,
    n_checkpoints: int,
    extra_checkpoints: int = EXTRA_CHECKPOINTS,
    max_save_freq: int | None = MAX_SAVE_FREQ,
) -> tuple[int, int, int, int]:
    """Lay checkpoints out so one lands exactly on the epoch target.

    Args:
        target_steps_min: Fewest steps that cover the requested epochs.
        n_checkpoints: Checkpoints wanted up to and including the target. Capped
            at `target_steps_min`, since save_freq cannot go below 1.
        extra_checkpoints: Checkpoint intervals to train past the target.
        max_save_freq: Upper bound on the spacing. When `n_checkpoints` would
            space them further apart, the count is raised to
            ``ceil(target_steps_min / max_save_freq)`` instead. None disables.

    Returns:
        ``(save_freq, target_step, total_steps, n_used)`` where
        ``save_freq <= max_save_freq``, ``target_step = n_used * save_freq >=
        target_steps_min`` (overshoot below `n_used` steps) and ``total_steps =
        target_step + extra_checkpoints * save_freq``.
    """
    if target_steps_min <= 0:
        raise ValueError("target_steps_min must be positive")
    if extra_checkpoints < 0:
        raise ValueError("extra_checkpoints must be >= 0")
    if max_save_freq is not None and max_save_freq < 1:
        raise ValueError("max_save_freq must be >= 1 (or None to disable)")
    n_used = max(1, min(n_checkpoints, target_steps_min))
    if max_save_freq is not None:
        # Fewest checkpoints whose spacing fits under the cap. Since
        # target_steps_min / n_used <= max_save_freq and the cap is an integer,
        # the ceil below cannot exceed it either.
        n_used = max(n_used, math.ceil(target_steps_min / max_save_freq))
    save_freq = math.ceil(target_steps_min / n_used)
    target_step = n_used * save_freq
    total_steps = target_step + extra_checkpoints * save_freq
    return save_freq, target_step, total_steps, n_used


def resolve_epoch_steps(cfg: Any, dataset: Any, epochs: float | None) -> EpochPlan:
    """Apply an epoch target to `cfg` in place and return the resolved plan.

    Call this from the dataset-factory hook, with the dataset the dataloader
    will actually see (i.e. AFTER any relative-pose wrapping — a wrapper can
    change ``len()``), and before the training loop reads ``cfg.steps``.

    With `epochs`, `cfg.steps`, `cfg.save_freq` and (if enabled)
    `cfg.eval_freq` are rewritten per the module's "Checkpoint schedule":
    a checkpoint lands on the epoch target, `save_freq` never exceeds
    `MAX_SAVE_FREQ`, and training continues for `EXTRA_CHECKPOINTS` more
    checkpoint intervals.

    With ``epochs=None`` nothing is modified; the equivalent epoch count is
    still computed and logged, so a run's length in epochs is always on record.

    Args:
        cfg: lerobot TrainPipelineConfig (mutated when `epochs` is given).
        dataset: Object supporting ``len()``.
        epochs: Target epoch count, or None to only report.

    Returns:
        The `EpochPlan`, for logging, stamping and tests.

    Raises:
        RuntimeError: if `batch_size`/`steps` cannot be read off `cfg`, the
            dataset is empty, or — with `epochs` — checkpointing is disabled or
            `save_freq` is unusable (the extra checkpoints could not exist).
    """
    num_samples = len(dataset)
    if num_samples <= 0:
        raise RuntimeError(
            "Cannot schedule by epochs: the dataset reports 0 samples. Check "
            "--dataset.repo_id / --dataset.root and the episode selection."
        )

    batch_size, grad_accum = effective_batch_size(cfg)
    effective_batch = batch_size * grad_accum

    steps_before = getattr(cfg, "steps", None)
    if not isinstance(steps_before, int) or steps_before <= 0:
        raise RuntimeError(
            "Cannot schedule by epochs: the training config has no usable 'steps' "
            f"(found {steps_before!r}). lerobot may have renamed it — update "
            "crisp_gym/util/epoch_steps.py:resolve_epoch_steps."
        )

    if epochs is None:
        save_freq = getattr(cfg, "save_freq", None)
        plan = EpochPlan(
            num_samples=num_samples,
            batch_size=batch_size,
            grad_accum=grad_accum,
            steps_before=steps_before,
            steps_after=steps_before,
            epochs_requested=None,
            rescaled={},
            save_freq=save_freq if isinstance(save_freq, int) else None,
        )
        for line in plan.describe().splitlines():
            logger.info(line)
        return plan

    # The extra checkpoints are the point of the schedule; refuse rather than
    # train past the target with nothing saved to compare.
    if getattr(cfg, "save_checkpoint", True) is False:
        raise RuntimeError(
            f"--epochs schedules {EXTRA_CHECKPOINTS} checkpoint intervals past the "
            "epoch target, but checkpointing is disabled (save_checkpoint=false). "
            "Enable checkpointing or drop --epochs."
        )
    save_before = getattr(cfg, "save_freq", None)
    if not isinstance(save_before, int) or save_before <= 0:
        raise RuntimeError(
            f"--epochs schedules {EXTRA_CHECKPOINTS} checkpoint intervals past the "
            f"epoch target, which needs a positive integer 'save_freq' (found "
            f"{save_before!r}). lerobot may have renamed it — update "
            "crisp_gym/util/epoch_steps.py:resolve_epoch_steps."
        )

    target_steps_min = steps_for_epochs(epochs, effective_batch, num_samples)
    # Keep the checkpoint count the incoming schedule would have produced, so a
    # shorter run does not silently save fewer checkpoints (and miss the best).
    n_checkpoints = max(1, round(steps_before / save_before))
    save_freq, target_step, steps_after, n_used = checkpoint_schedule(
        target_steps_min, n_checkpoints, EXTRA_CHECKPOINTS, MAX_SAVE_FREQ
    )
    # What the incoming checkpoint count would have given without the cap —
    # only for reporting whether (and why) the cap changed the layout.
    save_freq_uncapped, _, _, n_uncapped = checkpoint_schedule(
        target_steps_min, n_checkpoints, EXTRA_CHECKPOINTS, None
    )
    save_freq_capped = n_used > n_uncapped

    rescaled: dict[str, tuple[int, int]] = {}
    if save_freq != save_before:
        cfg.save_freq = save_freq
        rescaled["save_freq"] = (save_before, save_freq)

    # eval_freq follows save_freq by the same factor, so evals stay aligned with
    # checkpoints wherever they were before. 0/absent = disabled: left alone.
    eval_before = getattr(cfg, "eval_freq", None)
    if isinstance(eval_before, int) and eval_before > 0:
        eval_after = max(1, min(round(eval_before * save_freq / save_before), steps_after))
        if eval_after != eval_before:
            cfg.eval_freq = eval_after
            rescaled["eval_freq"] = (eval_before, eval_after)

    cfg.steps = steps_after

    plan = EpochPlan(
        num_samples=num_samples,
        batch_size=batch_size,
        grad_accum=grad_accum,
        steps_before=steps_before,
        steps_after=steps_after,
        epochs_requested=epochs,
        rescaled=rescaled,
        target_steps_min=target_steps_min,
        target_step=target_step,
        save_freq=save_freq,
        n_target_checkpoints=n_used,
        extra_checkpoints=EXTRA_CHECKPOINTS,
        n_checkpoints_incoming=n_uncapped,
        max_save_freq=MAX_SAVE_FREQ,
        save_freq_uncapped=save_freq_uncapped,
        save_freq_capped=save_freq_capped,
    )
    for line in plan.describe().splitlines():
        logger.info(line)
    return plan
