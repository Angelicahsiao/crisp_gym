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

This module converts an epoch target into the step count lerobot wants, and —
because `cfg.steps` changes — rescales the step-denominated checkpoint and eval
intervals so a shorter run still produces as many checkpoints as a longer one.
`log_freq` is deliberately NOT rescaled: it controls console/wandb density,
which should stay constant so logs remain readable across runs.

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
    "EpochPlan",
    "pop_epochs_flag",
    "effective_batch_size",
    "epochs_for_steps",
    "steps_for_epochs",
    "resolve_epoch_steps",
]

# Intervals rescaled alongside cfg.steps. log_freq is excluded on purpose: it
# sets logging DENSITY, which should not change with run length.
_RESCALED_FREQ_FIELDS = ("save_freq", "eval_freq")


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
            was None).
        epochs_before: Epoch equivalent of `steps_before`.
        epochs_after: Epoch equivalent of `steps_after`.
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
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.effective_batch = batch_size * grad_accum
        self.steps_before = steps_before
        self.steps_after = steps_after
        self.epochs_requested = epochs_requested
        self.rescaled = rescaled
        self.epochs_before = epochs_for_steps(steps_before, self.effective_batch, num_samples)
        self.epochs_after = epochs_for_steps(steps_after, self.effective_batch, num_samples)

    @property
    def changed(self) -> bool:
        """True when the step count was actually rewritten."""
        return self.steps_after != self.steps_before

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
            "[epochs] --epochs %g -> steps = ceil(%g x %d / %d) = %d  (was %d, i.e. %.1f epochs)"
            % (
                self.epochs_requested,
                self.epochs_requested,
                self.num_samples,
                self.effective_batch,
                self.steps_after,
                self.steps_before,
                self.epochs_before,
            )
        )
        for field, (old, new) in self.rescaled.items():
            lines.append(
                "[epochs] rescaled %s %d -> %d (same count per run at the new step total)"
                % (field, old, new)
            )
        if not self.rescaled:
            lines.append("[epochs] no interval fields to rescale on this config")
        return "\n".join(lines)


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


def resolve_epoch_steps(cfg: Any, dataset: Any, epochs: float | None) -> EpochPlan:
    """Apply an epoch target to `cfg` in place and return the resolved plan.

    Call this from the dataset-factory hook, with the dataset the dataloader
    will actually see (i.e. AFTER any relative-pose wrapping — a wrapper can
    change ``len()``), and before the training loop reads ``cfg.steps``.

    With ``epochs=None`` nothing is modified; the equivalent epoch count is
    still computed and logged, so a run's length in epochs is always on record.

    Args:
        cfg: lerobot TrainPipelineConfig (mutated when `epochs` is given).
        dataset: Object supporting ``len()``.
        epochs: Target epoch count, or None to only report.

    Returns:
        The `EpochPlan`, for logging, stamping and tests.

    Raises:
        RuntimeError: if `batch_size`/`steps` cannot be read off `cfg`, or the
            dataset is empty.
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

    rescaled: dict[str, tuple[int, int]] = {}
    steps_after = steps_before

    if epochs is not None:
        steps_after = steps_for_epochs(epochs, effective_batch, num_samples)
        # Keep the number of checkpoints/evals per run constant. Scaling by the
        # step ratio means a 40k-step run saves as often per unit of training as
        # a 100k-step one did, instead of producing 2.5x fewer checkpoints and
        # possibly skipping the best one.
        ratio = steps_after / steps_before
        for field in _RESCALED_FREQ_FIELDS:
            old = getattr(cfg, field, None)
            if isinstance(old, int) and old > 0:
                new = max(1, min(round(old * ratio), steps_after))
                if new != old:
                    setattr(cfg, field, new)
                    rescaled[field] = (old, new)
        cfg.steps = steps_after

    plan = EpochPlan(
        num_samples=num_samples,
        batch_size=batch_size,
        grad_accum=grad_accum,
        steps_before=steps_before,
        steps_after=steps_after,
        epochs_requested=epochs,
        rescaled=rescaled,
    )
    for line in plan.describe().splitlines():
        logger.info(line)
    return plan
