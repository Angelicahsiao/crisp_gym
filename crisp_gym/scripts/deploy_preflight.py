#!/usr/bin/env python3
"""Pre-rollout checks for a crisp_gym deployment: A5 + B1-B3 in one run.

Run this in the SAME environment the robot deploys from (the one with lerobot
and torch), pointed at the checkpoint you intend to roll out.

    python deploy_preflight.py <checkpoint_path> [--dataset <dataset_root>]

Checks, in the order they can bite you:

  B3  lerobot version, and whether the four APIs relative_lerobot_policy.py
      depends on still exist. HANDOFF.md:151 claims the wrapper needs 0.4.4;
      this says whether the claim still matters on 0.6.1.
  B2  pose_repr.json / action_repr.json discovery, using the SAME walk-up the
      wrapper uses. A missing pose_repr.json makes it assume ABSOLUTE, which
      silently moves a relative checkpoint the wrong way on the first step.
  B1  image keys: what the checkpoint expects vs what the dataset recorded.
      A mismatch means the policy sees blank cameras and nothing errors.
  A5  whether batch["task"] must be a bare str or a list[str]. Determined
      empirically by running the real preprocessor both ways, because
      numpy_obs_to_torch currently passes a bare str through untouched.

Exit code is 0 only when every check that could run, passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FAIL: list[str] = []
WARN: list[str] = []


def head(title: str) -> None:
    """Section banner."""
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


# ── B3 ────────────────────────────────────────────────────────────────────────


def check_b3() -> None:
    """Report the lerobot version and the API surface the deploy wrapper calls."""
    head("B3  lerobot version and API surface")
    import lerobot

    version = getattr(lerobot, "__version__", "unknown")
    print(f"  lerobot {version}  ({Path(lerobot.__file__).parent})")

    probes = [
        ("lerobot.configs.train", "TrainPipelineConfig"),
        ("lerobot.policies.factory", "get_policy_class"),
        ("lerobot.policies.factory", "make_pre_post_processors"),
        ("lerobot.policies.utils", "populate_queues"),
    ]
    for module, attr in probes:
        try:
            mod = __import__(module, fromlist=[attr])
            ok = hasattr(mod, attr)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  ** {module}.{attr}: import failed — {exc}")
        if ok:
            print(f"  ok  {module}.{attr}")
        else:
            FAIL.append(f"B3: {module}.{attr} missing — the deploy wrapper calls it")
            print(f"  ** {module}.{attr} MISSING")


# ── B2 ────────────────────────────────────────────────────────────────────────


def find_stamp(pretrained_path: Path, name: str) -> tuple[Path | None, dict | None]:
    """Same walk-up as relative_lerobot_policy.find_pose_repr (self + 4 parents)."""
    p = pretrained_path.resolve()
    for candidate in (p, *list(p.parents)[:4]):
        f = candidate / name
        if f.exists():
            try:
                return f, json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                return f, None
    return None, None


def check_b2(ckpt: Path) -> str:
    """Are the pose/action representation stamps discoverable from the checkpoint?

    Returns the observation.state mode the deploy wrapper will infer:
    "absolute", "relative", or "relative_wrt_start".
    """
    head("B2  pose_repr.json / action_repr.json")
    pose_path, pose = find_stamp(ckpt, "pose_repr.json")
    act_path, act = find_stamp(ckpt, "action_repr.json")

    if pose_path is None:
        WARN.append(
            "B2: no pose_repr.json found — the wrapper will ASSUME ABSOLUTE "
            "observation.state. If this checkpoint was trained on relative "
            "poses, force it with state_input in the policy config."
        )
        print("  ** pose_repr.json NOT FOUND in the checkpoint dir or 4 parents")
        print("     -> wrapper assumes ABSOLUTE observation.state")
        mode = "absolute"
    else:
        print(f"  found {pose_path}")
        print(f"        {json.dumps(pose, indent=8)[:400]}")
        obs_stamp = (pose or {}).get("observation", {})
        stamped = obs_stamp.get("observation.state", "")
        if str(stamped).startswith("relative"):
            mode = ("relative_wrt_start"
                    if obs_stamp.get("state_includes_wrt_start") else "relative")
        else:
            mode = "absolute"
        print(f"  observation stamp     : {stamped!r}")
        print(f"  state_includes_wrt_start: {obs_stamp.get('state_includes_wrt_start')}")
        print(f"  -> observation.state mode: {mode}")

    if act_path is None:
        print("  -- no action_repr.json (fine for a relative checkpoint)")
    else:
        print(f"  found {act_path}: {act}")
    return mode



# ── state width discovery ─────────────────────────────────────────────────────


def probe_state_width(policy, preprocessor) -> int | None:  # noqa: ANN001
    """The observation.state width the preprocessor really wants.

    config.input_features can disagree with the normalization statistics baked
    into the checkpoint -- a finetune started from a base model may keep the
    BASE input_features while refitting the stats on the new data. The stats
    win at runtime, so discover the true width by running the preprocessor and
    reading it out of the size-mismatch error.
    """
    in_feats = dict(getattr(policy.config, "input_features", {}) or {})
    declared = None
    if "observation.state" in in_feats:
        shape = tuple(getattr(in_feats["observation.state"], "shape", ()) or ())
        declared = int(shape[-1]) if shape else None
    if preprocessor is None or declared is None:
        return declared

    try:
        preprocessor(build_batch(policy, declared, "probe"))
        return declared
    except RuntimeError as exc:
        m = re.search(r"tensor a \((\d+)\) must match the size of tensor b \((\d+)\)",
                      str(exc))
        if m and int(m.group(1)) == declared:
            return int(m.group(2))
        return declared
    except Exception:  # noqa: BLE001
        return declared


def build_batch(policy, state_width: int | None, task_value):  # noqa: ANN001, ANN201
    """A dummy batch shaped from input_features, with observation.state overridden."""
    import torch

    batch = {}
    for key, feature in (getattr(policy.config, "input_features", {}) or {}).items():
        shape = tuple(getattr(feature, "shape", ()) or ())
        if key == "observation.state" and state_width is not None:
            shape = (state_width,)
        batch[key] = torch.zeros((1, *shape), dtype=torch.float32)
    batch["task"] = task_value
    return batch


# ── B1 ────────────────────────────────────────────────────────────────────────


def check_b1(policy, dataset: Path | None, state_mode: str, true_width: int | None) -> None:  # noqa: ANN001
    """Image/state keys the checkpoint expects vs what the dataset recorded.

    observation.state is NOT expected to match the dataset verbatim: crisp_gym's
    contract is absolute-on-disk / relative-at-train, and the
    "relative_wrt_start" generation appends a 6-D wrt-start rot6d to every
    frame. So the width the checkpoint wants is derived from the pose_repr mode.
    """
    head("B1  feature parity: checkpoint vs dataset")
    in_feats = dict(getattr(policy.config, "input_features", {}) or {})
    ckpt_images = sorted(k for k in in_feats if k.startswith("observation.images"))

    print("  checkpoint input_features:")
    for key, feature in in_feats.items():
        print(f"    {key:45s} shape={getattr(feature, 'shape', '?')}")

    if dataset is None:
        WARN.append("B1: no --dataset given — could not compare against training data")
        print("\n  (pass --dataset <root> to compare against the recorded dataset)")
        return

    feats = json.loads((dataset / "meta" / "info.json").read_text())["features"]
    ds_images = sorted(k for k in feats if k.startswith("observation.images"))
    print(f"\n  dataset image keys   : {ds_images}")
    print(f"  checkpoint image keys: {ckpt_images}")

    missing = [k for k in ckpt_images if k not in ds_images]
    extra = [k for k in ds_images if k not in ckpt_images]
    if missing:
        FAIL.append(f"B1: checkpoint expects image keys the dataset lacks: {missing}")
        print(f"  ** checkpoint expects, dataset lacks: {missing}")
        print("     (empty_cameras padding shows up here — the robot must fill "
              "these or the policy sees blanks)")
    if extra:
        print(f"  -- dataset has, checkpoint ignores: {extra}")
    if not missing and not extra:
        print("  ok  image keys match")

    if "observation.state" in in_feats and "observation.state" in feats:
        ck = tuple(getattr(in_feats["observation.state"], "shape", ()) or ())
        ds = tuple(feats["observation.state"].get("shape", ()) or ())
        extra = 6 if state_mode == "relative_wrt_start" else 0
        expected = int(ds[-1]) + extra if ds else None
        print(f"\n  observation.state  declared={ck}  dataset={ds}  mode={state_mode}")
        if true_width is not None and ck and int(ck[-1]) != true_width:
            FAIL.append(
                f"B1: the checkpoint contradicts ITSELF — input_features says "
                f"observation.state is {int(ck[-1])}, its normalization stats want "
                f"{true_width}. The stats win at runtime, so {int(ck[-1])} is stale "
                "metadata (typical when a finetune keeps the base model's features)."
            )
            print(f"  ** declared {int(ck[-1])} but the NORMALIZER wants {true_width}"
                  " — the checkpoint disagrees with itself; stats win at runtime")
        elif true_width is not None:
            print(f"  normalizer wants      : {true_width}")
        if expected is not None:
            note = f" (= dataset {int(ds[-1])} + {extra} wrt-start)" if extra else ""
            print(f"  expected for this mode: {expected}{note}")
        effective = true_width if true_width is not None else (int(ck[-1]) if ck else None)
        if effective is not None and expected is not None and effective != expected:
            FAIL.append(
                f"B1: effective observation.state is {effective} but mode "
                f"{state_mode!r} over a {int(ds[-1])}-D dataset implies {expected}"
            )
            print("  ** WIDTH MISMATCH — the normalizer will fail or silently mis-scale")
        elif ck:
            print("  ok  width matches the mode")


# ── A5 ────────────────────────────────────────────────────────────────────────


def check_a5(policy, preprocessor, state_width: int | None) -> None:  # noqa: ANN001
    """Does the preprocessor want batch["task"] as a bare str or a list[str]?"""
    head('A5  batch["task"]: bare str or list[str]?')

    if preprocessor is None:
        WARN.append("A5: no preprocessor for this checkpoint — nothing to probe")
        print("  (no preprocessor; skipping)")
        return


    results = {}
    shapes: dict[str, dict] = {}
    tensors: dict[str, dict] = {}
    for label, value in (("bare str", "open the power switch"),
                         ("list[str]", ["open the power switch"])):
        try:
            out = preprocessor(build_batch(policy, state_width, value))
            lang = {
                k: v for k, v in out.items()
                if any(t in k.lower() for t in ("lang", "token", "input_ids", "attention"))
            }
            shapes[label] = {k: tuple(getattr(v, "shape", ())) for k, v in sorted(lang.items())}
            tensors[label] = lang
            results[label] = ("OK", sorted(lang))
            print(f"  {label:9s} -> OK")
            for k, shp in shapes[label].items():
                print(f"              {k:45s} {shp}")
            if not lang:
                print("              (no language keys produced)")
        except Exception as exc:  # noqa: BLE001
            results[label] = ("FAIL", repr(exc)[:220])
            print(f"  {label:9s} -> FAIL {repr(exc)[:220]}")

    bare_ok = results.get("bare str", ("FAIL",))[0] == "OK"
    list_ok = results.get("list[str]", ("FAIL",))[0] == "OK"
    no_language = all(not hint for _, hint in results.values() if isinstance(hint, list))
    print()
    if bare_ok and list_ok and no_language:
        print("  NOT APPLICABLE — this policy consumes no language input, so both")
        print("  forms pass trivially and tell you nothing. Re-run against the")
        print("  SmolVLA checkpoint to answer A5.")
        WARN.append(
            "A5: unanswerable from this checkpoint — it takes no language input. "
            "Re-run against a VLA checkpoint."
        )
    elif bare_ok and list_ok:
        same_shapes = shapes.get("bare str") == shapes.get("list[str]")
        same_values = True
        try:
            import torch

            for k, v in tensors.get("bare str", {}).items():
                other = tensors.get("list[str]", {}).get(k)
                if other is None or not torch.equal(v, other):
                    same_values = False
                    break
        except Exception:  # noqa: BLE001
            same_values = False

        if same_shapes and same_values:
            print("  ok  IDENTICAL tokenization both ways — a bare str is already")
            print("      batched correctly. numpy_obs_to_torch needs NO change.")
        elif same_shapes:
            WARN.append(
                "A5: same shapes but different token VALUES between the two forms — "
                "inspect before trusting the bare str."
            )
            print("  ** same shapes but DIFFERENT values — inspect before trusting")
        else:
            FAIL.append(
                "A5: the two forms tokenize to different shapes "
                f"({shapes.get('bare str')} vs {shapes.get('list[str]')}); the list "
                "form is the batched one, so numpy_obs_to_torch must wrap "
                "(lerobot_features.py:326-327)."
            )
            print("  ** SHAPES DIFFER — the list form is the batched one;")
            print("     numpy_obs_to_torch must emit [task], not task")
    elif list_ok and not bare_ok:
        FAIL.append(
            'A5: preprocessor needs list[str]; numpy_obs_to_torch passes a bare str. '
            "Wrap it (lerobot_features.py:326-327)."
        )
        print("  ** WRAP REQUIRED: numpy_obs_to_torch must emit [task], not task")
    elif bare_ok and not list_ok:
        print("  ok  bare str is correct — numpy_obs_to_torch needs no change")
    else:
        FAIL.append("A5: neither form was accepted — inspect the traceback above")
        print("  ** neither form accepted")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="Path passed to --path at deploy time")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Dataset root (holding meta/) the checkpoint was trained on")
    args = ap.parse_args()

    check_b3()
    state_mode = check_b2(args.checkpoint)

    head("loading the checkpoint")
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    train_config = TrainPipelineConfig.from_pretrained(str(args.checkpoint))
    print(f"  policy type: {train_config.policy.type}")
    policy = get_policy_class(train_config.policy.type).from_pretrained(str(args.checkpoint))
    policy.eval()
    print(f"  loaded {policy.name}")
    for attr in ("n_obs_steps", "n_action_steps", "chunk_size"):
        if hasattr(policy.config, attr):
            print(f"  {attr}: {getattr(policy.config, attr)}")

    try:
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=str(args.checkpoint)
        )
    except Exception as exc:  # noqa: BLE001
        preprocessor = None
        WARN.append(f"make_pre_post_processors failed: {exc!r}")
        print(f"  ** make_pre_post_processors failed: {exc!r}")

    true_width = probe_state_width(policy, preprocessor)
    check_b1(policy, args.dataset, state_mode, true_width)
    check_a5(policy, preprocessor, true_width)

    head("summary")
    for w in WARN:
        print(f"  WARN  {w}")
    for f in FAIL:
        print(f"  FAIL  {f}")
    if not FAIL:
        print("  All checks that could run, passed."
              + ("  (see warnings above)" if WARN else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
