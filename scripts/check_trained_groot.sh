#!/usr/bin/env bash
# Did a finished GR00T run actually keep the flags you passed it?
#
# WHY THIS EXISTS
#   Fine-tuning from --policy.path can carry base-model metadata forward into
#   the saved config while the statistics are refit from your data. That is how
#   a SmolVLA checkpoint in this project ended up declaring a 6-wide state
#   against 16-wide stats: the command line said one thing and the checkpoint
#   recorded another, with nothing failing in between.
#
#   For GR00T the same drift is worse than a crash, because the two flags that
#   matter have no runtime symptom. A run that silently kept
#   use_relative_actions=false decodes relative actions against absolute
#   statistics; one that lost relative_exclude_joints trains the gripper as a
#   delta. Both produce a checkpoint that loads, deploys, and behaves badly.
#
#   So: compare what you ASKED for (train_config.json) against what the policy
#   SAVED (config.json), and fail when they disagree.
#
# STANDALONE
#   python3 with the stdlib only. No torch, no lerobot, no GPU.
#
# USAGE
#   bash scripts/check_trained_groot.sh <output_dir>
#   bash scripts/check_trained_groot.sh <output_dir>/checkpoints/020000/pretrained_model
#
#   Given a training output_dir it finds the newest checkpoint under
#   checkpoints/*/pretrained_model/ by itself.
#
# EXIT CODES
#   0  the flags survived
#   1  a flag was lost or disagrees with what was requested
#   2  no config.json found

set -u

DIR="${1:-}"
if [ -z "$DIR" ]; then
    echo "usage: bash scripts/check_trained_groot.sh <output_dir|pretrained_model_dir>" >&2
    exit 2
fi

PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" python3 - "$DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

BAR = "-" * 70
root = Path(sys.argv[1])
failures: list[str] = []
warnings: list[str] = []


def line(k, v):
    print(f"  {k:<26} {v}")


def find_config(base: Path) -> Path | None:
    """config.json here, else the newest checkpoints/*/pretrained_model/config.json."""
    direct = base / "config.json"
    if direct.exists():
        return direct
    candidates = sorted(base.glob("checkpoints/*/pretrained_model/config.json"))
    return candidates[-1] if candidates else None


def load(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  could not read {path}: {exc}")
        return {}


def find_key(node, key):
    """First value for `key` at any depth — draccus nests policy config differently
    between config.json (flat) and train_config.json (under 'policy')."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


cfg_path = find_config(root)
if cfg_path is None:
    print(f"  no config.json under {root} (nor checkpoints/*/pretrained_model/)")
    sys.exit(2)

cfg = load(cfg_path)
print(BAR)
print("  GR00T trained-checkpoint flag check")
print(BAR)
line("checkpoint", cfg_path.parent)

# train_config.json records the request; it sits beside config.json or one level up.
train_cfg, train_path = {}, None
for candidate in (cfg_path.parent / "train_config.json",
                  cfg_path.parent.parent.parent.parent / "train_config.json",
                  root / "train_config.json"):
    if candidate.exists():
        train_cfg, train_path = load(candidate), candidate
        break
line("train_config.json", train_path or "not found (comparison skipped)")

policy_type = cfg.get("type") or find_key(cfg, "type")
line("policy type", policy_type)
if policy_type not in (None, "groot"):
    warnings.append(f"policy type is {policy_type!r}, not 'groot' — checking anyway")
print()


def check(key, saved, want=None, contains=None):
    """Compare the saved value against the requested one and the required one."""
    asked = find_key(train_cfg, key) if train_cfg else None
    shown = f"{saved!r}"
    if asked is not None and asked != saved:
        shown += f"   (train_config asked for {asked!r})"
        failures.append(f"{key}: saved {saved!r} but {asked!r} was requested")
    line(key, shown)
    if want is not None and saved != want:
        failures.append(f"{key} is {saved!r}, must be {want!r}")
    if contains is not None:
        values = saved if isinstance(saved, list) else []
        if not any(contains in str(v).lower() for v in values):
            failures.append(f"{key} does not contain {contains!r}")


check("use_relative_actions", cfg.get("use_relative_actions"), want=True)
check("relative_exclude_joints", cfg.get("relative_exclude_joints"), contains="gripper")
check("embodiment_tag", cfg.get("embodiment_tag"))
line("chunk_size", f"{cfg.get('chunk_size')}  (n_action_steps {cfg.get('n_action_steps')})")

# The stale-metadata trap: declared feature widths vs what the data actually is.
print()
for name in ("input_features", "output_features"):
    feats = cfg.get(name) or {}
    if not isinstance(feats, dict) or not feats:
        continue
    print(f"  {name}:")
    for key, spec in feats.items():
        shape = spec.get("shape") if isinstance(spec, dict) else None
        print(f"    {key:<34} {shape}")

# Image geometry. These live in processor_kwargs, which lerobot loads from the
# checkpoint's processor sidecars -- NOT from config.json -- so search every
# JSON in the directory. Absent here means "not recorded in this checkpoint",
# which is not the same as "no transform": the base checkpoint's own files are
# what _load_n1_7_checkpoint_processor_assets reads at runtime.
kwargs_keys = ["image_target_size", "image_crop_size", "shortest_image_edge",
               "crop_fraction", "letter_box_transform", "use_albumentations"]
geometry: dict = {}
for json_path in sorted(cfg_path.parent.glob("*.json")):
    blob = load(json_path)
    for k in kwargs_keys:
        found_val = find_key(blob, k)
        if found_val is not None and k not in geometry:
            geometry[k] = (found_val, json_path.name)

print()
if geometry:
    print("  image geometry (what the VLM does to each frame):")
    for k, (v, where) in geometry.items():
        line(f"  {k}", f"{v!r}   [{where}]")
    if geometry.get("use_albumentations", (None,))[0]:
        warnings.append("use_albumentations=True: shortest-edge resize then CENTER CROP "
                        "-- a wide frame loses its periphery")
    elif geometry.get("image_target_size", (None,))[0] is None:
        pass
    elif not geometry.get("letter_box_transform", (None,))[0]:
        warnings.append("letter_box_transform is off: frames are resized to a square, so a "
                        "wide frame is squashed rather than cropped (FOV kept, aspect lost)")
    crop = geometry.get("crop_fraction", (None,))[0]
    if isinstance(crop, (int, float)) and 0 < crop < 1:
        warnings.append(f"crop_fraction={crop}: a centered crop keeps only the middle "
                        f"{100 * crop:.0f}% -- the periphery is discarded")
else:
    print("  image geometry: no processor_kwargs recorded in this checkpoint's JSON.")
    print("    That is NOT the same as 'no transform' -- lerobot reads them from the")
    print("    BASE checkpoint at runtime. To see what actually applies, probe the base:")
    print("      python -c \"from huggingface_hub import snapshot_download as d; print(d('nvidia/GR00T-N1.7-3B', allow_patterns=['*.json']))\"")
    print("    then grep those files for image_target_size / letter_box_transform /")
    print("    crop_fraction / use_albumentations.")

print()
print(BAR)
if failures:
    print("  FAIL — the run did not keep what it was given")
    for f in failures:
        print(f"    - {f}")
    print()
    print("    A checkpoint in this state still loads and still deploys. Retrain")
    print("    with the flags, or the gripper trains as a delta and the decode")
    print("    runs relative against absolute statistics.")
    print(BAR)
    sys.exit(1)

print("  PASS — use_relative_actions and relative_exclude_joints survived training.")
for w in warnings:
    print(f"    WARN  {w}")
print(BAR)
PYEOF
