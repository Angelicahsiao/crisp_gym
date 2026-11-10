#!/usr/bin/env bash
set -euo pipefail

# --path is a bind mount, so anchor to this script's own directory rather than
# to whatever cwd the platform sets. Everything written under here persists.
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

# ── lerobot multi_task_dit training (server / bind-mount variant) ─────────────
#
# The multi_task_dit sibling of train_groot_server.sh: same layout, same
# environment-variable interface, same logs. multi_task_dit is lerobot's
# built-in Multi-Task Diffusion Transformer (lerobot >= 0.6): a CLIP ViT-B/16
# image encoder, a frozen CLIP text encoder reading the task string, and a
# transformer action head trained with diffusion or flow matching.
#
# LAYOUT — everything is relative to WORKDIR, the directory holding this file:
#
#   $WORKDIR/
#   ├── train_multi_task_dit_server.sh   this file
#   ├── crisp_gym/                       a CLONE of the crisp_gym repo
#   │   └── crisp_gym/
#   │       ├── scripts/
#   │       │   ├── groot_preflight.py           the dataset gate (warn + continue if absent)
#   │       │   ├── lerobot_relative_pose.py     default mode (hard fail if absent)
#   │       │   └── train_absolute_next_pose.py  ABSOLUTE=1 only (hard fail if absent)
#   │       └── util/
#   │           └── epoch_steps.py               EPOCHS=N; found by the launcher itself
#   ├── datasets/<DATASET_NAME>/         meta/ + data/, or .../lerobot/meta
#   ├── output/train/                    runs land here; created below
#   └── .home/.cache/huggingface/        the CLIP model (~600 MB) is cached here
#
#   DATASET=/full/path and OUT=/full/path escape this layout entirely.
#
# POSE REPRESENTATION — multi_task_dit has no relative-action mode of its own
# (unlike GR00T), so crisp_gym's launcher decides what the model sees:
#   default     lerobot_relative_pose.py: UMI-style relative poses, re-expressed
#               against the current TCP frame at load time. The dataset on disk
#               stays absolute. Deploy with RelativeLerobotPolicy.
#   ABSOLUTE=1  train_absolute_next_pose.py: poses exactly as recorded.
#
# SETTINGS PINNED FOR THIS PIPELINE — lerobot's multi_task_dit defaults assume
# 30 Hz data and CLIP-sized images; each of these fails quietly if left alone:
#   image_resize_shape [236,236] + image_crop_shape [224,224]
#       The default is a 224x224 crop with NO resize: on an 800x1280 frame that
#       is ~5% of the image in training and only the centre 5% at deploy. Resize
#       first so the whole view reaches CLIP (squashed to square), then a random
#       224x224 crop (~95%, UMI's ratio) in training, centre crop at inference.
#       The crop must be 224: CLIP ViT-B/16 takes 224x224 input.
#   horizon 16 / n_action_steps 8
#       The defaults (32 / 24) execute 24 steps open-loop: 1.6 s at 15 fps.
#   DDIM + 10 inference steps (or 10 flow-matching integration steps)
#       The defaults run 100 per chunk. crisp_gym's deploy-time
#       --num-inference-steps / --scheduler do NOT reach this policy, so the
#       inference speed is fixed here, at training time.
#
# The TASK STRING is required: the model's input size includes the text
# features, so a frame without one fails with a shape error, in training and at
# deploy. The preflight (G5) refuses a dataset without task strings.
#
#   ./train_multi_task_dit_server.sh                      # relative, 16-D state
#   WRT_START=0 ./train_multi_task_dit_server.sh          # ... 10-D plain relative state
#   ABSOLUTE=1 ./train_multi_task_dit_server.sh           # absolute baseline
#   EPOCHS=120 ./train_multi_task_dit_server.sh           # schedule by epochs, see below
#   OBJECTIVE=flow_matching ./train_multi_task_dit_server.sh
#   AUGMENT=all ./train_multi_task_dit_server.sh          # augmentation incl. rotate/shift
#   AUGMENT=0 ./train_multi_task_dit_server.sh            # no augmentation
#   STEPS=2000 ./train_multi_task_dit_server.sh           # smoke test
#   DATASET_NAME=other_set ./train_multi_task_dit_server.sh
#   OUT=/mnt/nfs_share/runs/x ./train_multi_task_dit_server.sh
#   SKIP_PREFLIGHT=1 ./train_multi_task_dit_server.sh     # bypass the dataset gate
#   SKIP_HUB_CHECK=1 ./train_multi_task_dit_server.sh     # bypass the hub gate
#   DRY_RUN=1 ./train_multi_task_dit_server.sh            # print the command, launch nothing
#
# Anything passed on the command line is appended to the lerobot command, so it
# overrides the settings above (e.g. --policy.num_layers=8). The one exception
# is --epochs: use EPOCHS=N, so the launch message and the run agree.
#
# EPOCHS=N schedules by EPOCHS instead of STEPS: a checkpoint lands on the
# N-epoch target, then 2 more checkpoint intervals run past it, each at most
# 20000 steps. STEPS / SAVE_FREQ (default 5) then only set how many checkpoints
# come before the target. The [epochs] lines of the log list every checkpoint
# and mark the target. See crisp_gym/util/epoch_steps.py. The transformer head
# overfits more readily than diffusion's UNet: UMI gave its transformer variant
# 200 epochs against 120 for the UNet, on far more data than a few minutes.

DATASET_NAME="${DATASET_NAME:-open_electribox_sum}"
DATASET="${DATASET:-$WORKDIR/datasets/$DATASET_NAME}"
SCRIPTS="$WORKDIR/crisp_gym/crisp_gym/scripts"
PREFLIGHT="$SCRIPTS/groot_preflight.py"

ABSOLUTE="${ABSOLUTE:-0}"
# 1 -> 16-D observation.state [rel_pose9, gripper1, wrt_start6] (UMI parity)
# 0 -> 10-D plain relative state. Relative mode only.
# Deploy MUST match: state_input relative_wrt_start vs relative.
WRT_START="${WRT_START:-1}"

_mode_tag="_rel"
[ "$ABSOLUTE" = "1" ] && _mode_tag="_abs"
OUT="${OUT:-$WORKDIR/output/train/mtdit${_mode_tag}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)}"

STEPS="${STEPS:-100000}"
EPOCHS="${EPOCHS:-}"                # empty = schedule by STEPS
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-}"    # empty = lerobot's multi_task_dit default (0)

OBJECTIVE="${OBJECTIVE:-diffusion}" # diffusion | flow_matching
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
HORIZON="${HORIZON:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
IMAGE_RESIZE="${IMAGE_RESIZE:-236}" # square resize before the 224x224 CLIP crop
# noaffine: lerobot's colour/sharpness jitter without the random rotate/shift
# all:      lerobot's full default set, rotate/shift included
# 0:        no augmentation
AUGMENT="${AUGMENT:-noaffine}"

SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ── validate the settings before anything slow runs ──────────────────────────
fail() { echo "$*" >&2; exit 2; }
is_pos_int() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]; }

if [ -n "$EPOCHS" ] && ! awk -v e="$EPOCHS" 'BEGIN { exit !(e ~ /^[0-9]*\.?[0-9]+$/ && e + 0 > 0) }'; then
    fail "EPOCHS must be a positive number, got: $EPOCHS"
fi
for a in "$@"; do
    case "$a" in
        --epochs|--epochs=*) fail "use EPOCHS=N on this launcher, not $a (the launch message reads EPOCHS)" ;;
    esac
done
[ "$ABSOLUTE" = "0" ] || [ "$ABSOLUTE" = "1" ] || fail "ABSOLUTE must be 0 or 1, got: $ABSOLUTE"
[ "$WRT_START" = "0" ] || [ "$WRT_START" = "1" ] || fail "WRT_START must be 0 or 1, got: $WRT_START"
case "$OBJECTIVE" in diffusion|flow_matching) ;; *) fail "OBJECTIVE must be diffusion or flow_matching, got: $OBJECTIVE" ;; esac
case "$AUGMENT" in noaffine|all|0) ;; *) fail "AUGMENT must be noaffine, all or 0, got: $AUGMENT" ;; esac
for v in STEPS BATCH_SIZE SAVE_FREQ NUM_WORKERS INFERENCE_STEPS HORIZON N_ACTION_STEPS IMAGE_RESIZE; do
    is_pos_int "${!v}" || fail "$v must be a positive integer, got: ${!v}"
done
[ -z "$WARMUP_STEPS" ] || [[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || fail "WARMUP_STEPS must be a non-negative integer, got: $WARMUP_STEPS"
# A resize below the crop makes lerobot DISABLE the crop (a warning, not an
# error), and CLIP then gets a non-224 image.
[ "$IMAGE_RESIZE" -ge 224 ] || fail "IMAGE_RESIZE must be >= 224 (CLIP crops 224x224), got: $IMAGE_RESIZE"
[ "$N_ACTION_STEPS" -le "$HORIZON" ] || fail "N_ACTION_STEPS ($N_ACTION_STEPS) cannot exceed HORIZON ($HORIZON)"

# Some datasets keep meta/ and data/ directly, others nest them under lerobot/.
if [ ! -d "$DATASET/meta" ] && [ -d "$DATASET/lerobot/meta" ]; then
    DATASET="$DATASET/lerobot"
fi
if [ ! -f "$DATASET/meta/info.json" ]; then
    echo "No meta/info.json under $DATASET" >&2
    echo "Set DATASET=/full/path/to/dataset or DATASET_NAME=<name>." >&2
    exit 2
fi

if [ "$ABSOLUTE" = "1" ]; then
    TRAINER="$SCRIPTS/train_absolute_next_pose.py"
else
    TRAINER="$SCRIPTS/lerobot_relative_pose.py"
fi
if [ ! -f "$TRAINER" ]; then
    echo "Training launcher missing: $TRAINER" >&2
    echo "Clone crisp_gym beside this script (see LAYOUT above)." >&2
    exit 2
fi

# The gate runs BEFORE the HOME redirect below, deliberately. It only reads
# local parquet, and redirecting HOME hides anything pip-installed with --user
# (python resolves the user site directory from $HOME), which would break its
# pandas import for no reason.
# ── gate: is this dataset fit to train on? ────────────────────────────────────
# groot_preflight.py's checks apply here too: G2 absolute actions on disk (the
# relative wrapper converts at load time and must not be fed a converted set),
# G3 the 10-D xyz + rot6d + gripper layout, G4 cameras, G5 task strings (which
# multi_task_dit REQUIRES), G7 finite values. G1 is GR00T's: with no --exclude
# it only warns when action names exist, but it still FAILS a dataset that has
# no per-dimension action names, which this policy does not need — override
# with SKIP_PREFLIGHT=1 if that is the only failure. --chunk-size is the number
# of steps executed per inference, for G6's timing report.
if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
    if [ -f "$PREFLIGHT" ]; then
        echo "==> groot_preflight.py $DATASET  (shared dataset checks; G1 is GR00T-specific)"
        if ! python "$PREFLIGHT" "$DATASET" --exclude --chunk-size "$N_ACTION_STEPS"; then
            echo >&2
            echo "Preflight FAILED — not starting training." >&2
            echo "SKIP_PREFLIGHT=1 ./train_multi_task_dit_server.sh to override." >&2
            exit 1
        fi
        echo
    else
        echo "WARNING: $PREFLIGHT not found; skipping the dataset gate." >&2
    fi
fi

export HOME="$WORKDIR/.home"
export HF_HOME="$HOME/.cache/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$HF_LEROBOT_HOME" "$TORCH_HOME" "$WORKDIR/output/train"

# ── gate: is the CLIP model reachable? ───────────────────────────────────────
# multi_task_dit downloads its image AND text encoders from
# openai/clip-vit-base-patch16 when the policy is built, after the dataset has
# loaded. The repo is public, so a failure here is almost always no egress.
if [ "${SKIP_HUB_CHECK:-0}" -eq 0 ]; then
    echo "==> checking hub access for the CLIP encoder"
    if ! python - <<'PY'
import sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("  huggingface_hub not installed; skipping"); sys.exit(0)
repo = "openai/clip-vit-base-patch16"
try:
    hf_hub_download(repo, "config.json")
    print(f"  OK      {repo}")
except Exception as exc:
    print(f"  FAIL    {repo}: {type(exc).__name__}: {str(exc)[:200]}")
    print()
    print("  The repo is public, so this is most likely no egress from this node.")
    print("  Pre-download on a networked host, copy HF_HOME across, then")
    print("  SKIP_HUB_CHECK=1 HF_HUB_OFFLINE=1.")
    sys.exit(1)
PY
    then
        echo >&2
        echo "Hub check FAILED — not starting training." >&2
        echo "SKIP_HUB_CHECK=1 ./train_multi_task_dit_server.sh to override (e.g. fully cached offline)." >&2
        exit 1
    fi
    echo
fi

# ── the command ──────────────────────────────────────────────────────────────
CMD=(python "$TRAINER")
if [ "$ABSOLUTE" = "0" ]; then
    if [ "$WRT_START" = "1" ]; then CMD+=(--wrt-start); else CMD+=(--no-wrt-start); fi
fi
# Consumed by the launcher before lerobot parses the rest (lerobot itself would
# reject it). STEPS and SAVE_FREQ are still passed below: with EPOCHS they set
# how many checkpoints come before the epoch target.
if [ -n "$EPOCHS" ]; then
    CMD+=(--epochs "$EPOCHS")
fi

CMD+=(
    --dataset.repo_id="$DATASET_NAME"
    --dataset.root="$DATASET"
    --policy.type=multi_task_dit
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.horizon="$HORIZON"
    --policy.n_action_steps="$N_ACTION_STEPS"
    --policy.objective="$OBJECTIVE"
    --policy.image_resize_shape="[$IMAGE_RESIZE,$IMAGE_RESIZE]"
    --policy.image_crop_shape="[224,224]"
)

if [ "$OBJECTIVE" = "diffusion" ]; then
    # DDIM samples well in few steps; the training objective is unchanged.
    CMD+=(
        --policy.noise_scheduler_type=DDIM
        --policy.num_inference_steps="$INFERENCE_STEPS"
    )
else
    CMD+=(--policy.num_integration_steps="$INFERENCE_STEPS")
fi

if [ -n "$WARMUP_STEPS" ]; then
    CMD+=(--policy.scheduler_warmup_steps="$WARMUP_STEPS")
fi

# Augmentation runs on training frames only: lerobot passes no transforms to
# the held-out split, and none are applied at deploy.
if [ "$AUGMENT" != "0" ]; then
    CMD+=(--dataset.image_transforms.enable=true)
fi
if [ "$AUGMENT" = "noaffine" ]; then
    # lerobot 0.6.1's default transforms MINUS "affine". The WHOLE list must be
    # given: `--dataset.image_transforms.tfs.affine.weight=0` is rejected, and a
    # JSON with only the affine entry REPLACES the list, leaving 0 transforms.
    IMAGE_TFS='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, '\
'"contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, '\
'"saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, '\
'"hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}, '\
'"sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}}}'
    CMD+=(--dataset.image_transforms.tfs="$IMAGE_TFS")
fi

CMD+=(
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --num_workers="$NUM_WORKERS"
    --output_dir="$OUT"
    --wandb.enable=false
)
if [ $# -gt 0 ]; then
    CMD+=("$@")
fi

printf '==> '; printf '%q ' "${CMD[@]}"; printf '\n\n'
if [ "$DRY_RUN" -ne 0 ]; then
    echo "(dry run — nothing launched)"
    exit 0
fi

# Which flags a run used is the first question asked of a checkpoint that
# misbehaves. Write it BESIDE the output directory, never inside: lerobot
# refuses to start when output_dir already exists ("already exists and resume
# is False"), so creating it here to hold this file is exactly what stops the
# run. Only the PARENT gets created.
LAUNCH_LOG="${OUT}.launch_command.txt"
mkdir -p "$(dirname "$OUT")"
{ printf '%q ' "${CMD[@]}"; printf '\n'; } > "$LAUNCH_LOG"

if [ -n "$EPOCHS" ]; then
    echo "==> Scheduled by EPOCHS=$EPOCHS: the [epochs] lines near the start list"
    echo "    every checkpoint step. The one marked '<- epoch target' is the"
    echo "    ${EPOCHS}-epoch model; the two marked '<- extra' run past it."
    echo
fi

# Tee to a log beside the run. lerobot only creates output_dir when it writes
# its first checkpoint, so a run that dies before then leaves NOTHING behind —
# and on a server the traceback is gone with the scrollback. This keeps it.
RUN_LOG="${OUT}.log"
echo "==> logging to $RUN_LOG"
echo

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e

# If lerobot got as far as writing a checkpoint it created $OUT, so the records
# can live with it. Otherwise they stay beside it, which is the whole point.
if [ -d "$OUT" ]; then
    [ -f "$LAUNCH_LOG" ] && mv -f "$LAUNCH_LOG" "$OUT/launch_command.txt"
    [ -f "$RUN_LOG" ]    && mv -f "$RUN_LOG"    "$OUT/train.log"
fi

if [ $status -eq 0 ]; then
    echo
    echo "==> Finished. Deploy with crisp_gym/config/policy/:"
    if [ "$ABSOLUTE" = "1" ]; then
        echo "      absolute_lerobot_policy.yaml   # poses as recorded"
    else
        echo "      relative_lerobot_policy.yaml   # RelativeLerobotPolicy composes T_base @ T_rel"
        if [ "$WRT_START" = "1" ]; then
            echo "      state_input: \"relative_wrt_start\"   # 16-D [rel_pose9, gripper1, wrt_start6]"
        else
            echo "      state_input: \"relative\"             # 10-D plain relative state"
        fi
    fi
    echo
    echo "    The deploy env must publish the SAME task string as the dataset:"
    echo "    multi_task_dit fails on a frame without one. Its inference speed is"
    echo "    fixed by OBJECTIVE / INFERENCE_STEPS above; deploy-time"
    echo "    --num-inference-steps / --scheduler do not apply to this policy."
else
    echo
    echo "==> FAILED (exit $status). No output directory means lerobot never"
    echo "    reached its first checkpoint save. The traceback is in:"
    echo "      ${RUN_LOG}"
fi
exit $status
