#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on a crisp_gym recording, with the traps pinned shut.
#
# WHY A LAUNCHER
#   Three of lerobot's GR00T defaults are wrong for a rot6d pose dataset, and
#   each fails silently:
#
#     use_relative_actions=False    The base N1.7 checkpoint declares
#         use_relative_action=True in its processor_kwargs. The DECODE step
#         follows the checkpoint, while relative STATISTICS are computed only
#         when this config flag is set. Leave it False and the model decodes
#         relative actions against absolute stats.
#
#     relative_exclude_joints=[]    Every action dim is treated as relative,
#         gripper included. The exclusion is matched against per-DIMENSION
#         action names from info.json, so it also needs a dataset that carries
#         them -- see groot_preflight.py's G1.
#
#     push_to_hub=True              PreTrainedConfig's default. A local run
#         with no repo_id fails late, after the dataset has loaded.
#
#   This script pins all three and refuses to start if the dataset is not fit,
#   so a bad run costs seconds instead of GPU-hours.
#
# IT ALSO REFUSES a relative-converted dataset. GR00T composes its own relative
#   actions from absolute ones; feeding it a lerobot_relative_pose.py output
#   makes it learn deltas of deltas, and nothing errors.
#
# USAGE
#   bash scripts/train_groot.sh --dataset datasets/angelica/open_electribox_sum \
#                               --output outputs/groot_electricbox
#
#   --dataset DIR      dataset root holding meta/ and data/   (required)
#   --output DIR       training output directory              (required)
#   --repo-id ID       dataset repo id (default: last two path components)
#   --batch-size N     default 32; the H200 has room, a 32 GB card does not
#   --steps N          default 100000
#   --job-name NAME    default: the output directory's name
#   --save-freq N      default 20000
#   --num-workers N    default 8
#   --skip-preflight   run without gating on groot_preflight.py
#   --dry-run          print the command and exit
#   --                 everything after this is appended to the lerobot command
#
# EXIT CODES
#   0 training finished    1 preflight failed or training failed    2 bad usage

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

DATASET="" OUTPUT="" REPO_ID="" JOB_NAME=""
BATCH_SIZE=32 STEPS=100000 SAVE_FREQ=20000 NUM_WORKERS=8
SKIP_PREFLIGHT=0 DRY_RUN=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)        DATASET="${2:-}"; shift 2 ;;
        --output)         OUTPUT="${2:-}"; shift 2 ;;
        --repo-id)        REPO_ID="${2:-}"; shift 2 ;;
        --batch-size)     BATCH_SIZE="${2:-}"; shift 2 ;;
        --steps)          STEPS="${2:-}"; shift 2 ;;
        --job-name)       JOB_NAME="${2:-}"; shift 2 ;;
        --save-freq)      SAVE_FREQ="${2:-}"; shift 2 ;;
        --num-workers)    NUM_WORKERS="${2:-}"; shift 2 ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --)               shift; EXTRA=("$@"); break ;;
        -h|--help)        sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)                echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$DATASET" ] || [ -z "$OUTPUT" ]; then
    echo "usage: bash scripts/train_groot.sh --dataset DIR --output DIR [options]" >&2
    echo "       (--help for the full list)" >&2
    exit 2
fi
if [ ! -d "$DATASET" ]; then
    echo "dataset directory not found: $DATASET" >&2
    exit 2
fi

DATASET="$(cd "$DATASET" && pwd)"
[ -n "$REPO_ID" ] || REPO_ID="$(basename "$(dirname "$DATASET")")/$(basename "$DATASET")"
[ -n "$JOB_NAME" ] || JOB_NAME="$(basename "$OUTPUT")"

# ── gate: is this dataset fit for GR00T at all? ──────────────────────────────
if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
    echo "==> groot_preflight.py $DATASET"
    if ! python3 "$REPO/crisp_gym/scripts/groot_preflight.py" "$DATASET" \
             --exclude gripper --chunk-size 40; then
        echo
        echo "Preflight FAILED — not starting training." >&2
        echo "Fix the dataset, or re-run with --skip-preflight if you know better." >&2
        exit 1
    fi
    echo
fi

# ── the command ──────────────────────────────────────────────────────────────
CMD=(
    python3 -m lerobot.scripts.lerobot_train
    --dataset.repo_id="$REPO_ID"
    --dataset.root="$DATASET"
    --policy.type=groot
    --policy.use_relative_actions=true
    --policy.relative_exclude_joints='["gripper"]'
    --policy.embodiment_tag=new_embodiment
    --policy.push_to_hub=false
    --output_dir="$OUTPUT"
    --job_name="$JOB_NAME"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --num_workers="$NUM_WORKERS"
)
[ ${#EXTRA[@]} -gt 0 ] && CMD+=("${EXTRA[@]}")

printf '==> '; printf '%q ' "${CMD[@]}"; printf '\n\n'

if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry run — nothing launched)"
    exit 0
fi

# Keep the exact invocation with the run. Which flags a run used is the first
# question asked of a checkpoint that behaves oddly, and
# check_trained_groot.sh compares this against what the policy actually saved.
#
# BESIDE the output directory, never inside it. lerobot refuses to start when
# output_dir already exists (configs/train.py:259, "already exists and resume
# is False"), so creating it here to hold this file is precisely what stops the
# run. Only the parent is created; the file is moved in afterwards.
LAUNCH_LOG="${OUTPUT}.launch_command.txt"
mkdir -p "$(dirname "$OUTPUT")"
{ printf '%q ' "${CMD[@]}"; printf '\n'; } > "$LAUNCH_LOG"

echo "==> Watch the log for:"
echo "      'using the generic RelativeActionsProcessorStep fallback'"
echo "    That step SUBTRACTS, which is wrong for rot6d. If it appears, stop:"
echo "    the checkpoint is not carrying native relative statistics."
echo

"${CMD[@]}"
status=$?

# lerobot has created the directory by now, so the record can live with the
# checkpoints where check_trained_groot.sh and a future reader will find it.
if [ -d "$OUTPUT" ] && [ -f "$LAUNCH_LOG" ]; then
    mv -f "$LAUNCH_LOG" "$OUTPUT/launch_command.txt"
fi

if [ $status -eq 0 ]; then
    echo
    echo "==> Training finished. Verify the flags survived:"
    echo "      bash scripts/check_trained_groot.sh $OUTPUT"
fi
exit $status
