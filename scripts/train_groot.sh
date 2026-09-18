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

# ── gate: are BOTH hub repos reachable with this token? ──────────────────────
# GR00T pulls from two repos: the 3B policy and its Cosmos-Reason2-2B backbone
# (configuration_groot.py:44-45). A failure on either surfaces deep in training
# as "failed to get .../main/config", long after the dataset has loaded.
#
# The HOME redirect above is a live hazard here: huggingface-cli writes its
# token to $HF_HOME/token, so a login done under your real HOME is invisible
# once HF_HOME points into the bind mount, and gated NVIDIA repos then answer
# 401/403 anonymously. Export HF_TOKEN before running this script, or log in
# once with HF_HOME set to the path above.
if [ "${SKIP_HUB_CHECK:-0}" -eq 0 ]; then
    echo "==> checking hub access for the base model and its backbone"
    if ! python - <<'PY'
import os, sys
try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("  huggingface_hub not installed; skipping"); sys.exit(0)

# Whether a token is reaching the hub at all decides which remedy applies, and
# HF_TOKEN wins over HF_HOME, so it sidesteps the redirect above entirely.
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
try:
    who = HfApi().whoami()
    source = "HF_TOKEN" if token else os.environ.get("HF_HOME", "~/.cache/huggingface") + "/token"
    print(f"  token   authenticated as {who.get('name', '?')}  (from {source})")
    authed = True
except Exception:
    print("  token   NONE — requests are anonymous")
    authed = False

bad = []
for repo in ("nvidia/GR00T-N1.7-3B", "nvidia/Cosmos-Reason2-2B"):
    try:
        hf_hub_download(repo, "config.json")
        print(f"  OK      {repo}")
    except Exception as exc:
        text = str(exc)
        code = "401" if "401" in text else "403" if "403" in text else "?"
        bad.append((repo, code))
        print(f"  FAIL    {repo}: {type(exc).__name__} {code}")

if bad:
    print()
    if any(c == "401" for _, c in bad):
        print("  401 = UNAUTHENTICATED. The repo is gated and no valid token was sent.")
        print("    1. accept the terms on the model page (a click-through, needs an account)")
        print("    2. create a read token at https://huggingface.co/settings/tokens")
        print("    3. export HF_TOKEN=hf_...   (beats HF_HOME, so the redirect cannot hide it)")
    if any(c == "403" for _, c in bad):
        print("  403 = AUTHENTICATED BUT NOT PERMITTED. The token works; this account has")
        print("    not been granted the repo. Accept the terms on the model page with THIS")
        print("    account, or ask whoever owns the org for access.")
    if any(c == "?" for _, c in bad):
        print("  Neither 401 nor 403 — likely no egress from this node. Pre-download on a")
        print("    networked host, copy HF_HOME across, then SKIP_HUB_CHECK=1 HF_HUB_OFFLINE=1.")
    if not authed:
        print()
        print("  Note: nvidia/GR00T-N1.7-3B is public; only the Cosmos-Reason2-2B backbone")
        print("  is gated, so a token is needed even though the policy repo resolves fine.")
    sys.exit(1)
PY
    then
        echo >&2
        echo "Hub check FAILED — not starting training." >&2
        echo "SKIP_HUB_CHECK=1 bash scripts/train_groot.sh to override (e.g. fully cached offline)." >&2
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

# Tee to a log beside the run. lerobot only creates output_dir when it writes
# its first checkpoint, so a run that dies before then leaves nothing behind and
# the traceback goes with the scrollback. This keeps it.
RUN_LOG="${OUTPUT}.log"
echo "==> logging to $RUN_LOG"
echo

"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}

# lerobot has created the directory by now, so the record can live with the
# checkpoints where check_trained_groot.sh and a future reader will find it.
if [ -d "$OUTPUT" ]; then
    [ -f "$LAUNCH_LOG" ] && mv -f "$LAUNCH_LOG" "$OUTPUT/launch_command.txt"
    [ -f "$RUN_LOG" ]    && mv -f "$RUN_LOG"    "$OUTPUT/train.log"
fi

if [ $status -eq 0 ]; then
    echo
    echo "==> Training finished. Verify the flags survived:"
    echo "      bash scripts/check_trained_groot.sh $OUTPUT"
else
    echo
    echo "==> FAILED (exit $status). No output directory means lerobot never"
    echo "    reached its first checkpoint save. The traceback is in:"
    echo "      ${RUN_LOG}"
fi
exit $status
