#!/usr/bin/env bash
set -euo pipefail

# --path is a bind mount, so anchor to this script's own directory rather than
# to whatever cwd the platform sets. Everything written under here persists.
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

# ── GR00T N1.7 fine-tune (server / bind-mount variant) ───────────────────────
#
# The env-var sibling of scripts/train_groot.sh. Same gates, same pinned flags;
# this one is driven by environment variables and defaults every path off its
# OWN directory, for a training box where the script sits in a bind mount and
# the platform decides the cwd. Use train_groot.sh when you want to pass flags
# and choose paths per run.
#
# LAYOUT — everything is relative to WORKDIR, the directory holding this file:
#
#   $WORKDIR/
#   ├── train_groot_server.sh          this file
#   ├── crisp_gym/                     a CLONE of the crisp_gym repo
#   │   ├── scripts/check_trained_groot.sh
#   │   └── crisp_gym/scripts/
#   │       ├── groot_preflight.py     the dataset gate (warn + continue if absent)
#   │       └── lerobot_relative_pose.py   SE3=1 only (hard fail if absent)
#   ├── datasets/angelica/<DATASET_NAME>/  meta/ + data/, or .../lerobot/meta
#   ├── output/train/                  runs land here; created below
#   └── .home/.cache/huggingface/      ~10 GB of base checkpoint; needs 20 GB free
#
#   DATASET=/full/path and OUT=/full/path escape this layout entirely (e.g. a
#   dataset or a run directory on an NFS mount).
#
# HUGGING FACE TOKEN — export it, do not rely on a login:
#
#   export HF_TOKEN=hf_...
#
#   This script redirects HOME into the bind mount so the ~10 GB download
#   survives container restarts. huggingface-cli writes its token to
#   $HF_HOME/token, so a login done under your REAL home becomes invisible the
#   moment HF_HOME moves, and the gated backbone answers 401 anonymously.
#   HF_TOKEN is read before HF_HOME and sidesteps that. Only
#   nvidia/Cosmos-Reason2-2B is gated; accept its terms with the same account.
#
# NOT lerobot_relative_pose.py by default. That wrapper converts the dataset to
# relative poses, which is right for diffusion and SmolVLA and WRONG here: GR00T
# relativizes the ACTION itself, from ABSOLUTE actions and state
# (GrootN17PackInputsStep caches the raw state; GrootN17ActionDecodeStep adds it
# back). For a finetune that conversion is COMPONENTWISE SUBTRACTION, not SE(3):
# the synthesized `new_embodiment` action config is hardcoded NON_EEF/DEFAULT,
# and relative_eef_to_absolute -- the one SE(3) path -- runs only for type "eef"
# with format "xyz+rot6d". Either way, feeding it a pre-converted dataset makes
# it learn deltas of deltas, and nothing errors. So this calls lerobot's own
# trainer on the RAW recording.
#
# Three lerobot defaults are wrong for a rot6d dataset and each fails silently:
#   use_relative_actions=False  -> decodes relative against absolute statistics
#   relative_exclude_joints=[]  -> trains the gripper as a delta
#   push_to_hub=True            -> fails late, after the dataset has loaded
# All three are pinned below. groot_preflight.py gates the run before any of it.
#
#   ./train_groot_server.sh                          # uses the defaults below
#   SE3=1 ./train_groot_server.sh                    # CROSS-EMBODIMENT mode, see below
#   SE3=1 WRT_START=0 ./train_groot_server.sh        # ... with a 10-D relative state
#   STEPS=2000 ./train_groot_server.sh               # smoke test
#   BATCH_SIZE=16 ./train_groot_server.sh            # H200 has room; 32 is the default
#   DATASET_NAME=other_set ./train_groot_server.sh
#   OUT=/mnt/nfs_share/runs/x ./train_groot_server.sh
#   SKIP_PREFLIGHT=1 ./train_groot_server.sh         # bypass the gate
#   SKIP_HUB_CHECK=1 ./train_groot_server.sh         # bypass the hub gate
#   DRY_RUN=1 ./train_groot_server.sh                # print the command, launch nothing
#
# Anything passed on the command line is appended to the lerobot command.

DATASET_NAME="${DATASET_NAME:-open_electribox_sum}"
DATASET="${DATASET:-$WORKDIR/datasets/angelica/$DATASET_NAME}"
PREFLIGHT="$WORKDIR/crisp_gym/crisp_gym/scripts/groot_preflight.py"
_mode_tag=""
[ "${SE3:-0}" -ne 0 ] && _mode_tag="_se3"
OUT="${OUT:-$WORKDIR/output/train/groot${_mode_tag}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"      # H200: 150 GB, room to spare. 32 GB card: try 2.
SAVE_FREQ="${SAVE_FREQ:-20000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
DRY_RUN="${DRY_RUN:-0}"

# SE3=1 trains through crisp_gym's UMI relative-pose wrapper instead of plain
# lerobot_train. The difference that matters for CROSS-EMBODIMENT: that wrapper
# re-expresses the OBSERVATION relative to the current frame (T_base^-1 @ T,
# lerobot_relative_pose.py:129), which neither GR00T path does -- in lerobot and
# in Isaac-GR00T alike the state is only ever a reference, never transformed.
# An absolute TCP pose is in the robot's own base frame, so the same motion is
# different numbers on a Franka and a UR; relative observations remove that.
#
# GR00T's own relative conversion is turned OFF here (use_relative_actions
# false): the wrapper has already made the action relative, and GR00T would
# otherwise subtract the state a second time. relative_exclude_joints is
# dropped for the same reason -- nothing for it to exclude from.
#
# The dataset on disk stays ABSOLUTE (the wrapper converts per sample, see its
# docstring line 17), so groot_preflight.py's checks apply unchanged.
SE3="${SE3:-0}"
# 1 -> 16-D observation.state [rel_pose9, gripper1, wrt_start6] (UMI parity)
# 0 -> 10-D plain relative state
# Deploy MUST match: state_input relative_wrt_start vs relative.
WRT_START="${WRT_START:-1}"

# Some datasets keep meta/ and data/ directly, others nest them under lerobot/.
if [ ! -d "$DATASET/meta" ] && [ -d "$DATASET/lerobot/meta" ]; then
    DATASET="$DATASET/lerobot"
fi
if [ ! -f "$DATASET/meta/info.json" ]; then
    echo "No meta/info.json under $DATASET" >&2
    echo "Set DATASET=/full/path/to/dataset or DATASET_NAME=<name>." >&2
    exit 2
fi

# The gate runs BEFORE the HOME redirect below, deliberately. It only reads
# local parquet, and redirecting HOME hides anything pip-installed with --user
# (python resolves the user site directory from $HOME), which would break its
# pandas import for no reason.
# ── gate: is this dataset actually fit for GR00T? ────────────────────────────
if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
    if [ -f "$PREFLIGHT" ]; then
        echo "==> groot_preflight.py $DATASET"
        if ! python "$PREFLIGHT" "$DATASET" --exclude gripper --chunk-size 40; then
            echo >&2
            echo "Preflight FAILED — not starting training." >&2
            echo "SKIP_PREFLIGHT=1 ./train_groot_server.sh to override." >&2
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

# GR00T N1.7 pulls ~10 GB (the 3B model plus the Cosmos-Reason2-2B backbone)
# into HF_HOME on the first run. Because HOME is redirected into the bind mount
# it survives container restarts, but the space has to be there the first time.
avail_gb="$(df -PBG "$HF_HOME" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}' || true)"
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt 20 ]; then
    echo "WARNING: only ${avail_gb} GB free at $HF_HOME; the base checkpoint needs ~10 GB." >&2
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
        echo "SKIP_HUB_CHECK=1 ./train_groot_server.sh to override (e.g. fully cached offline)." >&2
        exit 1
    fi
    echo
fi

if [ "$SE3" -ne 0 ]; then
    RELTRAINER="$WORKDIR/crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py"
    if [ ! -f "$RELTRAINER" ]; then
        echo "SE3=1 needs $RELTRAINER, which is missing." >&2
        exit 2
    fi
    CMD=(python "$RELTRAINER")
    if [ "$WRT_START" -ne 0 ]; then CMD+=(--wrt-start); else CMD+=(--no-wrt-start); fi
else
    CMD=(python -m lerobot.scripts.lerobot_train)
fi

CMD+=(
    --dataset.repo_id="$DATASET_NAME"
    --dataset.root="$DATASET"
    --policy.type=groot
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.embodiment_tag=new_embodiment
)

if [ "$SE3" -ne 0 ]; then
    # The wrapper already made the action relative; GR00T must not subtract the
    # state again. Nothing to exclude either -- the gripper dim is passed
    # through by the wrapper untouched.
    CMD+=(--policy.use_relative_actions=false)
else
    CMD+=(
        --policy.use_relative_actions=true
        --policy.relative_exclude_joints='["gripper"]'
    )
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
# misbehaves, and check_trained_groot.sh compares this against what the policy
# actually saved. Write it BESIDE the output directory, never inside: lerobot
# refuses to start when output_dir already exists (configs/train.py:259,
# "already exists and resume is False"), so creating it here to hold this file
# is exactly what stops the run. Only the PARENT gets created, as in train.sh.
LAUNCH_LOG="${OUT}.launch_command.txt"
mkdir -p "$(dirname "$OUT")"
{ printf '%q ' "${CMD[@]}"; printf '\n'; } > "$LAUNCH_LOG"

echo "==> Watch the log for:"
echo "      'using the generic RelativeActionsProcessorStep fallback'"
echo "    That step SUBTRACTS, which is wrong for rot6d. If it appears, stop:"
echo "    the checkpoint is not carrying native relative statistics."
echo

# Tee to a log beside the run. lerobot only creates output_dir when it writes
# its first checkpoint, so a run that dies before then leaves NOTHING behind --
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
    echo "==> Finished. Verify the flags survived the run:"
    echo "      bash crisp_gym/scripts/check_trained_groot.sh $OUT"
    echo
    echo "==> Deploy with config/policy/groot_lerobot_policy.yaml set to:"
    if [ "$SE3" -ne 0 ]; then
        if [ "$WRT_START" -ne 0 ]; then
            echo "      state_input: \"relative_wrt_start\"   # 16-D [rel_pose9, gripper1, wrt_start6]"
        else
            echo "      state_input: \"relative\"             # 10-D plain relative state"
        fi
        echo "      action_repr: \"relative\"             # the wrapper made the action relative;"
        echo "                                          # RelativeLerobotPolicy composes T_base @ T_rel"
        echo "      compose_mode: \"coupled\"             # body-frame, matching make_relative()"
        echo
        echo "    lerobot_relative_pose.py stamped pose_repr.json next to the"
        echo "    checkpoint, so state_input: \"auto\" resolves correctly too. There is"
        echo "    no action_repr.json, and auto falls back to relative -- also correct"
        echo "    HERE, unlike the non-SE3 mode where it would be wrong."
    else
        echo "      state_input: \"absolute\"             # GR00T is fed the state as recorded"
        echo "      action_repr: \"absolute\"             # GR00T decoded to absolute internally;"
        echo "                                          # \"auto\" would wrongly pick relative here"
    fi
else
    echo
    echo "==> FAILED (exit $status). No output directory means lerobot never"
    echo "    reached its first checkpoint save. The traceback is in:"
    echo "      ${RUN_LOG}"
fi
exit $status
