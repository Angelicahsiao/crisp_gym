#!/usr/bin/env bash
set -euo pipefail

# Anchor to this script's own directory, like train_multi_task_dit_server.sh.
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

# ── point-track targets: build the training copy, then train multi_task_dit ──
#
# The last two steps of the point-track pipeline. The first two are done by
# hand before this: click_track_points.py (clicks) and track_points.py (tracks).
#
#   1. build_track_dataset.py writes a COPY of DATASET whose action also
#      carries the tracks: [pose 9, (x, y, visible) per point, gripper]. The
#      source is not changed. Skipped when the copy exists already.
#   2. train_multi_task_dit_server.sh trains on the copy with exactly the
#      settings it uses for any dataset. lerobot_relative_pose.py finds the
#      copy's meta/track_targets.json and weights the track dims by
#      TRACK_LOSS_WEIGHT, leaving hidden points' positions out of the loss.
#
# LAYOUT — this file and train_multi_task_dit_server.sh side by side, next to
# a crisp_gym clone (copy both out of crisp_gym/scripts/):
#
#   $WORKDIR/
#   ├── train_tracks_server.sh
#   ├── train_multi_task_dit_server.sh
#   └── crisp_gym/                       the clone
#
# PATHS — only DATASET is required; the rest default to the layout the
# point-track scripts' examples use:
#
#   DATASET=/workspace/datasets/franka_electricbox/lerobot      the source (clicked + tracked)
#   TRACKS=.../franka_electricbox/tracks                        track_points.py --out
#   QUERIES=.../franka_electricbox/track_queries.json           click_track_points.py --out
#   TRACK_DATASET=.../franka_electricbox_tracks/lerobot         the copy to build / train on
#
# TRACK SETTINGS
#
#   TRACK_LOSS_WEIGHT=0.25     weight of the track dims (pose and gripper keep 1);
#                              empty = the launcher's default (0.25)
#   NO_TRACK_LOSS_PATCH=1      weight every action dim equally instead
#   BUILD_ONLY=1               build the copy, do not train
#   DRY_RUN=1                  print both commands, run neither
#
# Every other variable of train_multi_task_dit_server.sh (STEPS, EPOCHS,
# BATCH_SIZE, HORIZON, N_ACTION_STEPS, IMAGE_RESIZE, OBJECTIVE, AUGMENT, OUT,
# ...) passes straight through, and command-line arguments are appended to the
# training command as there. ABSOLUTE=1 is refused: only the relative launcher
# applies the track-aware loss.
#
#   DATASET=/workspace/datasets/franka_electricbox/lerobot ./train_tracks_server.sh
#   DATASET=... EPOCHS=200 ./train_tracks_server.sh
#   DATASET=... TRACK_LOSS_WEIGHT=0.1 ./train_tracks_server.sh
#   DATASET=... BUILD_ONLY=1 ./train_tracks_server.sh
#
# For a comparison without tracks, train the SOURCE with the same settings:
#   DATASET=/workspace/datasets/franka_electricbox/lerobot ./train_multi_task_dit_server.sh

fail() { echo "$*" >&2; exit 2; }

DATASET="${DATASET:-}"
[ -n "$DATASET" ] || fail "Set DATASET=/path/to/the/source/dataset (the one you clicked and tracked)."
if [ ! -d "$DATASET/meta" ] && [ -d "$DATASET/lerobot/meta" ]; then
    DATASET="$DATASET/lerobot"
fi
[ -f "$DATASET/meta/info.json" ] || fail "No meta/info.json under $DATASET"
DATASET="$(cd "$DATASET" && pwd)"
BASE="$(dirname "$DATASET")"            # .../franka_electricbox
TRACKS="${TRACKS:-$BASE/tracks}"
QUERIES="${QUERIES:-$BASE/track_queries.json}"
TRACK_DATASET="${TRACK_DATASET:-$(dirname "$BASE")/$(basename "$BASE")_tracks/lerobot}"
TRACK_NAME="${TRACK_NAME:-$(basename "$BASE")_tracks}"

TRACK_LOSS_WEIGHT="${TRACK_LOSS_WEIGHT:-}"
NO_TRACK_LOSS_PATCH="${NO_TRACK_LOSS_PATCH:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

[ "${ABSOLUTE:-0}" = "0" ] || fail "ABSOLUTE=1 is not supported here: only the relative launcher applies the track-aware loss."
if [ -n "$TRACK_LOSS_WEIGHT" ] && ! awk -v w="$TRACK_LOSS_WEIGHT" 'BEGIN { exit !(w ~ /^[0-9]*\.?[0-9]+$/) }'; then
    fail "TRACK_LOSS_WEIGHT must be a non-negative number, got: $TRACK_LOSS_WEIGHT"
fi
[ "$NO_TRACK_LOSS_PATCH" = "0" ] || [ "$NO_TRACK_LOSS_PATCH" = "1" ] || fail "NO_TRACK_LOSS_PATCH must be 0 or 1"
if [ -n "$TRACK_LOSS_WEIGHT" ] && [ "$NO_TRACK_LOSS_PATCH" = "1" ]; then
    fail "TRACK_LOSS_WEIGHT and NO_TRACK_LOSS_PATCH=1 contradict each other: pick one."
fi

BUILDER="$WORKDIR/crisp_gym/crisp_gym/scripts/build_track_dataset.py"
TRAIN="$WORKDIR/train_multi_task_dit_server.sh"
[ -f "$BUILDER" ] || fail "Missing $BUILDER — clone crisp_gym beside this script (see LAYOUT)."
[ "$BUILD_ONLY" = "1" ] || [ -f "$TRAIN" ] || \
    fail "Missing $TRAIN — copy crisp_gym/scripts/train_multi_task_dit_server.sh beside this script."

# ── 1. build the copy ─────────────────────────────────────────────────────────
# The stamp is the last thing the builder writes, so a folder without it is an
# interrupted build: refuse rather than train on half a dataset.
if [ -f "$TRACK_DATASET/meta/track_targets.json" ]; then
    echo "==> $TRACK_DATASET exists already: not rebuilt (delete it to rebuild)."
    echo
elif [ -e "$TRACK_DATASET" ]; then
    fail "$TRACK_DATASET exists but has no meta/track_targets.json (an interrupted build?). Remove it and rerun."
else
    [ -d "$TRACKS" ] || fail "No tracks at $TRACKS — run track_points.py first, or set TRACKS=."
    BUILD=(python "$BUILDER" --root "$DATASET" --tracks "$TRACKS" --out "$TRACK_DATASET")
    if [ -f "$QUERIES" ]; then
        BUILD+=(--queries "$QUERIES")
    fi
    printf '==> '; printf '%q ' "${BUILD[@]}"; printf '\n\n'
    if [ "$DRY_RUN" -ne 0 ]; then
        echo "(dry run — not built; the training command can be shown once the copy exists)"
        exit 0
    else
        mkdir -p "$(dirname "$TRACK_DATASET")"
        BUILD_LOG="${TRACK_DATASET%/}.build.log"
        set +e
        "${BUILD[@]}" 2>&1 | tee "$BUILD_LOG"
        status=${PIPESTATUS[0]}
        set -e
        if [ $status -ne 0 ]; then
            echo >&2
            echo "==> Build FAILED (exit $status); log: $BUILD_LOG" >&2
            exit $status
        fi
        mv -f "$BUILD_LOG" "$TRACK_DATASET/build.log"
        echo
        echo "==> Built $TRACK_DATASET (log: $TRACK_DATASET/build.log)"
        echo
    fi
fi

if [ "$BUILD_ONLY" = "1" ]; then
    echo "==> BUILD_ONLY=1: not training."
    exit 0
fi

# ── 2. train on the copy ──────────────────────────────────────────────────────
TRACK_ARGS=()
if [ -n "$TRACK_LOSS_WEIGHT" ]; then
    TRACK_ARGS+=(--track-loss-weight "$TRACK_LOSS_WEIGHT")
fi
if [ "$NO_TRACK_LOSS_PATCH" = "1" ]; then
    TRACK_ARGS+=(--no-track-loss-patch)
fi
echo "==> Training on $TRACK_DATASET with train_multi_task_dit_server.sh."
echo "    Look for 'Track targets: ... loss weight ...' near the start of its log."
echo
DATASET="$TRACK_DATASET" DATASET_NAME="$TRACK_NAME" \
    exec bash "$TRAIN" ${TRACK_ARGS[@]+"${TRACK_ARGS[@]}"} "$@"
