#!/usr/bin/env bash
# Set up lerobot for crisp_gym.
#
# Default: clone lerobot v0.6.1, which is what pixi.toml's humble-lerobot env
# expects (Python 3.12 + numpy 2). It needs NO patches: its
# opencv-python-headless bound (<4.14) accepts the conda-provided opencv, and
# pixi.toml pins numpy/packaging to lerobot's own ranges.
#
#   bash scripts/setup_lerobot.sh
#
# To clone a different rev:
#   LEROBOT_REV=v0.4.4 bash scripts/setup_lerobot.sh
#
# NOTE for old revs: lerobot 0.4.x has NO "dataset" extra (it was added in
# 0.6.x), so pixi.toml's `extras = ["dataset"]` will not resolve against it.
# Pin the lerobot dependency back at the same time.
#
# The numpy-1 / Python-3.11 relaxation patches are OFF by default and must be
# requested explicitly:
#
#   LEROBOT_PATCH_NUMPY1=1 bash scripts/setup_lerobot.sh
#
# They rewrite the CLONE's requires-python 3.12 -> 3.11 and numpy >=2.0 ->
# >=1.26, and drop rerun-sdk. That is only correct when the pixi env is itself
# on Python 3.11 + numpy 1.26. Running them against the current numpy-2 env
# silently undoes the very requirements it was migrated to — which is why they
# are no longer applied automatically on a version grep.
#
# (Historical note: this header used to claim "v0.5.1 requires Python 3.12 +
# numpy 2.x and won't work with ROS2 Humble". That is false — robostack-humble
# resolves against numpy 2 once ros-humble-image-transport-plugins is dropped,
# which is what pixi.toml now does.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_DIR="$(cd "$REPO_DIR/.." && pwd)/lerobot"
LEROBOT_REV="${LEROBOT_REV:-v0.6.1}"
LEROBOT_PATCH_NUMPY1="${LEROBOT_PATCH_NUMPY1:-0}"
# 1 = shallow (fast). 0 = full history, so you can `git diff v0.4.4 v0.6.1`
# while porting against lerobot internals.
LEROBOT_CLONE_DEPTH="${LEROBOT_CLONE_DEPTH:-1}"

if [ -d "$LEROBOT_DIR" ]; then
    existing="$(git -C "$LEROBOT_DIR" describe --tags --exact-match 2>/dev/null \
             || git -C "$LEROBOT_DIR" rev-parse --short HEAD 2>/dev/null \
             || echo "unknown")"
    echo "lerobot already exists at $LEROBOT_DIR (checked out: $existing) — skipping clone."
    if [ "$existing" != "$LEROBOT_REV" ]; then
        echo ""
        echo "  WARNING: the existing checkout is '$existing', NOT '$LEROBOT_REV'."
        echo "  The env is installed from THIS directory, so pixi will use"
        echo "  '$existing' no matter what pixi.toml's comments say. To switch:"
        echo "      rm -rf $LEROBOT_DIR && LEROBOT_REV=$LEROBOT_REV bash $0"
        echo ""
    fi
else
    if ! command -v git &>/dev/null; then
        echo "Error: git not found. Please install git or manually clone lerobot $LEROBOT_REV to $LEROBOT_DIR"
        exit 1
    fi
    echo "Cloning lerobot $LEROBOT_REV to $LEROBOT_DIR..."
    if [ "$LEROBOT_CLONE_DEPTH" = "0" ]; then
        git clone --branch "$LEROBOT_REV" \
            https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
    else
        git clone --branch "$LEROBOT_REV" --depth "$LEROBOT_CLONE_DEPTH" \
            https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
    fi
fi

PYPROJECT="$LEROBOT_DIR/pyproject.toml"
if [ ! -f "$PYPROJECT" ]; then
    echo "Error: $PYPROJECT not found. Is $LEROBOT_DIR a valid lerobot checkout?"
    exit 1
fi

# pixi.toml requests extras = ["dataset"]; that extra only exists from 0.6.x.
# Without it, `datasets`/`torchcodec`/`av` are absent and
# lerobot/datasets/__init__.py raises at import — recording cannot start.
if ! grep -qE '^\s*dataset\s*=\s*\[' "$PYPROJECT"; then
    echo ""
    echo "  WARNING: this checkout declares no 'dataset' extra (pre-0.6.x)."
    echo "  pixi.toml requests extras = [\"dataset\"] and will fail to resolve."
    echo "  Either use v0.6.1, or drop the extras from pixi.toml's lerobot entry."
    echo ""
fi

if [ "$LEROBOT_PATCH_NUMPY1" = "1" ]; then
    echo "LEROBOT_PATCH_NUMPY1=1 — applying Python 3.11 / numpy 1.26 relaxation patches."

    if grep -q 'requires-python = ">=3.12"' "$PYPROJECT"; then
        echo "Patching $PYPROJECT: requires-python >=3.12 → >=3.11"
        sed -i.bak 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' "$PYPROJECT"
    fi

    if grep -qE '"numpy>=2\.0\.0' "$PYPROJECT"; then
        echo "Patching $PYPROJECT: numpy >=2.0.0 → >=1.26.0"
        sed -i.bak2 's/"numpy>=2\.0\.0/"numpy>=1.26.0/g' "$PYPROJECT"
    fi

    # rerun-sdk >=0.24.0 requires numpy>=2, which conflicts with a numpy-1.26
    # env. crisp_gym does not use rerun; the only lerobot path that does is the
    # standalone visualize_dataset.py script — not recording or inference.
    if grep -qE '^\s*"rerun-sdk' "$PYPROJECT"; then
        echo "Patching $PYPROJECT: removing rerun-sdk dependency"
        sed -i.bak3 '/^\s*"rerun-sdk[^"]*",\?\s*$/d' "$PYPROJECT"
    fi
else
    echo "Skipping the numpy-1 / Python-3.11 patches (default)."
    echo "  The current pixi.toml env is Python 3.12 + numpy 2, which needs none."
    echo "  Set LEROBOT_PATCH_NUMPY1=1 only when the env is on numpy 1.26."
fi

echo ""
echo "Done. Now run:"
echo "  rm -f pixi.lock"
echo "  pixi install -e humble-lerobot"
