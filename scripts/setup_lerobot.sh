#!/usr/bin/env bash
# Set up lerobot for crisp_gym.
#
# Default: clone lerobot v0.4.4 (Python 3.11 + ROS2 Humble compatible).
# To use a different version, set LEROBOT_REV before running, e.g.:
#   LEROBOT_REV=v0.5.1 bash scripts/setup_lerobot.sh
#
# v0.5.1 requires Python 3.12 + numpy 2.x and won't work with ROS2 Humble.
# This script automatically applies relaxation patches if you select v0.5.1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_DIR="$(cd "$REPO_DIR/.." && pwd)/lerobot"
LEROBOT_REV="${LEROBOT_REV:-v0.4.4}"

if [ -d "$LEROBOT_DIR" ]; then
    echo "lerobot already exists at $LEROBOT_DIR — skipping clone."
    echo "  (delete this directory if you want to re-clone at a different rev)"
else
    if ! command -v git &>/dev/null; then
        echo "Error: git not found. Please install git or manually clone lerobot $LEROBOT_REV to $LEROBOT_DIR"
        exit 1
    fi
    echo "Cloning lerobot $LEROBOT_REV to $LEROBOT_DIR..."
    git clone --branch "$LEROBOT_REV" --depth 1 \
        https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

PYPROJECT="$LEROBOT_DIR/pyproject.toml"
if [ ! -f "$PYPROJECT" ]; then
    echo "Error: $PYPROJECT not found. Is $LEROBOT_DIR a valid lerobot checkout?"
    exit 1
fi

# Patches only needed if running v0.5.1 on Python 3.11 + Humble.
# v0.4.4 needs no patches — it already supports Python >=3.10.
if grep -q 'requires-python = ">=3.12"' "$PYPROJECT"; then
    echo "Patching $PYPROJECT: requires-python >=3.12 → >=3.11"
    sed -i.bak 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' "$PYPROJECT"
fi

if grep -qE '"numpy>=2\.0\.0' "$PYPROJECT"; then
    echo "Patching $PYPROJECT: numpy >=2.0.0 → >=1.26.0"
    sed -i.bak2 's/"numpy>=2\.0\.0/"numpy>=1.26.0/g' "$PYPROJECT"
fi

# rerun-sdk >=0.24.0 requires numpy>=2, conflicting with ROS2 Humble's
# numpy==1.26.4 pin. crisp_gym does not use rerun; the only lerobot path
# that uses it is the standalone visualize_dataset.py script — not the
# recording or inference pipelines. Drop the dependency.
if grep -qE '^\s*"rerun-sdk' "$PYPROJECT"; then
    echo "Patching $PYPROJECT: removing rerun-sdk dependency"
    sed -i.bak3 '/^\s*"rerun-sdk[^"]*",\?\s*$/d' "$PYPROJECT"
fi

echo ""
echo "Done. Now run:"
echo "  rm -f pixi.lock"
echo "  pixi install -e humble-lerobot"
