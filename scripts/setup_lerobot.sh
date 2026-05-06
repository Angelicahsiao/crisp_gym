#!/usr/bin/env bash
# Clone lerobot v0.5.1 to a sibling directory and patch its pyproject.toml
# to allow Python 3.11. Required because lerobot v0.5.1 declares
# requires-python>=3.12 in metadata, but uses no 3.12-only syntax.
#
# Usage:
#   bash scripts/setup_lerobot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_DIR="$(cd "$REPO_DIR/.." && pwd)/lerobot"
LEROBOT_REV="v0.5.1"

if [ -d "$LEROBOT_DIR" ]; then
    echo "lerobot already exists at $LEROBOT_DIR — skipping clone."
else
    if ! command -v git &>/dev/null; then
        echo "Error: git not found. Please install git or manually clone lerobot v0.5.1 to $LEROBOT_DIR"
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

if grep -q 'requires-python = ">=3.12"' "$PYPROJECT"; then
    echo "Patching $PYPROJECT: requires-python >=3.12 → >=3.11"
    sed -i.bak 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' "$PYPROJECT"
    echo "Patch applied."
else
    echo "$PYPROJECT already patched or has unexpected requires-python value — no change needed."
fi

echo ""
echo "Done. Now run:"
echo "  pixi install -e humble-lerobot"
