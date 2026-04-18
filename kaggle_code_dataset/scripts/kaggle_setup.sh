#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/kaggle/working/ui-gen}"

mkdir -p \
  "$SCRATCH_ROOT/data/raw/images" \
  "$SCRATCH_ROOT/data/raw/json" \
  "$SCRATCH_ROOT/data/processed/layouts" \
  "$SCRATCH_ROOT/data/processed/tensors" \
  "$SCRATCH_ROOT/data/processed/splits" \
  "$SCRATCH_ROOT/checkpoints"

cd "$REPO_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cat <<EOF
Kaggle setup complete.

Working directory:
  $SCRATCH_ROOT

Suggested data layout:
  $SCRATCH_ROOT/data/raw/images
  $SCRATCH_ROOT/data/raw/json

If your data lives under /kaggle/input, copy or symlink it into the paths above
before running preprocessing.
EOF
