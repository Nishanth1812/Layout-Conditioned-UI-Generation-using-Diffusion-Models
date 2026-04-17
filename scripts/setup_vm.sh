#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/ui-gen}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROCM_WHEEL="${ROCM_WHEEL:-rocm6.2}"

echo "Repo dir: $REPO_DIR"
echo "Scratch root: $SCRATCH_ROOT"
echo "Venv dir: $VENV_DIR"
echo "PyTorch ROCm wheel: $ROCM_WHEEL"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python binary '$PYTHON_BIN' was not found."
  exit 1
fi

if ! command -v rocminfo >/dev/null 2>&1; then
  echo "ROCm does not appear to be installed yet. Install ROCm on the VM first, then rerun this script."
  exit 1
fi

mkdir -p \
  "$SCRATCH_ROOT/data/raw/images" \
  "$SCRATCH_ROOT/data/raw/json" \
  "$SCRATCH_ROOT/data/processed/layouts" \
  "$SCRATCH_ROOT/data/processed/tensors" \
  "$SCRATCH_ROOT/data/processed/splits" \
  "$SCRATCH_ROOT/checkpoints"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$ROCM_WHEEL"
python -m pip install -r "$REPO_DIR/requirements.txt"

cat <<EOF

Environment setup complete.

Place raw screenshots under:
  $SCRATCH_ROOT/data/raw/images

Place raw layout JSON files under:
  $SCRATCH_ROOT/data/raw/json

Then run preprocessing and training from the repo root.
EOF
