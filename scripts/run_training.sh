#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/ui-gen}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python -m training.run_train \
  --train-json "${TRAIN_JSON:-$SCRATCH_ROOT/data/processed/splits/train.json}" \
  --tensor-dir "${TENSOR_DIR:-$SCRATCH_ROOT/data/processed/tensors}" \
  --image-dir "${IMAGE_DIR:-$SCRATCH_ROOT/data/raw/images}" \
  --output-dir "${OUTPUT_DIR:-$SCRATCH_ROOT/checkpoints}" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --lr "${LR:-1e-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --precision "${PRECISION:-fp16}" \
  --base-model "${BASE_MODEL:-runwayml/stable-diffusion-v1-5}"
