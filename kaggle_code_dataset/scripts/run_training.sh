#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/ui-gen}"

if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi
cd "$REPO_DIR"

echo "[train] starting training"
exec python -u -m training.run_train \
  --train-json "${TRAIN_JSON:-$SCRATCH_ROOT/data/processed/splits/train.json}" \
  --val-json "${VAL_JSON:-$SCRATCH_ROOT/data/processed/splits/val.json}" \
  --tensor-dir "${TENSOR_DIR:-$SCRATCH_ROOT/data/processed/tensors}" \
  --image-dir "${IMAGE_DIR:-$SCRATCH_ROOT/data/raw/images}" \
  --output-dir "${OUTPUT_DIR:-$SCRATCH_ROOT/checkpoints}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-1e-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --precision "${PRECISION:-fp16}" \
  --base-model "${BASE_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}" \
  --hf-token "${HF_TOKEN:-}" \
  --log-every "${LOG_EVERY:-50}"
