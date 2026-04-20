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
  --epochs "${EPOCHS:-10}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-8e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --precision "${PRECISION:-bf16}" \
  --train-sample-limit "${TRAIN_SAMPLE_LIMIT:-10000}" \
  --sample-seed "${SAMPLE_SEED:-42}" \
  --base-model "${BASE_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}" \
  --hf-token "${HF_TOKEN:-}" \
  --log-every "${LOG_EVERY:-20}"
