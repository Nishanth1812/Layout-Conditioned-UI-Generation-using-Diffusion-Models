#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_dir() {
  local candidate
  local repo_url="${GIT_REPO_URL:-https://github.com/Nishanth1812/Layout-Conditioned-UI-Generation-using-Diffusion-Models.git}"
  local clone_root="${CLONE_ROOT:-/kaggle/working}"
  local clone_dir="${CLONE_DIR:-$clone_root/Layout-Conditioned-UI-Generation-using-Diffusion-Models}"

  if [[ -n "${REPO_DIR:-}" && -f "$REPO_DIR/training/run_train.py" ]]; then
    printf '%s\n' "$REPO_DIR"
    return 0
  fi

  for candidate in \
    "$SCRIPT_DIR/.." \
    "$PWD" \
    "/kaggle/working" \
    "/kaggle/input"
  do
    if [[ -f "$candidate/training/run_train.py" ]]; then
      (cd "$candidate" && pwd)
      return 0
    fi
  done

  while IFS= read -r candidate; do
    if [[ -f "$candidate/training/run_train.py" ]]; then
      (cd "$candidate" && pwd)
      return 0
    fi
  done < <(find /kaggle/working /kaggle/input -maxdepth 4 -type f -path '*/training/run_train.py' -printf '%h\n' 2>/dev/null | sed 's#/training$##' | sort -u)

  if command -v git >/dev/null 2>&1; then
    if [[ ! -f "$clone_dir/training/run_train.py" ]]; then
      mkdir -p "$clone_root"
      git clone "$repo_url" "$clone_dir" >/dev/null 2>&1 || true
    fi
    if [[ -f "$clone_dir/training/run_train.py" ]]; then
      (cd "$clone_dir" && pwd)
      return 0
    fi
  fi

  return 1
}

REPO_DIR="${REPO_DIR:-$(resolve_repo_dir || true)}"

if [[ -z "$REPO_DIR" || ! -f "$REPO_DIR/training/run_train.py" ]]; then
  echo "[train] could not locate repo root containing training/run_train.py" >&2
  echo "[train] set REPO_DIR explicitly or run from a directory that contains the repo" >&2
  exit 127
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-/kaggle/working/ui-gen}"
NUM_GPUS="${NUM_GPUS:-2}"
LOG_FILE="${LOG_FILE:-$SCRATCH_ROOT/train.log}"
CHECKPOINT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/checkpoints}"
LOG_MIRROR="${LOG_MIRROR:-$CHECKPOINT_DIR/train.log}"
CPU_CORES="${CPU_CORES:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)}"

cd "$REPO_DIR"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[train] torchrun was not found in PATH" >&2
  exit 127
fi

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

echo "[train] log file: $LOG_FILE"
echo "[train] log mirror: $LOG_MIRROR"
echo "[train] cpu cores: $CPU_CORES"
echo "[train] starting distributed training on ${NUM_GPUS} GPU(s)"
set +e
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m training.run_train \
  --train-json "${TRAIN_JSON:-$SCRATCH_ROOT/data/processed/splits/train.json}" \
  --val-json "${VAL_JSON:-$SCRATCH_ROOT/data/processed/splits/val.json}" \
  --tensor-dir "${TENSOR_DIR:-$SCRATCH_ROOT/data/processed/tensors}" \
  --image-dir "${IMAGE_DIR:-$SCRATCH_ROOT/data/raw/images}" \
  --output-dir "${OUTPUT_DIR:-$SCRATCH_ROOT/checkpoints}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE_PER_GPU:-1}" \
  --lr "${LR:-1e-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --precision "${PRECISION:-fp16}" \
  --base-model "${BASE_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}" \
  --hf-token "${HF_TOKEN:-}" \
  --log-every "${LOG_EVERY:-50}" \
  --max-train-hours "${MAX_TRAIN_HOURS:-0}" \
  --resume-from "${RESUME_FROM:-}"
EXIT_CODE=$?
set -e

if [[ -f "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_MIRROR")"
  cp -f "$LOG_FILE" "$LOG_MIRROR"
  echo "[train] copied log to $LOG_MIRROR"
fi

echo "[train] finished with exit code $EXIT_CODE"
exit "$EXIT_CODE"
