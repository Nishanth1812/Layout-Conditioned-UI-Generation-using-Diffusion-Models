#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/ui-gen}"
LOG_FILE="${LOG_FILE:-$SCRATCH_ROOT/preprocess.log}"
SKIPPED_LAYOUTS_FILE="${SKIPPED_LAYOUTS_FILE:-$SCRATCH_ROOT/data/processed/layouts/skipped_layouts.json}"
CPU_CORES="${CPU_CORES:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)}"

if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi
cd "$REPO_DIR"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_CORES}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_CORES}"
export EXTRACT_WORKERS="${EXTRACT_WORKERS:-$CPU_CORES}"
export TENSOR_WORKERS="${TENSOR_WORKERS:-$CPU_CORES}"

echo "[preprocess] log file: $LOG_FILE"
echo "[preprocess] cpu cores: $CPU_CORES"
echo "[preprocess] extract workers: $EXTRACT_WORKERS"
echo "[preprocess] tensor workers: $TENSOR_WORKERS"
echo "[preprocess] extracting layouts"
python -u -c "import os; from dataset.extract_layout import run_extraction; run_extraction('${RAW_JSON_DIR:-$SCRATCH_ROOT/data/raw/json}', '${LAYOUTS_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts.json}', '${SKIPPED_LAYOUTS_FILE}', num_workers=int(os.environ.get('EXTRACT_WORKERS', '1')))"
echo "[preprocess] generating captions"
python -u -c "from dataset.caption_generator import generate_all_captions; generate_all_captions('${LAYOUTS_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts.json}', '${CAPTIONED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_with_captions.json}')"
echo "[preprocess] filtering and balancing"
python -u -c "from dataset.filter_and_balance import run_filter_balance; run_filter_balance('${CAPTIONED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_with_captions.json}', '${FILTERED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_filtered.json}')"
echo "[preprocess] encoding tensors"
python -u -c "import os; from dataset.tensor_encoder import process_all_layouts; process_all_layouts('${FILTERED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_filtered.json}', '${TENSOR_DIR:-$SCRATCH_ROOT/data/processed/tensors}', num_workers=int(os.environ.get('TENSOR_WORKERS', '1')))"
echo "[preprocess] creating splits"
python -u -c "from dataset.precompute import create_splits; create_splits('${FILTERED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_filtered.json}', '${SPLITS_DIR:-$SCRATCH_ROOT/data/processed/splits}')"
echo "[preprocess] skipped samples: $SKIPPED_LAYOUTS_FILE"
echo "[preprocess] done"
