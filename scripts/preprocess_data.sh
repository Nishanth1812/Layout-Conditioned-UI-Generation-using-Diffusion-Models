#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/ui-gen}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "[preprocess] extracting layouts"
python -c "from dataset.extract_layout import run_extraction; run_extraction('${RAW_JSON_DIR:-$SCRATCH_ROOT/data/raw/json}', '${LAYOUTS_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts.json}')"
echo "[preprocess] generating captions"
python -c "from dataset.caption_generator import generate_all_captions; generate_all_captions('${LAYOUTS_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts.json}', '${CAPTIONED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_with_captions.json}')"
echo "[preprocess] filtering and balancing"
python -c "from dataset.filter_and_balance import run_filter_balance; run_filter_balance('${CAPTIONED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_with_captions.json}', '${FILTERED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_filtered.json}')"
echo "[preprocess] encoding tensors"
python -c "from dataset.tensor_encoder import process_all_layouts; process_all_layouts('${FILTERED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_filtered.json}', '${TENSOR_DIR:-$SCRATCH_ROOT/data/processed/tensors}')"
echo "[preprocess] creating splits"
python -c "from dataset.precompute import create_splits; create_splits('${FILTERED_JSON:-$SCRATCH_ROOT/data/processed/layouts/all_layouts_filtered.json}', '${SPLITS_DIR:-$SCRATCH_ROOT/data/processed/splits}')"
echo "[preprocess] done"
