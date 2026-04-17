# Layout-Conditioned-UI-Generation-using-Diffusion-Models

This repo now includes a minimal Linux ROCm training workflow for an AMD GPU VM.

## Quick Start On The VM

From the repository root:

```bash
bash scripts/setup_vm.sh
bash scripts/preprocess_data.sh
bash scripts/run_training.sh
```

## What Each Script Does

- `scripts/setup_vm.sh`: creates the scratch directory layout, builds a virtual environment, and installs the Python dependencies plus the ROCm PyTorch wheel.
- `scripts/preprocess_data.sh`: extracts layouts, generates captions, filters the dataset, encodes tensors, and creates train/val/test splits.
- `scripts/run_training.sh`: launches the training CLI with sensible defaults that match the scratch-disk layout from the setup guide.

## Important Notes

- ROCm must already be installed on the Linux VM before `scripts/setup_vm.sh` is run.
- Raw screenshots should be placed in `/mnt/scratch/ui-gen/data/raw/images`.
- Raw UI JSON files should be placed in `/mnt/scratch/ui-gen/data/raw/json`.
- You can override paths and hyperparameters by exporting environment variables before running the scripts.
- Training now defaults to the public Stable Diffusion v1.5 mirror at `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- If you want to use the gated `runwayml/stable-diffusion-v1-5` checkpoint instead, export `HF_TOKEN` before training.
