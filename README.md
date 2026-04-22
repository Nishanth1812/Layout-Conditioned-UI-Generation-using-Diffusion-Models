# Layout-Conditioned-UI-Generation-using-Diffusion-Models

This repo now includes training workflows for a Linux ROCm VM and a Modal A10G launcher.

## Quick Start On The VM

From the repository root:

```bash
bash scripts/setup_vm.sh
bash scripts/preprocess_data.sh
bash scripts/run_training.sh
```

## Quick Start On Modal

Use [main.py](main.py) with the Modal CLI to launch training on an A10G GPU. The launcher defaults to a 20,000-sample training cap, `bf16` precision, and the existing `/data/ui-gen` scratch layout so it matches the repo's preprocessing output.

Create a Modal secret for Kaggle credentials (run once):

```bash
modal secret create "Kaggle Secret" KAGGLE_USERNAME=YOUR_USERNAME KAGGLE_KEY=YOUR_KEY
```

Run the Windows launcher script. It downloads the dataset from Kaggle directly inside Modal, preprocesses it inside Modal, then starts detached training:

```bash
powershell -ExecutionPolicy Bypass -File .\run_modal_pipeline.ps1 -KaggleDataset "owner/dataset-slug" -SecretName "Kaggle Secret"
```

If your Kaggle dataset contains multiple zip files, choose one explicitly:

```bash
powershell -ExecutionPolicy Bypass -File .\run_modal_pipeline.ps1 -KaggleDataset "owner/dataset-slug" -KaggleFile "archive.zip" -SecretName "Kaggle Secret"
```

The detached training step logs Kaggle download, extraction, preprocessing, and training progress inside Modal.

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
