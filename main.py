from __future__ import annotations

import argparse
import base64
import io
import logging
import json
import os
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


APP_NAME = "layout-conditioned-ui-training"
DATA_VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "layout-ui-training-data")
SOURCE_VOLUME_NAME = os.environ.get("MODAL_SOURCE_VOLUME_NAME", DATA_VOLUME_NAME)
KAGGLE_SECRET_NAME = os.environ.get("MODAL_KAGGLE_SECRET_NAME", "Kaggle_Secret")
DEFAULT_SCRATCH_ROOT = "/data/ui-gen"
SOURCE_SCRATCH_ROOT = os.environ.get("MODAL_SOURCE_SCRATCH_ROOT", "/data/ui-gen-source")
DEFAULT_USABLE_LAYOUT_LIMIT = 30000
DEFAULT_TRAIN_SAMPLE_LIMIT = DEFAULT_USABLE_LAYOUT_LIMIT
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PREPROCESS_MANIFEST_NAME = "preprocessing_manifest.json"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _build_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "torchvision", "torchaudio", extra_index_url="https://download.pytorch.org/whl/cu121")
        .pip_install_from_requirements("requirements.txt")
        .add_local_dir("dataset", "/root/dataset")
        .add_local_dir("training", "/root/training")
        .add_local_dir("models", "/root/models")
    )


app = modal.App(APP_NAME)
image = _build_image()
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=True)


def _apply_root_paths(config: dict[str, Any]) -> None:
    raw_root = str(config.get("raw_root") or DEFAULT_SCRATCH_ROOT)
    processed_root = str(config.get("processed_root") or DEFAULT_SCRATCH_ROOT)

    config["raw_root"] = raw_root
    config["processed_root"] = processed_root
    config["download_dir"] = f"{processed_root}/downloads"
    config["train_json"] = f"{processed_root}/data/processed/splits/train.json"
    config["val_json"] = f"{processed_root}/data/processed/splits/val.json"
    config["tensor_dir"] = f"{processed_root}/data/processed/tensors"
    config["latent_dir"] = f"{processed_root}/data/processed/latents"
    config["image_dir"] = f"{raw_root}/data/raw/images"
    config["raw_json_dir"] = f"{raw_root}/data/raw/json"
    config["layouts_json"] = f"{processed_root}/data/processed/layouts/all_layouts.json"
    config["captioned_json"] = f"{processed_root}/data/processed/layouts/all_layouts_with_captions.json"
    config["filtered_json"] = f"{processed_root}/data/processed/layouts/all_layouts_filtered.json"
    config["skipped_layouts_file"] = f"{processed_root}/data/processed/layouts/skipped_layouts.json"
    config["split_dir"] = f"{processed_root}/data/processed/splits"
    config["preprocess_manifest"] = f"{processed_root}/data/processed/{PREPROCESS_MANIFEST_NAME}"


def _load_config(raw_data: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "raw_root": DEFAULT_SCRATCH_ROOT,
        "processed_root": DEFAULT_SCRATCH_ROOT,
        "output_dir": f"{DEFAULT_SCRATCH_ROOT}/checkpoints",
        "epochs": 20,
        "batch_size": 8,
        "lr": 8e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "num_workers": 8,
        "grad_accum_steps": 1,
        "precision": "bf16",
        "usable_layout_limit": DEFAULT_USABLE_LAYOUT_LIMIT,
        "train_sample_limit": DEFAULT_TRAIN_SAMPLE_LIMIT,
        "sample_seed": 42,
        "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "hf_token": None,
        "save_unet_in_checkpoint": False,
        "max_checkpoints": 2,
        "log_every": 10,
        "preprocess_workers": 8,
        "force_preprocess": False,
        "reuse_preprocessed": True,
        "use_latent_cache": True,
        "latent_precompute_batch_size": 8,
        "skip_training": False,
        "download_from_kaggle": False,
        "kaggle_dataset": "",
        "kaggle_file": "",
        "kaggle_username": "",
        "kaggle_key": "",
        "clear_raw_before_kaggle": True,
        "cleanup_downloads": True,
        "resume_from": None,
        "max_train_hours": 0.0,
    }

    if raw_data:
        if isinstance(raw_data, str) and raw_data.startswith("base64:"):
            encoded = raw_data[len("base64:") :]
            raw_data = base64.b64decode(encoded.encode("ascii")).decode("utf-8")

        override = json.loads(raw_data)
        if not isinstance(override, dict):
            raise TypeError("Modal payload must decode to a JSON object")
        config.update(override)

    usable_layout_limit = max(1, int(config.get("usable_layout_limit", DEFAULT_USABLE_LAYOUT_LIMIT) or DEFAULT_USABLE_LAYOUT_LIMIT))
    config["usable_layout_limit"] = usable_layout_limit
    config["train_sample_limit"] = min(
        usable_layout_limit,
        max(1, int(config.get("train_sample_limit", DEFAULT_TRAIN_SAMPLE_LIMIT) or DEFAULT_TRAIN_SAMPLE_LIMIT)),
    )
    _apply_root_paths(config)
    return config


def _path_has_files(path: str, suffixes: set[str] | None = None) -> bool:
    directory = Path(path)
    if not directory.exists():
        return False

    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue
        if suffixes is None or candidate.suffix.lower() in suffixes:
            return True
    return False


def _count_files(path: str, suffix: str) -> int:
    directory = Path(path)
    if not directory.exists():
        return 0
    return sum(1 for candidate in directory.rglob(f"*{suffix}") if candidate.is_file())


def _read_json_file(path: str) -> dict[str, Any] | list[Any] | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latent_precompute_collate(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    image_ids, images = zip(*batch)
    return list(image_ids), torch.stack(images, dim=0)


class _LatentPrecomputeDataset(Dataset):
    def __init__(self, items: list[dict[str, str]], image_size: int = 512):
        self.items = items
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        record = self.items[index]
        image = Image.open(record["image_path"]).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return record["image_id"], tensor


def _manifest_matches(config: dict[str, Any]) -> bool:
    manifest = _read_json_file(str(config["preprocess_manifest"]))
    if not isinstance(manifest, dict):
        logger.info("Preprocess manifest missing or invalid at %s", config["preprocess_manifest"])
        return False

    expected_seed = int(config["sample_seed"])
    expected_limit = int(config["usable_layout_limit"])
    if int(manifest.get("sample_seed", -1)) != expected_seed:
        logger.info("Preprocess manifest seed mismatch: expected=%s found=%s", expected_seed, manifest.get("sample_seed"))
        return False
    if int(manifest.get("usable_layout_limit", -1)) != expected_limit:
        logger.info("Preprocess manifest limit mismatch: expected=%s found=%s", expected_limit, manifest.get("usable_layout_limit"))
        return False

    tensor_count = _count_files(str(config["tensor_dir"]), ".npy")
    latent_count = _count_files(str(config["latent_dir"]), ".npy")
    split_dir = Path(str(config["split_dir"]))
    train_count = len(_read_json_file(str(config["train_json"])) or [])
    val_count = len(_read_json_file(str(config["val_json"])) or [])
    test_path = split_dir / "test.json"
    test_count = len(_read_json_file(str(test_path)) or [])

    expected_train = int(manifest.get("train_count", -1))
    expected_val = int(manifest.get("val_count", -1))
    expected_test = int(manifest.get("test_count", -1))
    expected_tensors = int(manifest.get("tensor_count", -1))
    expected_latents = int(manifest.get("latent_count", -1))
    expected_filtered = int(manifest.get("filtered_count", -1))

    if expected_tensors not in (-1, tensor_count):
        logger.info("Tensor cache mismatch: expected=%s found=%s", expected_tensors, tensor_count)
        return False
    if config.get("use_latent_cache") and expected_latents not in (-1, latent_count):
        logger.info("Latent cache mismatch: expected=%s found=%s", expected_latents, latent_count)
        return False
    if expected_filtered not in (-1, tensor_count):
        logger.info("Filtered layouts mismatch: expected=%s found=%s", expected_filtered, tensor_count)
        return False
    if expected_train not in (-1, train_count):
        logger.info("Train split mismatch: expected=%s found=%s", expected_train, train_count)
        return False
    if expected_val not in (-1, val_count):
        logger.info("Val split mismatch: expected=%s found=%s", expected_val, val_count)
        return False
    if expected_test not in (-1, test_count):
        logger.info("Test split mismatch: expected=%s found=%s", expected_test, test_count)
        return False

    logger.info(
        "Preprocess cache validated: seed=%s usable_layout_limit=%s layouts=%s filtered=%s tensors=%s latents=%s train=%s val=%s test=%s",
        expected_seed,
        expected_limit,
        manifest.get("layouts_count"),
        manifest.get("filtered_count"),
        manifest.get("tensor_count"),
        manifest.get("latent_count"),
        manifest.get("train_count"),
        manifest.get("val_count"),
        manifest.get("test_count"),
    )
    return True


def _dataset_is_ready(config: dict[str, Any]) -> bool:
    required_paths = [
        config["layouts_json"],
        config["captioned_json"],
        config["filtered_json"],
        config["train_json"],
        config["val_json"],
    ]
    if not all(Path(path).is_file() for path in required_paths):
        return False
    if not _path_has_files(str(config["tensor_dir"]), {".npy"}):
        return False
    if config.get("use_latent_cache") and not _path_has_files(str(config["latent_dir"]), {".npy"}):
        return False
    if not _path_has_files(str(config["raw_json_dir"]), {".json"}):
        return False
    if not _path_has_files(str(config["image_dir"]), IMAGE_EXTENSIONS):
        return False
    return _manifest_matches(config)


def _precompute_latents(config: dict[str, Any]) -> None:
    from diffusers import AutoencoderKL
    from training.dataset import _build_image_index, _resolve_image_path

    filtered = _read_json_file(str(config["filtered_json"]))
    if not isinstance(filtered, list):
        raise TypeError(f"Expected filtered layouts list at {config['filtered_json']}")

    latent_dir = Path(str(config["latent_dir"]))
    latent_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(str(config["image_dir"]))
    image_index = _build_image_index(image_dir)

    resolved_items: list[dict[str, str]] = []
    skipped_images = 0
    for layout in filtered:
        image_id = layout.get("image_id")
        if not image_id:
            skipped_images += 1
            continue
        image_path = _resolve_image_path(image_dir, image_id, image_index)
        if image_path is None:
            logger.warning("Skipping latent precompute for %s: missing image", image_id)
            skipped_images += 1
            continue
        resolved_items.append({"image_id": image_id, "image_path": str(image_path)})

    if skipped_images:
        logger.info("Latent precompute will skip %s layouts with missing or invalid images", skipped_images)

    batch_size = max(1, int(config.get("latent_precompute_batch_size", 8) or 8))
    worker_count = max(0, int(config.get("preprocess_workers", 8) or 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pretrained_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if config.get("hf_token"):
        pretrained_kwargs["token"] = config["hf_token"]

    logger.info(
        "Loading VAE for latent precompute on %s using %s worker(s) and batch_size=%s",
        device,
        worker_count,
        batch_size,
    )
    vae = AutoencoderKL.from_pretrained(config["base_model"], subfolder="vae", **pretrained_kwargs)
    vae.to(device)
    vae.eval()

    dataset = _LatentPrecomputeDataset(resolved_items)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": _latent_precompute_collate,
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    saved = 0
    total = len(dataset)

    logger.info("Precomputing cached latents for %s layouts into %s", total, latent_dir)
    for index, (image_ids, images) in enumerate(tqdm(loader, desc="Precompute latents", unit="batch", leave=True), start=1):
        batch = images.to(device=device, dtype=dtype, non_blocking=True)
        batch = (batch * 2.0) - 1.0
        with torch.inference_mode():
            latents = vae.encode(batch).latent_dist.sample() * 0.18215
        latents = latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for latent, image_id in zip(latents, image_ids):
            latent_path = latent_dir / f"{image_id}.npy"
            np.save(latent_path, latent.numpy())
            saved += 1
        if index == 1 or index % 50 == 0 or saved == total:
            logger.info("Latent precompute progress: %s/%s batches processed, %s/%s saved", index, len(loader), saved, total)

    data_volume.commit()
    logger.info("Latent precompute complete: %s cached tensors written to %s", saved, latent_dir)


def _preprocess_dataset(config: dict[str, Any]) -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from dataset.caption_generator import generate_all_captions
    from dataset.extract_layout import run_extraction
    from dataset.filter_and_balance import run_filter_balance
    from dataset.precompute import create_splits
    from dataset.tensor_encoder import process_all_layouts

    raw_json_dir = Path(str(config["raw_json_dir"]))
    layouts_json = Path(str(config["layouts_json"]))
    captioned_json = Path(str(config["captioned_json"]))
    filtered_json = Path(str(config["filtered_json"]))
    skipped_layouts_file = Path(str(config["skipped_layouts_file"]))
    tensor_dir = Path(str(config["tensor_dir"]))
    split_dir = Path(str(config["split_dir"]))

    logger.info(
        "Preprocessing dataset from raw json=%s and raw images=%s",
        config["raw_json_dir"],
        config["image_dir"],
    )

    layouts_json.parent.mkdir(parents=True, exist_ok=True)
    captioned_json.parent.mkdir(parents=True, exist_ok=True)
    filtered_json.parent.mkdir(parents=True, exist_ok=True)
    skipped_layouts_file.parent.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    run_extraction(
        str(raw_json_dir),
        str(layouts_json),
        str(skipped_layouts_file),
        num_workers=int(config["preprocess_workers"]),
        max_kept=int(config["usable_layout_limit"]),
        sample_seed=int(config["sample_seed"]),
    )
    generate_all_captions(str(layouts_json), str(captioned_json))
    run_filter_balance(
        str(captioned_json),
        str(filtered_json),
        max_layouts=int(config["usable_layout_limit"]),
        sample_seed=int(config["sample_seed"]),
    )
    process_all_layouts(
        str(filtered_json),
        str(tensor_dir),
        num_workers=int(config["preprocess_workers"]),
    )
    if config.get("use_latent_cache"):
        logger.info("Precomputing latent cache into %s", config["latent_dir"])
        _precompute_latents(config)
    create_splits(str(filtered_json), str(split_dir))

    manifest = {
        "usable_layout_limit": int(config["usable_layout_limit"]),
        "sample_seed": int(config["sample_seed"]),
        "layouts_count": len(_read_json_file(str(layouts_json)) or []),
        "captioned_count": len(_read_json_file(str(captioned_json)) or []),
        "filtered_count": len(_read_json_file(str(filtered_json)) or []),
        "tensor_count": _count_files(str(tensor_dir), ".npy"),
        "latent_count": _count_files(str(config["latent_dir"]), ".npy"),
        "train_count": len(_read_json_file(str(split_dir / "train.json")) or []),
        "val_count": len(_read_json_file(str(split_dir / "val.json")) or []),
        "test_count": len(_read_json_file(str(split_dir / "test.json")) or []),
    }
    preprocess_manifest = Path(str(config["preprocess_manifest"]))
    preprocess_manifest.parent.mkdir(parents=True, exist_ok=True)
    with preprocess_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    data_volume.commit()

    logger.info(
        "Preprocessing complete: layouts=%s captions=%s filtered=%s tensors=%s latents=%s splits=%s manifest=%s",
        layouts_json,
        captioned_json,
        filtered_json,
        tensor_dir,
        Path(str(config["latent_dir"])),
        split_dir,
        preprocess_manifest,
    )


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _member_relative_path(member_name: str, kind: str) -> Path | None:
    parts = [part for part in Path(member_name).parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return None

    lowered = [part.lower() for part in parts]
    if kind in lowered:
        index = lowered.index(kind)
        trailing = parts[index + 1 :]
        if trailing:
            return Path(*trailing)
        return Path(parts[-1])

    return Path(*parts)


def _extract_archive_to_raw(zip_path: Path, raw_root: Path) -> dict[str, int]:
    image_root = raw_root / "images"
    json_root = raw_root / "json"
    image_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "json": 0, "other": 0}
    logger.info("Extracting Kaggle archive %s into %s", zip_path, raw_root)
    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        progress = tqdm(members, desc=f"Extracting {zip_path.name}", unit="file", leave=True)
        for index, member in enumerate(progress, start=1):
            suffix = Path(member.filename).suffix.lower()

            if suffix in IMAGE_EXTENSIONS:
                relative = _member_relative_path(member.filename, "images")
                if relative is None:
                    counts["other"] += 1
                    continue
                target = image_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                counts["images"] += 1
            elif suffix == ".json":
                relative = _member_relative_path(member.filename, "json")
                if relative is None:
                    counts["other"] += 1
                    continue
                target = json_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                counts["json"] += 1
            else:
                counts["other"] += 1

            progress.set_postfix(images=counts["images"], json=counts["json"], skipped=counts["other"])
            if index == 1 or index % 500 == 0 or index == len(members):
                logger.info(
                    "Kaggle extract progress: %s/%s files processed (images=%s, json=%s, skipped=%s)",
                    index,
                    len(members),
                    counts["images"],
                    counts["json"],
                    counts["other"],
                )

    return counts


def _download_dataset_from_kaggle(config: dict[str, Any]) -> dict[str, int]:
    dataset = str(config.get("kaggle_dataset") or "").strip()
    if not dataset:
        raise ValueError("Set kaggle_dataset when download_from_kaggle=true")

    if config.get("kaggle_username"):
        os.environ["KAGGLE_USERNAME"] = str(config["kaggle_username"])
    if config.get("kaggle_key"):
        os.environ["KAGGLE_KEY"] = str(config["kaggle_key"])

    from kaggle.api.kaggle_api_extended import KaggleApi

    download_dir = Path(str(config["download_dir"]))
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Kaggle dataset %s into %s", dataset, download_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(download_dir), force=True, quiet=False, unzip=False)

    archive_name = str(config.get("kaggle_file") or "").strip()
    if archive_name:
        zip_path = download_dir / archive_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Configured kaggle_file not found after download: {zip_path}")
    else:
        zip_candidates = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zip_candidates:
            raise FileNotFoundError(f"No zip files downloaded from Kaggle dataset {dataset} into {download_dir}")
        zip_path = zip_candidates[0]

    raw_root = Path(str(config["raw_root"]))
    if config.get("clear_raw_before_kaggle", True):
        _clear_directory(raw_root / "images")
        _clear_directory(raw_root / "json")

    counts = _extract_archive_to_raw(zip_path, raw_root)
    data_volume.commit()

    if config.get("cleanup_downloads", True):
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove downloaded archive %s", zip_path)

    logger.info(
        "Kaggle download + extract complete (images=%s, json=%s, skipped=%s)",
        counts["images"],
        counts["json"],
        counts["other"],
    )
    return counts


def _build_command(config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "training.run_train",
        "--train-json",
        str(config["train_json"]),
        "--val-json",
        str(config.get("val_json") or ""),
        "--tensor-dir",
        str(config["tensor_dir"]),
        "--image-dir",
        str(config["image_dir"]),
        "--output-dir",
        str(config["output_dir"]),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--max-grad-norm",
        str(config["max_grad_norm"]),
        "--num-workers",
        str(config["num_workers"]),
        "--grad-accum-steps",
        str(config["grad_accum_steps"]),
        "--precision",
        str(config["precision"]),
        "--train-sample-limit",
        str(config["train_sample_limit"]),
        "--sample-seed",
        str(config["sample_seed"]),
        "--base-model",
        str(config["base_model"]),
        "--log-every",
        str(config["log_every"]),
        "--max-train-hours",
        str(config["max_train_hours"]),
    ]

    if config.get("hf_token"):
        command.extend(["--hf-token", str(config["hf_token"])])
    if config.get("resume_from"):
        command.extend(["--resume-from", str(config["resume_from"])])
    if config.get("save_unet_in_checkpoint"):
        command.append("--save-unet-in-checkpoint")

    return command


def _run_training(config: dict[str, Any]) -> None:
    from training.train import Trainer

    logger.info(
        "Launching trainer in-process: epochs=%s batch_size=%s num_workers=%s train_sample_limit=%s output_dir=%s",
        config["epochs"],
        config["batch_size"],
        config["num_workers"],
        config["train_sample_limit"],
        config["output_dir"],
    )
    trainer = Trainer(config)
    trainer.train(int(config["epochs"]))


def _upload_file_batch(batch: Any, source_file: Path, remote_path: str) -> None:
    batch.put_file(str(source_file), remote_path)


def _upload_directory_tree(batch: Any, source_root: Path, remote_root: str) -> int:
    source_files = [source_file for source_file in source_root.rglob("*") if source_file.is_file()]
    uploaded = 0
    progress = tqdm(source_files, desc=f"Uploading {source_root.name}", unit="file", leave=True)
    for source_file in progress:
        relative_path = source_file.relative_to(source_root).as_posix()
        remote_path = f"{remote_root.rstrip('/')}/{relative_path}"
        _upload_file_batch(batch, source_file, remote_path)
        uploaded += 1
        progress.set_postfix(uploaded=uploaded)
    return uploaded


def _upload_mixed_dataset(batch: Any, source_root: Path, remote_root: str) -> dict[str, int]:
    image_root = f"{remote_root.rstrip('/')}/images"
    json_root = f"{remote_root.rstrip('/')}/json"
    counts = {"images": 0, "json": 0}

    source_files = [source_file for source_file in source_root.rglob("*") if source_file.is_file()]
    progress = tqdm(source_files, desc=f"Uploading {source_root.name}", unit="file", leave=True)
    for source_file in progress:
        suffix = source_file.suffix.lower()
        relative_path = source_file.relative_to(source_root).as_posix()
        parts = Path(relative_path).parts
        if parts and parts[0].lower() in {"images", "json"}:
            relative_path = Path(*parts[1:]).as_posix() if len(parts) > 1 else source_file.name
        if suffix in IMAGE_EXTENSIONS:
            remote_path = f"{image_root}/{relative_path}"
            _upload_file_batch(batch, source_file, remote_path)
            counts["images"] += 1
        elif suffix == ".json":
            remote_path = f"{json_root}/{relative_path}"
            _upload_file_batch(batch, source_file, remote_path)
            counts["json"] += 1

        progress.set_postfix(images=counts["images"], json=counts["json"])

    return counts


def _upload_zip_archive(batch: Any, source_zip: Path, remote_root: str) -> dict[str, int]:
    image_root = f"{remote_root.rstrip('/')}/images"
    json_root = f"{remote_root.rstrip('/')}/json"
    counts = {"images": 0, "json": 0, "other": 0}

    logger.info("Scanning ZIP archive %s", source_zip)
    with zipfile.ZipFile(source_zip) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        logger.info("ZIP contains %s files", len(members))
        progress = tqdm(members, desc=f"Uploading {source_zip.name}", unit="file", leave=True)
        for index, member in enumerate(progress, start=1):
            suffix = Path(member.filename).suffix.lower()
            parts = Path(member.filename).parts
            relative_path = Path(*parts[1:]).as_posix() if parts and parts[0].lower() in {"images", "json"} and len(parts) > 1 else Path(member.filename).name if len(parts) == 1 else Path(*parts).as_posix()

            if suffix in IMAGE_EXTENSIONS:
                remote_path = f"{image_root}/{relative_path}"
                batch.put_file(io.BytesIO(archive.read(member)), remote_path)
                counts["images"] += 1
            elif suffix == ".json":
                remote_path = f"{json_root}/{relative_path}"
                batch.put_file(io.BytesIO(archive.read(member)), remote_path)
                counts["json"] += 1
            else:
                counts["other"] += 1

            progress.set_postfix(images=counts["images"], json=counts["json"], skipped=counts["other"])
            if index == 1 or index % 500 == 0 or index == len(members):
                logger.info(
                    "ZIP upload progress: %s/%s files processed (images=%s, json=%s, skipped=%s)",
                    index,
                    len(members),
                    counts["images"],
                    counts["json"],
                    counts["other"],
                )

    return counts


def _upload_dataset_to_volume(
    source: str,
    remote_root: str,
    volume_name: str,
    force: bool,
    images_source: str = "",
    json_source: str = "",
) -> dict[str, int]:
    if not source and not images_source and not json_source:
        raise ValueError("Provide source, images_source, or json_source for the upload")

    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    logger.info(
        "Starting dataset upload to Modal volume %s with remote root %s",
        volume_name,
        remote_root,
    )

    upload_roots: list[tuple[Path, str, bool]] = []
    if images_source:
        upload_roots.append((Path(images_source).expanduser().resolve(), f"{remote_root.rstrip('/')}/images", False))
    if json_source:
        upload_roots.append((Path(json_source).expanduser().resolve(), f"{remote_root.rstrip('/')}/json", False))

    if not upload_roots:
        source_root = Path(source).expanduser().resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Source path not found: {source_root}")
        upload_roots.append((source_root, remote_root, True))

    total_files = 0
    with volume.batch_upload(force=force) as batch:
        for source_root, destination_root, classify_mixed in upload_roots:
            if not source_root.exists():
                raise FileNotFoundError(f"Source path not found: {source_root}")

            logger.info("Uploading from %s to %s", source_root, destination_root)
            if source_root.is_file():
                if source_root.suffix.lower() == ".zip":
                    counts = _upload_zip_archive(batch, source_root, destination_root)
                    total_files += counts["images"] + counts["json"]
                    continue
                remote_path = f"{destination_root.rstrip('/')}/{source_root.name}"
                _upload_file_batch(batch, source_root, remote_path)
                total_files += 1
                continue

            if classify_mixed:
                counts = _upload_mixed_dataset(batch, source_root, destination_root)
                total_files += counts["images"] + counts["json"]
            else:
                total_files += _upload_directory_tree(batch, source_root, destination_root)

    volume.commit()
    logger.info(
        "Uploaded %s file(s) into Modal volume '%s' under %s",
        total_files,
        volume_name,
        remote_root,
    )
    return {"files": total_files}


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    volumes={DEFAULT_SCRATCH_ROOT: data_volume},
)
def run_pipeline(data: str = "") -> dict[str, Any]:
    config = _load_config(data or None)
    repo_root = Path(__file__).resolve().parent

    data_volume.reload()
    logger.info("Reloaded Modal volume %s before preprocessing/training", DATA_VOLUME_NAME)

    if config.get("download_from_kaggle"):
        logger.info("download_from_kaggle=true; downloading dataset inside Modal")
        _download_dataset_from_kaggle(config)
        config["force_preprocess"] = True

    if config.get("reuse_preprocessed") and _dataset_is_ready(config):
        logger.info("Cached preprocessing artifacts match sample_seed=%s and usable_layout_limit=%s; reusing them", config["sample_seed"], config["usable_layout_limit"])
    elif config.get("force_preprocess") or not _dataset_is_ready(config):
        logger.info("Dataset artifacts are missing or stale; running preprocessing")
        _preprocess_dataset(config)
    else:
        logger.info("Dataset artifacts already present; skipping preprocessing")

    if config.get("skip_training"):
        data_volume.commit()
        logger.info("skip_training=true; preprocessing complete and volume committed")
        return {
            "status": "preprocessed",
            "usable_layout_limit": config["usable_layout_limit"],
            "train_sample_limit": config["train_sample_limit"],
            "output_dir": config["output_dir"],
            "gpu": "A10G",
        }

    logger.info(
        "Training inputs: train_json=%s tensor_dir=%s image_dir=%s output_dir=%s train_sample_limit=%s reusable_preprocess=%s",
        config["train_json"],
        config["tensor_dir"],
        config["image_dir"],
        config["output_dir"],
        config["train_sample_limit"],
        config.get("reuse_preprocessed"),
    )

    logger.info(
        "Starting Modal training on A10G: epochs=%s batch_size=%s num_workers=%s usable_layout_limit=%s train_sample_limit=%s output_dir=%s",
        config["epochs"],
        config["batch_size"],
        config["num_workers"],
        config["usable_layout_limit"],
        config["train_sample_limit"],
        config["output_dir"],
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
    env.setdefault("SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT)
    env.setdefault("LOG_FILE", str(Path(config["output_dir"]) / "train.log"))

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    _run_training(config)
    data_volume.commit()
    logger.info("Training run completed and volume committed")

    return {
        "status": "ok",
        "usable_layout_limit": config["usable_layout_limit"],
        "train_sample_limit": config["train_sample_limit"],
        "output_dir": config["output_dir"],
        "gpu": "A10G",
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name(KAGGLE_SECRET_NAME)],
    volumes={DEFAULT_SCRATCH_ROOT: data_volume},
)
def run_pipeline_kaggle(data: str = "") -> dict[str, Any]:
    config = _load_config(data or None)
    repo_root = Path(__file__).resolve().parent

    data_volume.reload()
    logger.info("Reloaded Modal volume %s before preprocessing/training", DATA_VOLUME_NAME)

    if config.get("download_from_kaggle"):
        logger.info("download_from_kaggle=true; downloading dataset inside Modal")
        _download_dataset_from_kaggle(config)
        config["force_preprocess"] = True

    if config.get("reuse_preprocessed") and _dataset_is_ready(config):
        logger.info("Cached preprocessing artifacts match sample_seed=%s and usable_layout_limit=%s; reusing them", config["sample_seed"], config["usable_layout_limit"])
    elif config.get("force_preprocess") or not _dataset_is_ready(config):
        logger.info("Dataset artifacts are missing or stale; running preprocessing")
        _preprocess_dataset(config)
    else:
        logger.info("Dataset artifacts already present; skipping preprocessing")

    if config.get("skip_training"):
        data_volume.commit()
        logger.info("skip_training=true; preprocessing complete and volume committed")
        return {
            "status": "preprocessed",
            "usable_layout_limit": config["usable_layout_limit"],
            "train_sample_limit": config["train_sample_limit"],
            "output_dir": config["output_dir"],
            "gpu": "A10G",
        }

    logger.info(
        "Training inputs: train_json=%s tensor_dir=%s image_dir=%s output_dir=%s train_sample_limit=%s reusable_preprocess=%s",
        config["train_json"],
        config["tensor_dir"],
        config["image_dir"],
        config["output_dir"],
        config["train_sample_limit"],
        config.get("reuse_preprocessed"),
    )

    logger.info(
        "Starting Modal training on A10G: epochs=%s batch_size=%s num_workers=%s usable_layout_limit=%s train_sample_limit=%s output_dir=%s",
        config["epochs"],
        config["batch_size"],
        config["num_workers"],
        config["usable_layout_limit"],
        config["train_sample_limit"],
        config["output_dir"],
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
    env.setdefault("SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT)
    env.setdefault("LOG_FILE", str(Path(config["output_dir"]) / "train.log"))

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    _run_training(config)
    data_volume.commit()
    logger.info("Training run completed and volume committed")

    return {
        "status": "ok",
        "usable_layout_limit": config["usable_layout_limit"],
        "train_sample_limit": config["train_sample_limit"],
        "output_dir": config["output_dir"],
        "gpu": "A10G",
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    volumes={SOURCE_SCRATCH_ROOT: source_volume, DEFAULT_SCRATCH_ROOT: data_volume},
)
def preprocess_existing_volume(data: str = "") -> dict[str, Any]:
    config = _load_config(data or None)
    config["raw_root"] = SOURCE_SCRATCH_ROOT
    config["processed_root"] = DEFAULT_SCRATCH_ROOT
    _apply_root_paths(config)

    source_volume.reload()
    data_volume.reload()
    logger.info(
        "Reloaded source volume %s and output volume %s for preprocess-only run",
        SOURCE_VOLUME_NAME,
        DATA_VOLUME_NAME,
    )

    if config.get("download_from_kaggle"):
        logger.info("download_from_kaggle=true; downloading dataset inside Modal")
        _download_dataset_from_kaggle(config)

    logger.info("Running preprocess-only pipeline from %s into %s", SOURCE_VOLUME_NAME, DATA_VOLUME_NAME)
    _preprocess_dataset(config)

    data_volume.commit()
    logger.info("Preprocess-only run completed and output volume committed")

    return {
        "status": "preprocessed",
        "usable_layout_limit": config["usable_layout_limit"],
        "train_sample_limit": config["train_sample_limit"],
        "output_dir": config["output_dir"],
        "source_volume": SOURCE_VOLUME_NAME,
        "output_volume": DATA_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(data: str = "") -> None:
    logger.info("Submitting Modal training run")
    print(run_pipeline.remote(data))


@app.local_entrypoint()
def main_kaggle(data: str = "") -> None:
    logger.info("Submitting Modal Kaggle training run")
    print(run_pipeline_kaggle.remote(data))


@app.local_entrypoint()
def upload_dataset(
    source: str = "",
    remote_root: str = f"{DEFAULT_SCRATCH_ROOT}/data/raw",
    volume_name: str = DATA_VOLUME_NAME,
    force: bool = False,
    images_source: str = "",
    json_source: str = "",
) -> None:
    _upload_dataset_to_volume(source, remote_root, volume_name, force, images_source, json_source)


@app.local_entrypoint()
def prepare_and_train(
    source: str = "",
    data: str = "",
    remote_root: str = f"{DEFAULT_SCRATCH_ROOT}/data/raw",
    volume_name: str = DATA_VOLUME_NAME,
    force: bool = False,
    images_source: str = "",
    json_source: str = "",
) -> None:
    logger.info("Preparing Modal run: upload first, then train in detached mode")
    _upload_dataset_to_volume(source, remote_root, volume_name, force, images_source, json_source)
    logger.info("Dataset upload finished; starting training")
    config = _load_config(data or None)
    config["force_preprocess"] = True
    print(run_pipeline.remote(json.dumps(config)))