import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _build_image_index(image_dir):
    image_index = {}
    for processed, candidate in enumerate(image_dir.rglob("*"), start=1):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_index.setdefault(candidate.stem, candidate)
        if processed % 10000 == 0:
            logger.info("Indexed %s candidate files under %s", processed, image_dir)
    logger.info("Indexed %s image files under %s", len(image_index), image_dir)
    return image_index


def _resolve_image_path(image_dir, image_id, image_index=None):
    if image_index is not None:
        match = image_index.get(image_id)
        if match is not None and match.is_file():
            return match

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
        matches = sorted(image_dir.rglob(f"{image_id}{extension}"))
        if matches:
            return matches[0]
    return None

class LayoutDataset(Dataset):
    def __init__(self, split_json, tensor_dir, image_dir, skip_report=None, latent_dir=None):
        with open(split_json, "r") as f:
            raw_data = json.load(f)

        self.tensor_dir = Path(tensor_dir)
        self.image_dir = Path(image_dir)
        self.latent_dir = Path(latent_dir) if latent_dir else None
        self.image_index = None if self.latent_dir is not None else _build_image_index(self.image_dir)
        self.data = self._filter_valid_items(raw_data, split_json, skip_report)
        logger.info(
            "Dataset ready from %s: %s valid samples",
            split_json,
            len(self.data),
        )

    def _filter_valid_items(self, data, split_json, skip_report):
        valid = []
        skipped = []
        processed = 0

        for item in data:
            processed += 1
            image_id = item.get("image_id")
            if not image_id:
                skipped.append({"item": item, "reason": "missing_image_id"})
                continue

            tensor_path = self.tensor_dir / f"{image_id}.npy"
            latent_path = self.latent_dir / f"{image_id}.npy" if self.latent_dir is not None else None
            image_path = None if self.latent_dir is not None else _resolve_image_path(self.image_dir, image_id, self.image_index)

            if not tensor_path.is_file():
                skipped.append(
                    {"image_id": image_id, "reason": "missing_tensor", "path": str(tensor_path)}
                )
                continue

            if latent_path is not None and not latent_path.is_file():
                skipped.append(
                    {"image_id": image_id, "reason": "missing_latent", "path": str(latent_path)}
                )
                continue

            if image_path is None:
                if self.latent_dir is not None:
                    valid.append(item)
                    continue
                skipped.append(
                    {
                        "image_id": image_id,
                        "reason": "missing_image",
                        "path": str(self.image_dir / f"{image_id}.[jpg|jpeg|png|webp]"),
                    }
                )
                continue

            valid.append(item)

        if skipped:
            logger.warning(
                "Skipped %s invalid samples while loading %s",
                len(skipped),
                split_json,
            )
            if skip_report:
                report_path = Path(skip_report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with report_path.open("w") as f:
                    json.dump(skipped, f, indent=2)
                logger.info("Wrote skipped-sample report to %s", report_path)

            if processed % 5000 == 0:
                logger.info(
                    "Dataset loading progress: processed=%s valid=%s skipped=%s from %s",
                    processed,
                    len(valid),
                    len(skipped),
                    split_json,
                )

        if not valid:
            raise ValueError(f"No valid samples left after filtering {split_json}")

        return valid
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        item = self.data[idx]
        image_id = item["image_id"]

        try:
            layout = np.load(self.tensor_dir / f"{image_id}.npy", allow_pickle=False)
            layout = torch.from_numpy(layout).permute(2, 0, 1).float()

            control = layout[:3]

            if self.latent_dir is not None:
                latent = np.load(self.latent_dir / f"{image_id}.npy", allow_pickle=False)
                latent = torch.from_numpy(latent).float()
                caption = item.get("caption") or "UI with no valid elements"
                return {"layout": layout, "control": control, "latent": latent, "caption": caption}

            image_path = _resolve_image_path(self.image_dir, image_id, self.image_index)
            if image_path is None:
                raise FileNotFoundError(f"No image found for {image_id}")
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        except Exception as exc:
            logger.warning("Skipping sample %s during read: %s", image_id, exc)
            return None

        caption = item.get("caption") or "UI with no valid elements"
        return {"layout": layout, "control": control, "image": image, "caption": caption}


def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)
        
        
