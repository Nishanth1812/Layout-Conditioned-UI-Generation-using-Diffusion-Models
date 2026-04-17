import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate

logger = logging.getLogger(__name__)

class LayoutDataset(Dataset):
    def __init__(self, split_json, tensor_dir, image_dir, skip_report=None):
        with open(split_json, "r") as f:
            raw_data = json.load(f)

        self.tensor_dir = Path(tensor_dir)
        self.image_dir = Path(image_dir)
        self.data = self._filter_valid_items(raw_data, split_json, skip_report)
        logger.info(
            "Dataset ready from %s: %s valid samples",
            split_json,
            len(self.data),
        )

    def _filter_valid_items(self, data, split_json, skip_report):
        valid = []
        skipped = []

        for item in data:
            image_id = item.get("image_id")
            if not image_id:
                skipped.append({"item": item, "reason": "missing_image_id"})
                continue

            tensor_path = self.tensor_dir / f"{image_id}.npy"
            image_path = self.image_dir / f"{image_id}.jpg"

            if not tensor_path.is_file():
                skipped.append(
                    {"image_id": image_id, "reason": "missing_tensor", "path": str(tensor_path)}
                )
                continue

            if not image_path.is_file():
                skipped.append(
                    {"image_id": image_id, "reason": "missing_image", "path": str(image_path)}
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

            image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
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
        
        
