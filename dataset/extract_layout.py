import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ALLOWED_TYPES = ["Text", "Button", "Image", "Input", "Icon", "Toolbar", "List Item", "Card", "Advertisement", "Background"]
logger = logging.getLogger(__name__)

def normalize_bbox(bounds, w, h):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    w = max(w, 1)
    h = max(h, 1)
    x1 = max(0.0, min(1.0, x1 / w))
    y1 = max(0.0, min(1.0, y1 / h))
    x2 = max(0.0, min(1.0, x2 / w))
    y2 = max(0.0, min(1.0, y2 / h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def extract_elements(node, w, h, elements):
    if not isinstance(node, dict):
        return

    label = node.get("componentLabel")
    if label in ALLOWED_TYPES:
        bbox = normalize_bbox(node.get("bounds"), w, h)
        if bbox is not None:
            elements.append({"type": label, "bbox": bbox})

    for child in node.get("children", []) or []:
        extract_elements(child, w, h, elements)


def process_rico_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("root is not a JSON object")

    bounds = data.get("bounds", [0, 0, 1440, 2560])
    w = bounds[2] if len(bounds) >= 4 and bounds[2] > 0 else 1440
    h = bounds[3] if len(bounds) >= 4 and bounds[3] > 0 else 2560

    elements = []
    extract_elements(data, w, h, elements)

    return {"image_id": os.path.basename(path).replace(".json", ""), "elements": elements}

def _process_file(path):
    path = Path(path)
    try:
        layout = process_rico_json(path)
    except Exception as exc:
        return None, {"image_id": path.stem, "file": str(path), "reason": f"parse_error:{exc}"}

    if layout["elements"]:
        return layout, None

    return None, {"image_id": layout["image_id"], "file": str(path), "reason": "no_allowed_elements"}

def _worker_count(explicit):
    if explicit is not None:
        return max(1, int(explicit))

    raw = os.environ.get("EXTRACT_WORKERS") or os.environ.get("PREPROCESS_WORKERS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass

    return max(1, min((os.cpu_count() or 1), 8))

def run_extraction(input_dir, output_file, skipped_file=None, num_workers=None, log_every=500, **_kwargs):
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))
    total = len(json_files)
    all_layouts = []
    skipped = []
    workers = _worker_count(num_workers)

    logger.info("Scanning %s JSON files in %s using %s worker(s)", total, input_dir, workers)
    if total == 0:
        raise ValueError(f"No JSON files found in {input_dir}")

    iterator = None
    if workers == 1:
        iterator = map(_process_file, json_files)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_process_file, json_files, chunksize=16)
            for index, (layout, skipped_item) in enumerate(iterator, start=1):
                if layout is not None:
                    all_layouts.append(layout)
                if skipped_item is not None:
                    skipped.append(skipped_item)
                if index == 1 or index % log_every == 0 or index == total:
                    logger.info(
                        "Extracted progress: %s/%s files processed, %s layouts kept, %s skipped",
                        index,
                        total,
                        len(all_layouts),
                        len(skipped),
                    )

    if workers == 1:
        for index, (layout, skipped_item) in enumerate(iterator, start=1):
            if layout is not None:
                all_layouts.append(layout)
            if skipped_item is not None:
                skipped.append(skipped_item)
            if index == 1 or index % log_every == 0 or index == total:
                logger.info(
                    "Extracted progress: %s/%s files processed, %s layouts kept, %s skipped",
                    index,
                    total,
                    len(all_layouts),
                    len(skipped),
                )

    with open(output_file, "w") as f:
        json.dump(all_layouts, f, indent=2)

    if skipped_file:
        with open(skipped_file, "w") as f:
            json.dump(skipped, f, indent=2)

    logger.info("Extracted %s layouts to %s", len(all_layouts), output_file)
    logger.info("Skipped %s layouts", len(skipped))
    if skipped and skipped_file:
        logger.info("Skipped sample notes written to %s", skipped_file)
