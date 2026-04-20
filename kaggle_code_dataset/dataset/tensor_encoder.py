import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TYPE_TO_CHANNEL = {
    "Text": 0,
    "Button": 1,
    "Image": 2,
    "Input": 3,
    "Icon": 4,
    "Toolbar": 5,
    "List Item": 6,
    "Card": 7,
    "Advertisement": 8,
    "Background": 9,
}

GRID_SIZE = 64
CHANNELS = 10
logger = logging.getLogger(__name__)


def encode_layout_to_tensor(layout):
    grid = np.zeros((GRID_SIZE, GRID_SIZE, CHANNELS), dtype=np.float32)

    for el in layout.get("elements", []):
        channel = TYPE_TO_CHANNEL.get(el["type"])
        if channel is None:
            continue

        bbox = el.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        x1 = int(x1 * GRID_SIZE)
        y1 = int(y1 * GRID_SIZE)
        x2 = int(x2 * GRID_SIZE)
        y2 = int(y2 * GRID_SIZE)
        grid[y1:y2, x1:x2, channel] = 1.0

    return grid


def _worker_count(explicit):
    if explicit is not None:
        return max(1, int(explicit))

    raw = os.environ.get("TENSOR_WORKERS") or os.environ.get("PREPROCESS_WORKERS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass

    return max(1, min((os.cpu_count() or 1), 8))


def _encode_and_save(layout, output_dir):
    tensor = encode_layout_to_tensor(layout)
    if not np.any(tensor):
        return layout.get("image_id", "unknown"), False

    np.save(os.path.join(output_dir, layout["image_id"] + ".npy"), tensor)
    return layout.get("image_id", "unknown"), True


def process_all_layouts(input_json, output_dir, num_workers=None, log_every=500, **_kwargs):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r") as f:
        layouts = json.load(f)

    workers = _worker_count(num_workers)
    total = len(layouts)
    logger.info("Encoding %s layouts into tensors at %s using %s worker(s)", total, output_dir, workers)

    skipped = 0
    if workers == 1:
        iterator = map(lambda layout: _encode_and_save(layout, output_dir), layouts)
        for index, (image_id, saved) in enumerate(iterator, start=1):
            if not saved:
                skipped += 1
                logger.warning("Skipping empty tensor for %s", image_id)
            if index == 1 or index % log_every == 0 or index == total:
                logger.info(
                    "Tensor progress: %s/%s layouts processed, %s saved, %s skipped",
                    index,
                    total,
                    index - skipped,
                    skipped,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_encode_and_save, layouts, [output_dir] * total, chunksize=16)
            for index, (image_id, saved) in enumerate(iterator, start=1):
                if not saved:
                    skipped += 1
                    logger.warning("Skipping empty tensor for %s", image_id)
                if index == 1 or index % log_every == 0 or index == total:
                    logger.info(
                        "Tensor progress: %s/%s layouts processed, %s saved, %s skipped",
                        index,
                        total,
                        index - skipped,
                        skipped,
                    )

    logger.info("Saved tensors to %s (skipped %s empty layouts)", output_dir, skipped)

