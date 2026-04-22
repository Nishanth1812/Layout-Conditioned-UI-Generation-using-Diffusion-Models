import json
import logging
import random 

logger=logging.getLogger(__name__)

def filter_layouts(layouts, min_elements=3):
    return [
        l for l in layouts
        if len(l.get("elements", [])) >= min_elements
        and l.get("caption")
        and l["caption"] != "UI with no valid elements"
    ]

def balance_dataset(layouts, max_per_type=5000, sample_seed=42):
    buckets = {}
    rng = random.Random(sample_seed)

    for layout in layouts:
        key = layout.get("caption", "uncategorized ui")
        buckets.setdefault(key, []).append(layout)

    balanced = []
    for k in sorted(buckets):
        v = buckets[k]
        rng.shuffle(v)
        balanced.extend(v[:max_per_type])

    rng.shuffle(balanced)

    return balanced

def run_filter_balance(input_json, output_json, max_layouts=30000, sample_seed=42):
    with open(input_json, 'r') as f:
        layouts = json.load(f)

    logger.info("Filtering %s layouts from %s",len(layouts),input_json)
    rng = random.Random(sample_seed)
    rng.shuffle(layouts)
    layouts = filter_layouts(layouts)
    layouts = balance_dataset(layouts, sample_seed=sample_seed)

    if max_layouts is not None:
        max_layouts = max(1, int(max_layouts))
        if len(layouts) > max_layouts:
            layouts = layouts[:max_layouts]
            logger.info("Capped randomized usable layouts to %s", max_layouts)

    with open(output_json, 'w') as f:
        json.dump(layouts, f, indent=2)

    logger.info("Filtered + balanced dataset saved to %s (%s layouts)",output_json,len(layouts))
