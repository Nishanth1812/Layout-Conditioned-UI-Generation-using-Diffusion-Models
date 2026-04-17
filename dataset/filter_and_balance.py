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

def balance_dataset(layouts, max_per_type=5000):
    buckets = {}

    for layout in layouts:
        key = layout.get("caption", "uncategorized ui")
        buckets.setdefault(key, []).append(layout)

    balanced = []
    for k, v in buckets.items():
        random.shuffle(v)
        balanced.extend(v[:max_per_type])

    return balanced

def run_filter_balance(input_json, output_json):
    with open(input_json, 'r') as f:
        layouts = json.load(f)

    logger.info("Filtering %s layouts from %s",len(layouts),input_json)
    layouts = filter_layouts(layouts)
    layouts = balance_dataset(layouts)

    with open(output_json, 'w') as f:
        json.dump(layouts, f, indent=2)

    logger.info("Filtered + balanced dataset saved to %s (%s layouts)",output_json,len(layouts))
