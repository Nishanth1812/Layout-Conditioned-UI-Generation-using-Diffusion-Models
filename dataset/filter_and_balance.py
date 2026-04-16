import os
import json
import random 

def filter_layouts(layouts, min_elements=3):
    return [l for l in layouts if len(l["elements"]) >= min_elements]

def balance_dataset(layouts, max_per_type=5000):
    buckets = {}

    for layout in layouts:
        key = layout["caption"]
        buckets.setdefault(key, []).append(layout)

    balanced = []
    for k, v in buckets.items():
        random.shuffle(v)
        balanced.extend(v[:max_per_type])

    return balanced

def run_filter_balance(input_json, output_json):
    with open(input_json, 'r') as f:
        layouts = json.load(f)

    layouts = filter_layouts(layouts)
    layouts = balance_dataset(layouts)

    with open(output_json, 'w') as f:
        json.dump(layouts, f, indent=2)

    print("Filtered + Balanced")