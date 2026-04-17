import json
import logging
from collections import Counter

logger=logging.getLogger(__name__)

def generate_caption(layout):
    types=[el["type"] for el in layout["elements"]]
    count=Counter(types)
    
    parts=[]
    
    for k,v in count.items():
        parts.append(f"{v} {k}")
        
    return "UI with " + ", ".join(parts)

def generate_all_captions(input_json, output_json):
    with open(input_json, 'r') as f:
        layouts = json.load(f)

    logger.info("Generating captions for %s layouts from %s",len(layouts),input_json)
        
    for layout in layouts:
        layout["caption"] = generate_caption(layout)
        
    with open(output_json, 'w') as f:
        json.dump(layouts, f, indent=2)
        
    logger.info("Captions generated at %s",output_json)
    
    
