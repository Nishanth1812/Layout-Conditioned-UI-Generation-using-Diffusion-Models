import json
import logging
import os
from pathlib import Path

ALLOWED_TYPES=["Text", "Button", "Image", "Input", "Icon", "Toolbar", "List Item", "Card", "Advertisement", "Background"]
logger=logging.getLogger(__name__)

def normalize_bbox(bounds,w,h):
    if not isinstance(bounds,(list,tuple)) or len(bounds) != 4:
        return None
    try:
        x1,y1,x2,y2=(float(v) for v in bounds)
    except (TypeError,ValueError):
        return None
    w=max(w,1)
    h=max(h,1)
    x1=max(0.0,min(1.0,x1/w))
    y1=max(0.0,min(1.0,y1/h))
    x2=max(0.0,min(1.0,x2/w))
    y2=max(0.0,min(1.0,y2/h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1,y1,x2,y2]

def extract_elements(node,w,h,elements):
    if not isinstance(node, dict):
        return

    label=node.get("componentLabel")
    if label in ALLOWED_TYPES:
        bbox=normalize_bbox(node.get("bounds"),w,h)
        if bbox is not None:
            elements.append({"type":label,"bbox":bbox})

    for child in node.get("children",[]) or []:
        extract_elements(child,w,h,elements) 


def process_rico_json(path):
    with open(path,'r') as f:
        data=json.load(f)
    if not isinstance(data, dict):
        raise ValueError("root is not a JSON object")

    bounds=data.get("bounds",[0,0,1440,2560])
    w=bounds[2] if len(bounds) >= 4 and bounds[2] > 0 else 1440
    h=bounds[3] if len(bounds) >= 4 and bounds[3] > 0 else 2560

    elements=[]
    extract_elements(data,w,h,elements)

    return {"image_id":os.path.basename(path).replace(".json",""),"elements":elements}

def run_extraction(input_dir,output_file,skipped_file=None):
    logger.info("Scanning JSON files in %s",input_dir)
    all_layouts=[]
    skipped=[]
    
    for f in sorted(os.listdir(input_dir)):
        if f.endswith(".json"):
            path=Path(input_dir)/f
            try:
                layout=process_rico_json(path)
            except Exception as exc:
                skipped.append({"image_id":path.stem,"file":str(path),"reason":f"parse_error:{exc}"})
                continue

            if len(layout["elements"]) >0:
                all_layouts.append(layout)
            else:
                skipped.append({"image_id":layout["image_id"],"file":str(path),"reason":"no_allowed_elements"})

    with open(output_file,'w') as f:
        json.dump(all_layouts,f,indent=2)

    if skipped_file:
        with open(skipped_file,'w') as f:
            json.dump(skipped,f,indent=2)

    logger.info("Extracted %s layouts to %s",len(all_layouts),output_file)
    logger.info("Skipped %s layouts",len(skipped))
    if skipped:
        logger.info("Skipped sample notes written to %s",skipped_file)
