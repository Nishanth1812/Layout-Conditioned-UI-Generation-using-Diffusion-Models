import json
import logging
import os

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
    "Background": 9
}

GRID_SIZE=64
CHANNELS=10
logger=logging.getLogger(__name__)

def encode_layout_to_tensor(layout):
    grid=np.zeros((GRID_SIZE,GRID_SIZE,CHANNELS),dtype=np.float32)
    
    for el in layout.get("elements", []):
        ch=TYPE_TO_CHANNEL.get(el["type"],None) 
        if ch is None:
            continue
        
        bbox=el.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1,y1,x2,y2=(float(v) for v in bbox)
        except (TypeError,ValueError):
            continue
        x1=max(0.0,min(1.0,x1))
        y1=max(0.0,min(1.0,y1))
        x2=max(0.0,min(1.0,x2))
        y2=max(0.0,min(1.0,y2))
        if x2 <= x1 or y2 <= y1:
            continue
        
        x1=int(x1*GRID_SIZE)
        y1=int(y1*GRID_SIZE)
        x2=int(x2*GRID_SIZE)
        y2=int(y2*GRID_SIZE) 
        
        grid[y1:y2,x1:x2,ch]=1.0 
    return grid 

def process_all_layouts(input_json,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    
    with open(input_json,'r') as f:
        layouts=json.load(f)

    logger.info("Encoding %s layouts into tensors at %s",len(layouts),output_dir)
    skipped=0
    for layout in layouts:
        tensor=encode_layout_to_tensor(layout)
        if not np.any(tensor):
            skipped += 1
            logger.warning("Skipping empty tensor for %s",layout.get("image_id","unknown"))
            continue
        np.save(os.path.join(output_dir,layout["image_id"]+".npy"),tensor)

    logger.info("Saved tensors to %s (skipped %s empty layouts)",output_dir,skipped)
        
