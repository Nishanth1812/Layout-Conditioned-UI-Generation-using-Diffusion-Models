import json
import os  

ALLOWED_TYPES=["Text", "Button", "Image", "Input", "Icon", "Toolbar", "List Item", "Card", "Advertisement", "Background"]

# w = width, h = height
def normalize_bbox(bounds,w,h):
    x1,y1,x2,y2=bounds
    w=max(w,1)
    h=max(h,1)
    return [x1/w,y1/h,x2/w,y2/h]

def extract_elements(node,w,h,elements):
    if not isinstance(node, dict):
        return

    if "bounds" in node and "componentLabel" in node:
        label=node["componentLabel"] 
        if label in ALLOWED_TYPES:
            bbox=normalize_bbox(node["bounds"],w,h) 
            elements.append({"type":label,"bbox":bbox}) 

    for child in node.get("children",[]):
        extract_elements(child,w,h,elements) 


def process_rico_json(path):
    with open(path,'r') as f:
        data=json.load(f)
        
    bounds=data.get("bounds",[0,0,1440,2560])
    w=bounds[2] if len(bounds) >= 4 and bounds[2] > 0 else 1440
    h=bounds[3] if len(bounds) >= 4 and bounds[3] > 0 else 2560
    
    elements=[]
    extract_elements(data,w,h,elements)
    
    return {"image_id":os.path.basename(path).replace(".json",""),"elements":elements} 

def run_extraction(input_dir,output_file):
    all_layouts=[]
    
    for f in os.listdir(input_dir):
        if f.endswith(".json"):
            layout=process_rico_json(os.path.join(input_dir,f))
            if len(layout["elements"]) >0:
                all_layouts.append(layout) 
                
    with open(output_file,'w') as f:
        json.dump(all_layouts,f,indent=2)
    
    print(f"Extracted {len(all_layouts)} layouts to {output_file}")
