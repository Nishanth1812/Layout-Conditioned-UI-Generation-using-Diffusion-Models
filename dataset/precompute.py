import os
import json
from sklearn.model_selection import train_test_split

def create_splits(input_json,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    
    with open(input_json,'r') as f: 
        layouts = json.load(f)
    
    train, temp = train_test_split(layouts, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    json.dump(train, open(os.path.join(out_dir,'train.json'),'w'), indent=2)
    json.dump(val, open(os.path.join(out_dir,'val.json'),'w'), indent=2)
    json.dump(test, open(os.path.join(out_dir,'test.json'),'w'), indent=2)
    
    print("Data splits created")