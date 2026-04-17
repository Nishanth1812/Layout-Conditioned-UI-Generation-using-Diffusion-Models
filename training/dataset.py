import os 
import json
import numpy as np
import torch 
from torch.utils.data import Dataset
from PIL import Image 

class LayoutDataset(Dataset):
    def __init__(self,split_json,tensor_dir,image_dir):
        with open(split_json,'r') as f:
            self.data=json.load(f)
            
        self.tensor_dir=tensor_dir
        self.image_dir=image_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        item=self.data[idx] 
        
        # layout tensor 
        layout=np.load(os.path.join(self.tensor_dir,item['image_id']+'.npy'))
        layout=torch.tensor(layout).permute(2,0,1).float()
        
        # Control Image 
        control=layout[:3]
        
        
        # Load Image
        image=Image.open(os.path.join(self.image_dir,item['image_id']+ '.jpg')).convert("RGB")
        image=image.resize((512,512))
        image=torch.tensor(np.array(image)).permute(2,0,1).float()/255.0 
        
        return {"layout":layout,"control":control,"image":image,"caption":item['caption']}
        
        