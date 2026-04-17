import torch
import torch.nn as nn 

# Convert layout tensor to image conditioning map
class ControlNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.initial=nn.Sequential(
            nn.Conv2d(10,32,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final=nn.Conv2d(64,3,kernel_size=1)
        
    def forward(self,x):
        x=self.initial(x)
        x=self.up1(x)
        x=self.final(x)
        return x
