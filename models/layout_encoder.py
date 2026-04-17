import torch 
import torch.nn as nn 

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stide=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self,x):
        return self.block(x)
    
class LayoutEncoder(nn.Module):
    
    def __init__(self,in_channels=10,embed_dim=1024):
        super().__init__()
        
        self.conv_stack=nn.Sequential(
            ConvBlock(in_channels=in_channels,out_channels=32),   # -> (B,32,32,32)
            ConvBlock(in_channels=32,out_channels=64),   # -> (B,64,16,16)
            ConvBlock(in_channels=64,out_channels=128),  # -> (B,128,8,8)
            ConvBlock(in_channels=128,out_channels=256),   # -> (B,256,4,4)
            ConvBlock(in_channels=256,out_channels=512),   # -> (B,512,2,2)
        )
        
        self.flatten=nn.flatten()
        self.fc=nn.Linear(512*2*2,embed_dim)
        
    def forward(self,x):
        x=self.conv_stack(x)
        x=self.flatten(x)
        x=self.fc(x)
        x=x.unsqueeze(1)  # (B,1,1024)
        return x 
