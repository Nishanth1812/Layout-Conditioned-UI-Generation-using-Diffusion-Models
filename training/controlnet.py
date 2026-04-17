import torch 
import torch.nn as nn

class ZeroConv2d(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self,x):
        return self.conv(x)
    
# Simplified controlnet style encoder 

class ControlNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv_in=nn.Conv2d(3,64,3,padding=1)
        
        self.down1=nn.Conv2d(64,128,3,stride=2,padding=1)
        self.down2=nn.Conv2d(128,256,3,stride=2,padding=1)
        self.down3=nn.Conv2d(256,512,3,stride=2,padding=1)
        
        self.zero1=ZeroConv2d(128,4)
        self.zero2=ZeroConv2d(256,4)
        self.zero3=ZeroConv2d(512,4)
        
        self.act=nn.ReLU()
        
    
    def forward(self,x):
        x=self.act(self.conv_in(x))
        
        d1=self.act(self.down1(x))
        d2=self.act(self.down2(d1))
        d3=self.act(self.down3(d2))
        
        return [self.zero1(d1),self.zero2(d2),self.zero3(d3)]
    
    
