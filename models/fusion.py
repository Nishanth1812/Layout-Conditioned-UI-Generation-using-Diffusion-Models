import torch
import torch.nn as nn 


# Combines CLIP and Layout embeddings
class ConditioningFusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    # clip_embed = clip embeddings
    # layout_embed = layout embeddings
    def forward(self,clip_embed,layout_embed):
        return torch.cat([clip_embed,layout_embed],dim=1)
    