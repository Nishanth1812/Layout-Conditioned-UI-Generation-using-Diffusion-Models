import torch
import torch.nn as nn


def _norm(num_channels):
    groups = min(8, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            _norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class LayoutEncoder(nn.Module):
    def __init__(self, in_channels=10, embed_dim=768):
        super().__init__()

        self.conv_stack = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=16),   # -> (B,16,32,32)
            ConvBlock(in_channels=16, out_channels=32),   # -> (B,32,16,16)
            ConvBlock(in_channels=32, out_channels=64),  # -> (B,64,8,8)
            ConvBlock(in_channels=64, out_channels=128),   # -> (B,128,4,4)
            ConvBlock(in_channels=128, out_channels=256),   # -> (B,256,2,2)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 2 * 2, embed_dim)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.unsqueeze(1)  # (B, 1, embed_dim)
        return x
