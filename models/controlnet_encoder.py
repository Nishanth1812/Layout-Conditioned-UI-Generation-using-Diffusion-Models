import torch
import torch.nn as nn


def _norm(num_channels):
    groups = min(8, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            _norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            _norm(32),
            nn.SiLU(),
        )
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        self.zero1 = ZeroConv2d(64, 4)
        self.zero2 = ZeroConv2d(128, 4)
        self.zero3 = ZeroConv2d(256, 4)

    def forward(self, x):
        x = self.stem(x)
        feat1 = self.down1(x)
        feat2 = self.down2(feat1)
        feat3 = self.down3(feat2)
        return [self.zero1(feat1), self.zero2(feat2), self.zero3(feat3)]


ControlNetEncoder = ControlNet
