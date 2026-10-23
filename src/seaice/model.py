"""U-Net from dl-learning Day 8, with one output head per task.

Unchanged from Day 8: DoubleConv blocks, four pooling levels, transposed-conv
upsampling, skip concatenation. Changed: the single 1x1 `head` becomes a
ModuleDict of 1x1 heads (SIC, SOD, FLOE) that share the whole encoder-decoder,
and forward() returns {task: logits}. The use_skip ablation flag is dropped.

Trained from scratch: Day 12 measured no benefit from ImageNet initialisation on
multispectral input and 5-6x higher seed-to-seed variance; 17 channels of SAR
and passive microwave are further still from ImageNet.

Input height/width must be divisible by 16 (four 2x poolings).
"""
from __future__ import annotations

import torch
import torch.nn as nn

TASKS = {"SIC": 11, "SOD": 6, "FLOE": 7}


class DoubleConv(nn.Module):
    """conv-BN-ReLU twice -- the basic block at every U-Net level."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskUNet(nn.Module):
    def __init__(self, in_ch: int = 17, tasks: dict[str, int] | None = None, base: int = 32):
        super().__init__()
        tasks = tasks or TASKS
        b = base
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_ch, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        # One 1x1 conv per task on the shared final feature map.
        self.heads = nn.ModuleDict({t: nn.Conv2d(b, n, 1) for t, n in tasks.items()})

    def forward(self, x) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return {t: head(d1) for t, head in self.heads.items()}
