# src/unet3d.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):  # (B,C,Z,H,W)
        return self.conv(x)

def center_crop_3d(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Recorta centrado src para igualar las dims espaciales de target."""
    _, _, z, h, w = src.shape
    _, _, Z, H, W = target.shape
    cz = (z - Z) // 2
    ch = (h - H) // 2
    cw = (w - W) // 2
    return src[:, :, cz:cz+Z, ch:ch+H, cw:cw+W]

class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, levels: int = 4):
        """
        U-Net 3D con upsample trilineal (estable con tamaños impares).
        """
        super().__init__()
        chs = [base_ch * (2**i) for i in range(levels)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.down_blocks.append(ConvBlock3D(prev, c))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev = c

        self.bottleneck = ConvBlock3D(chs[-1], chs[-1]*2)

        # Decoder
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        dec_chs = list(reversed(chs))
        prev = chs[-1]*2
        for c in dec_chs:
            self.up_convs.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            self.up_blocks.append(ConvBlock3D(prev + c, c))
            prev = c

        self.final_conv = nn.Conv3d(dec_chs[-1], 1, kernel_size=1)

    def forward(self, x):  # x: (B,1,Z,H,W)
        skips = []
        out = x
        for blk, pool in zip(self.down_blocks, self.pools):
            out = blk(out)
            skips.append(out)
            out = pool(out)
        out = self.bottleneck(out)

        for up, blk, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            out = up(out)
            if out.shape[-3:] != skip.shape[-3:]:
                # Ajustar skip con crop centrado si difiere por impares
                skip = center_crop_3d(skip, out)
            out = torch.cat([skip, out], dim=1)
            out = blk(out)

        logits = self.final_conv(out)
        probs = torch.sigmoid(logits)
        return probs  # (B,1,Z,H,W)
