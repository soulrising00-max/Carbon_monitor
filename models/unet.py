"""
Lightweight U-Net for binary forest segmentation.

Input:  (B, 6, 128, 128)  — 6 HLS spectral bands, normalized to [0, 1]
Output: (B, 1, 128, 128)  — logits; apply sigmoid + threshold 0.5 for binary mask

Designed to run on CPU. ~7M parameters.
Train on Kaggle with Hansen GFC labels, export as unet_forest.pth,
then drop into ml_models/ for inference.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d downsampling followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsampling + skip connection + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # after concat with skip: in_channels channels total
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd spatial dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class ForestUNet(nn.Module):
    """
    U-Net for binary forest segmentation from 6-band HLS imagery.

    Architecture:
        Encoder:    6  → 64 → 128 → 256 → 512
        Bottleneck: 512 → 1024
        Decoder:    1024+512 → 512 → 256+256 → 256 → 128+128 → 128 → 64+64 → 64
        Head:       64 → 1  (logit)

    Usage:
        model = ForestUNet()
        logits = model(x)          # x: (B, 6, 128, 128) → (B, 1, 128, 128)
        mask   = (logits.sigmoid() > 0.5).squeeze(1)   # (B, 128, 128) bool
    """

    def __init__(self, in_channels: int = 6, base_features: int = 64):
        super().__init__()

        f = base_features  # 64

        # Encoder
        self.inc   = DoubleConv(in_channels, f)       # → (B, 64,  128, 128)
        self.down1 = Down(f,     f * 2)               # → (B, 128,  64,  64)
        self.down2 = Down(f * 2, f * 4)               # → (B, 256,  32,  32)
        self.down3 = Down(f * 4, f * 8)               # → (B, 512,  16,  16)

        # Bottleneck
        self.down4 = Down(f * 8, f * 16)              # → (B, 1024,  8,   8)

        # Decoder
        self.up1 = Up(f * 16 + f * 8,  f * 8)        # 1024+512  → 512
        self.up2 = Up(f * 8  + f * 4,  f * 4)        # 512+256   → 256
        self.up3 = Up(f * 4  + f * 2,  f * 2)        # 256+128   → 128
        self.up4 = Up(f * 2  + f,      f)             # 128+64    → 64

        # Output head
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path — save skip connections
        x1 = self.inc(x)     # (B, 64,  128, 128)
        x2 = self.down1(x1)  # (B, 128,  64,  64)
        x3 = self.down2(x2)  # (B, 256,  32,  32)
        x4 = self.down3(x3)  # (B, 512,  16,  16)
        x5 = self.down4(x4)  # (B, 1024,  8,   8)

        # Decoder path — upsample + skip
        x = self.up1(x5, x4)  # (B, 512,  16, 16)
        x = self.up2(x,  x3)  # (B, 256,  32, 32)
        x = self.up3(x,  x2)  # (B, 128,  64, 64)
        x = self.up4(x,  x1)  # (B, 64,  128, 128)

        return self.out_conv(x)  # (B, 1,  128, 128)


# ---------------------------------------------------------------------------
# Convenience: binary mask from logits
# ---------------------------------------------------------------------------

def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert raw logits (B, 1, H, W) to a boolean mask (B, H, W).

    Args:
        logits:    raw model output before sigmoid
        threshold: probability cutoff (default 0.5)

    Returns:
        bool tensor of shape (B, H, W)
    """
    return (logits.sigmoid() > threshold).squeeze(1)
