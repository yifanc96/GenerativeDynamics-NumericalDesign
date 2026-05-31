"""Thin wrapper around the existing lucidrains Unet (same as NSforecasting-*
scripts). Appends the coarse-observation conditioning as an extra channel.
"""
import sys
import os

# Reuse the UNet implementation in the parent folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet import Unet   # noqa: E402

import torch
import torch.nn as nn


class DriftNet(nn.Module):
    """b_theta(x, t, x0_up) for NS forecasting.

    - x:      (B, 1, H, W) current interpolant state
    - t:      (B, 1)       time
    - x0_up:  (B, 1, H, W) coarsened-then-upsampled conditioning
    Output shape: (B, 1, H, W).
    """
    def __init__(self, img_channels=1, cond_channels=1, unet_channels=32,
                 unet_dim_mults=(1, 2, 2, 2),
                 resnet_block_groups=8,
                 learned_sinusoidal_dim=32,
                 attn_dim_head=32, attn_heads=4):
        super().__init__()
        self.arch = Unet(
            num_classes=1,
            in_channels=img_channels + cond_channels,
            out_channels=img_channels,
            dim=unet_channels,
            dim_mults=unet_dim_mults,
            resnet_block_groups=resnet_block_groups,
            learned_sinusoidal_cond=True,
            random_fourier_features=False,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            use_classes=False,
        )

    def forward(self, x, t, x0_up):
        # Unet expects class label; we pass None
        h = torch.cat([x, x0_up], dim=1)
        # lucidrains Unet expects time as (B,) or (B, 1)? Check: it uses learned-sinusoidal embedding on t.
        t_ = t.reshape(-1) if t.dim() > 1 else t
        return self.arch(h, t_, None)
