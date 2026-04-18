"""Small MLP with Gaussian-Fourier time embedding. Optionally takes a flattened
conditioning vector (appended to x).
"""
import math
import torch
import torch.nn as nn


class GaussianFourierEmbed(nn.Module):
    def __init__(self, dim_out=64, scale=10.0):
        super().__init__()
        assert dim_out % 2 == 0
        self.W = nn.Parameter(torch.randn(dim_out // 2) * scale, requires_grad=False)

    def forward(self, t):
        # t: (B, 1). Output (B, dim_out).
        theta = 2.0 * math.pi * t * self.W
        return torch.cat([theta.sin(), theta.cos()], dim=-1)


class MLPNet(nn.Module):
    def __init__(self, x_dim=1, cond_dim=0, hidden=128, n_layers=3, time_embed=64):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.time_embed = GaussianFourierEmbed(time_embed)
        in_dim = x_dim + cond_dim + time_embed
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, *cond):
        # x: (B, x_dim). t: (B, 1). cond: tuple of (B, d_c).
        h = [x, self.time_embed(t)]
        for c in cond:
            h.append(c)
        return self.net(torch.cat(h, dim=-1))
