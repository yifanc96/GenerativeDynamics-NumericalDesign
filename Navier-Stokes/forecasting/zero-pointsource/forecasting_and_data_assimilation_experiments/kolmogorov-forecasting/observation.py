"""Observation model for data-assimilation experiment.

y = A(omega_{t+tau}) + eta,   eta ~ N(0, sigma_y^2 I).

Two observation operators provided:
  - avgpool(factor) : k-fold average pool
  - pixel_mask(frac) : random boolean mask, keeping a fraction of pixels
"""
import torch
import torch.nn.functional as F


def obs_avgpool(x, factor=8):
    """x: (..., 1, H, W) -> (..., 1, H//f, W//f)."""
    leading = x.shape[:-2]
    h, w = x.shape[-2:]
    xx = x.reshape(-1, 1, h, w)
    y = F.avg_pool2d(xx, factor, stride=factor)
    return y.reshape(*leading, y.shape[-2], y.shape[-1])


def make_avgpool_operator(factor):
    """Return a callable A that does k-fold average pool."""
    def A(x):
        return obs_avgpool(x, factor=factor)
    A.name = f'avgpool{factor}'
    return A


def log_likelihood(y_pred_x1, y_obs, sigma_y):
    """-1/(2 sigma_y^2) * sum( (A(x1_hat) - y_obs)^2 ) per sample.
    y_pred_x1, y_obs: (B, 1, h, w) (observation-resolution)."""
    diff = y_pred_x1 - y_obs
    return -0.5 * (diff * diff).sum(dim=(-1, -2, -3)) / (sigma_y ** 2)
