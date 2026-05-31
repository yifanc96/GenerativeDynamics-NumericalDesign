"""Load cached Kolmogorov-flow snapshots; build (coarse x_t, fine x_{t+lag}) pairs.

The generative model's conditioning input is a *partially observed* (spatially
coarsened, then bilinearly upsampled back) version of the current state. This
is what makes the forecasting problem probabilistic: many fine-scale
configurations are consistent with the same coarse observation.
"""
import os

import torch
import torch.nn.functional as F


def load_snapshots(path):
    """Return (omega_tensor, args_dict) from a simulate.py output .pt file."""
    ck = torch.load(path, weights_only=False, map_location='cpu')
    return ck['omega'], ck['args']


def coarsen(x, factor=4):
    """AvgPool by `factor` along each spatial dim. Input shape (..., H, W)."""
    leading = x.shape[:-2]
    h, w = x.shape[-2:]
    xx = x.reshape(-1, 1, h, w)
    y = F.avg_pool2d(xx, factor, stride=factor)
    return y.reshape(*leading, y.shape[-2], y.shape[-1])


def upsample_bilinear(x, target_hw):
    leading = x.shape[:-2]
    h, w = x.shape[-2:]
    xx = x.reshape(-1, 1, h, w)
    y = F.interpolate(xx, size=target_hw, mode='bilinear', align_corners=False)
    return y.reshape(*leading, *target_hw)


def make_pairs(omega, lag, coarsen_factor=4):
    """From (N_traj, N_snap, H, W) extract all (omega_t, omega_{t+lag}) pairs.
    Returns (x0_full, x0_coarse_upsampled, x1_full) each shape (M, 1, H, W).
    """
    N, T, H, W = omega.shape
    assert lag >= 1 and lag < T
    x0 = omega[:, :-lag]                         # (N, T-lag, H, W)
    x1 = omega[:,  lag:]                         # (N, T-lag, H, W)
    x0 = x0.reshape(-1, 1, H, W)
    x1 = x1.reshape(-1, 1, H, W)
    x0_coarse = coarsen(x0, factor=coarsen_factor)
    x0_up = upsample_bilinear(x0_coarse, (H, W))
    return x0, x0_up, x1


def split_train_val_test(N_traj, n_train, n_val, n_test, seed=0):
    assert n_train + n_val + n_test <= N_traj
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N_traj, generator=g)
    return (perm[:n_train],
            perm[n_train:n_train + n_val],
            perm[n_train + n_val:n_train + n_val + n_test])


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, omega, lag, coarsen_factor=4):
        """omega: (N_traj, N_snap, H, W), a subset (e.g. after train split)."""
        self.x0_full, self.x0_up, self.x1_full = make_pairs(omega, lag, coarsen_factor)

    def __len__(self):
        return self.x0_full.shape[0]

    def __getitem__(self, i):
        return {
            'x0': self.x0_full[i],     # (1, H, W) full current state
            'x0_up': self.x0_up[i],    # (1, H, W) coarse-then-upsampled conditioning
            'x1': self.x1_full[i],     # (1, H, W) full future state
        }


def make_loaders(omega_all, lag, train_idx, val_idx, test_idx,
                 batch_size, coarsen_factor=4, num_workers=0):
    tr = omega_all[train_idx]
    va = omega_all[val_idx]
    te = omega_all[test_idx]
    train_ds = PairDataset(tr, lag, coarsen_factor)
    val_ds = PairDataset(va, lag, coarsen_factor)
    test_ds = PairDataset(te, lag, coarsen_factor)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl
