"""
Empirical validation of the compact-support assumption in the Cameron-Martin
space (Proposition 3.3 / appendix F). We compute, for each of the three test
distributions, the empirical distribution of the Cameron-Martin norm

    ||x_1||_V^2 = sum_m |xhat_1(m)|^2 / spec_C0(m)

against several candidate noise covariances C_0:
  - Smoother than the data : Cameron-Martin norm diverges (ill-defined)
  - Matched to the data    : V-norm is bounded but heavy-tailed for non-Gauss
  - Rougher than the data  : V-norm is bounded, light-tailed (Prop 3.3 applies)

This script computes mean and 99th-percentile of ||x_1||_V across an empirical
sample, for all combinations.
"""

import math
import os
import numpy as np
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))


# ─── Helpers ──────────────────────────────────────────────────────────

def matern_density(grid_size, sigma_sq, ls, s):
    f = torch.fft.fftfreq(grid_size, device=DEVICE) * 2 * math.pi * grid_size
    fx, fy = torch.meshgrid(f, f, indexing='ij')
    return sigma_sq * (fx ** 2 + fy ** 2 + ls ** 2) ** (-s)


def sample_matern(num, grid_size, sigma_sq, ls, s, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    spec = matern_density(grid_size, sigma_sq, ls, s); spec[0, 0] = 0
    re = torch.randn(num, grid_size, grid_size, generator=g, device=DEVICE)
    im = torch.randn(num, grid_size, grid_size, generator=g, device=DEVICE)
    return torch.fft.ifft2(torch.sqrt(spec)[None] * (re + 1j * im), norm='forward').real


def cm_norm_sq(field, spec_C0):
    """||x||_V^2 = sum_m |xhat(m)|^2 / spec_C0(m)
    Truncated to resolved Fourier modes (so well-defined on the grid).
    field: (B, H, W); spec_C0: (H, W); returns (B,) cm-norm-sq.
    """
    if field.dim() == 4:
        field = field.squeeze(1)
    fhat = torch.fft.fftn(field, dim=(1, 2), norm='forward')
    amp2 = fhat.abs() ** 2
    inv_spec = torch.where(spec_C0 > 1e-30, 1.0 / spec_C0, torch.zeros_like(spec_C0))
    inv_spec[0, 0] = 0  # zero mean mode
    cm = (amp2 * inv_spec[None]).sum(dim=(1, 2))
    return cm


def summary(arr):
    a = arr.cpu().numpy()
    return dict(mean=float(a.mean()), std=float(a.std()),
                p50=float(np.percentile(a, 50)),
                p99=float(np.percentile(a, 99)),
                pmax=float(a.max()))


# ─── 1) Gaussian field test ─────────────────────────────────────────

def test_gaussian():
    print('\n=== Gaussian field, x_1 ~ Matern with s_1 = 3, on 128x128 ===')
    grid = 128
    ls = 1.0
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** 3
    x = sample_matern(2000, grid, sig1, ls, 3, seed=42)
    print(f'data ||x||_2^2 mean = {(x ** 2).mean(dim=(1, 2)).mean().item():.3f}')
    print()
    print(f"{'noise C_0':<35} {'mean ||x||_V^2':>16} {'99%':>16}")
    print('-' * 80)
    for s0, label in [(0, 'White noise (rougher)'),
                      (1, 'Matern s=1 (rougher)'),
                      (2, 'Matern s=2 (rougher)'),
                      (3, 'Matern s=3 (matched)'),
                      (4, 'Matern s=4 (smoother) -> ill-defined')]:
        sig0 = 1.0 if s0 == 0 else ((2 * math.pi) ** 2 + ls ** 2) ** s0
        spec_C0 = matern_density(grid, sig0, ls, s0)
        cm = cm_norm_sq(x, spec_C0)
        s = summary(cm)
        flag = '   <-- diverges as N -> inf' if s0 > 3 else ''
        print(f"  s_0={s0} ({label:<25}) {s['mean']:>14.3e}  {s['p99']:>14.3e} {flag}")
    return None


# ─── 2) Allen-Cahn test ───────────────────────────────────────────────

def test_allen_cahn():
    print('\n=== Allen-Cahn invariant distribution (1D, N=128) ===')
    # Approximate truth via white-noise + tanh, since we don't easily have
    # MCMC samples. Use the saved truth spectrum instead to derive ||x||_V^2.
    base = '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/Allen-Cahn/results/'
    # Load truth spectrum at N=128
    S_truth = torch.load(base + 'AllenCahn-res128-truth.pt', weights_only=False)
    kvals = torch.load(base + 'AllenCahn-res128-kvals.pt', weights_only=False)
    # Empirical S_truth = E[|hat x_1|^2] for the data, summed radially.
    print(f"  S_truth (ensemble enstrophy spectrum) shape: {S_truth.shape}")
    # ||x||_V^2 (matched) = sum_k S_truth(k) / S_truth(k) = #modes (trivial). So instead compute
    # ||x||_V^2 (under candidate C_0) = sum_k S_truth(k) / S_C0(k).
    # For Allen-Cahn, the matched prior has S_C0(k) ~ 1/(1 + k^2) (gradient norm density),
    # i.e. C_0 = (-d^2/dx^2)^{-1} restricted to non-zero modes.
    ks = np.array(kvals)
    spec_match = 1.0 / (1.0 + ks ** 2)
    spec_white = np.ones_like(ks)
    spec_smooth = 1.0 / (1.0 + ks ** 2) ** 2  # smoother than matched
    print(f"{'noise C_0':<35} {'mean ||x||_V^2':>16}")
    print('-' * 60)
    for spec, lbl in [(spec_match, 'Matched (-d^2/dx^2)^{-1}'),
                      (spec_white, 'White noise (rougher)'),
                      (spec_smooth, 'Smoother (-d^2/dx^2)^{-2}')]:
        # ||x||_V^2 = sum_k S_truth(k) / spec(k)
        total = float(np.sum(np.array(S_truth) / spec))
        print(f"  {lbl:<35}: {total:>14.3e}")
    return None


# ─── 3) Navier-Stokes test ─────────────────────────────────────────────

def test_navier_stokes():
    print('\n=== Navier-Stokes invariant distribution (2D, 128x128) ===')
    # Load NS data
    ns_path = '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file.pt'
    avg_pixel_norm = 3.0679163932800293
    data_raw, _ = torch.load(ns_path, weights_only=False)
    Ntj, Nts, Nx, Ny = data_raw.shape
    if Nx != 128:
        # downsample to 128
        data = data_raw / avg_pixel_norm
        data = data.reshape(-1, Nx, Ny)
        data = torch.nn.functional.interpolate(data.unsqueeze(1), size=(128, 128), mode='bilinear').squeeze(1)
    else:
        data = (data_raw / avg_pixel_norm).reshape(-1, 128, 128)
    # Take a 2000-sample subset
    data = data[:2000].to(DEVICE)
    print(f'data shape: {data.shape}, std = {data.std().item():.3f}')

    # Load candidate noise covariances
    amp_match = torch.load(os.path.join(HOME, 'enstrohpy_spectrum_amplitude.pt'), weights_only=False).to(DEVICE) / 5.0
    spec_match = (amp_match ** 2).squeeze(0) if amp_match.dim() > 2 else amp_match ** 2
    if spec_match.dim() == 3:
        spec_match = spec_match.squeeze(0)

    # white noise
    spec_white = torch.ones(128, 128, device=DEVICE)

    # mul-k rougher noise
    f = torch.fft.fftfreq(128, device=DEVICE) * 128
    fx, fy = torch.meshgrid(f, f, indexing='ij')
    k_mag = torch.sqrt(fx ** 2 + fy ** 2)
    spec_mulk = (amp_match.squeeze() ** 2) * (k_mag ** 2)

    print(f"{'noise C_0':<40} {'mean ||x||_V^2':>16} {'99%':>16}")
    print('-' * 80)
    for spec, lbl in [(spec_match.squeeze(), 'Matched-spectrum noise'),
                      (spec_white, 'White noise (much rougher)'),
                      (spec_mulk, 'Mul-k rougher spectrum noise')]:
        cm = cm_norm_sq(data, spec)
        s = summary(cm)
        print(f"  {lbl:<40}: {s['mean']:>14.3e}  {s['p99']:>14.3e}")

    # also compute pixel-value flatness to confirm strong non-Gaussianity
    # of NS vorticity vs Gaussian baseline (3)
    pix = data.flatten().cpu().numpy()
    pix = pix - pix.mean()
    flatness = (pix ** 4).mean() / (pix ** 2).mean() ** 2
    print(f"\n  Pixel-value flatness of NS vorticity = {flatness:.3f}  (Gaussian baseline = 3)")
    return None


if __name__ == '__main__':
    test_gaussian()
    test_allen_cahn()
    test_navier_stokes()
