"""Ensemble-forecasting + physical-observable metrics for 2D vorticity forecasts.

Conventions:
    ens: (K, B, 1, H, W) — K ensemble members, B samples
    truth: (B, 1, H, W) — single deterministic truth per sample
    H = W = N
"""
import math

import torch
import torch.fft as fft
import numpy as np


# -------------- ensemble forecasting metrics --------------

def crps_ensemble(ens, truth):
    """Empirical CRPS, per-pixel-averaged.
    CRPS = E|X - y| - (1/2) E|X - X'|, X, X' iid ~ forecast ensemble."""
    K = ens.shape[0]
    y = truth.unsqueeze(0)                           # (1, B, 1, H, W)
    term1 = (ens - y).abs().mean(dim=0)              # (B, 1, H, W) — E|X-y|
    # E|X-X'| via sorted ensemble (efficient)
    # Use the V-statistic version: (1/(K*K)) sum_{i,j} |X_i - X_j|
    e = ens.reshape(K, -1)                           # (K, B*H*W)
    e_sorted, _ = e.sort(dim=0)
    weights = torch.arange(1, K + 1, device=ens.device, dtype=ens.dtype)
    pair = 2.0 * ((weights - (K + 1) / 2.0).unsqueeze(-1) * e_sorted).sum(dim=0) / (K * K)
    term2 = pair.reshape(*ens.shape[1:])             # (B, 1, H, W)
    crps = term1 - 0.5 * term2
    return crps.mean().item()


def rmse_ensemble_mean(ens, truth):
    """||mean(ens) - truth||_2 per sample, then pixel-averaged RMS."""
    em = ens.mean(dim=0)
    return (em - truth).pow(2).mean().sqrt().item()


def ensemble_spread(ens):
    """Per-pixel std of ensemble, pixel-averaged."""
    return ens.std(dim=0, unbiased=False).mean().item()


def spread_skill_ratio(ens, truth):
    return ensemble_spread(ens) / max(rmse_ensemble_mean(ens, truth), 1e-10)


def rank_histogram(ens, truth, n_bins=None):
    """Talagrand rank histogram. For each pixel, rank truth among the K ensemble
    members. Flat histogram = well-calibrated."""
    K = ens.shape[0]
    if n_bins is None:
        n_bins = K + 1
    # rank of truth in the sorted ensemble: count of ens members < truth, + 0.5 * ties
    less = (ens < truth.unsqueeze(0)).sum(dim=0)
    eq = (ens == truth.unsqueeze(0)).sum(dim=0)
    rank = less + (eq / 2).clamp_min(0)
    # rank in [0, K], we want histogram
    rank_flat = rank.reshape(-1).cpu().numpy()
    hist, _ = np.histogram(rank_flat, bins=np.arange(n_bins + 1) - 0.5)
    return hist / max(hist.sum(), 1)  # normalised


def anomaly_correlation_coefficient(ens, truth, climate_mean):
    """ACC = corr((mean_ens - climate), (truth - climate)), field-wide."""
    em = ens.mean(dim=0)
    a = em - climate_mean
    b = truth - climate_mean
    num = (a * b).mean()
    den = (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt()).clamp_min(1e-10)
    return (num / den).item()


# -------------- physical observables --------------

def radial_spectrum(omega):
    """Return (k_bins, E(k)) for batch of vorticity fields (B, H, W).
    E(k) = |omega_hat(k)|^2 / (2 k^2) summed over the annulus k <= |k'| < k+1.
    """
    B, H, W = omega.shape
    w_hat = fft.fft2(omega) / (H * W)
    power = w_hat.real ** 2 + w_hat.imag ** 2
    kx = fft.fftfreq(H, d=1.0 / H).view(-1, 1).expand(H, W).to(omega.device)
    ky = fft.fftfreq(W, d=1.0 / W).view(1, -1).expand(H, W).to(omega.device)
    kmag = (kx * kx + ky * ky).sqrt()
    k_bins = torch.arange(0, H // 2 + 1, device=omega.device, dtype=omega.dtype)
    spec = torch.zeros(B, len(k_bins), device=omega.device, dtype=omega.dtype)
    for i, k in enumerate(k_bins):
        mask = (kmag >= k) & (kmag < k + 1)
        if not mask.any():
            continue
        denom = max(k.item() ** 2, 1.0)
        s = power[:, mask].sum(dim=-1) / (2.0 * denom)
        spec[:, i] = s
    return k_bins, spec


def spectrum_rmse(ens_omega, truth_omega, log_domain=True):
    """RMSE between ensemble-averaged and truth-averaged spectra.
    ens_omega: (K, B, H, W) — pool across K and B for ensemble-average.
    truth_omega: (B, H, W) — pool across B.
    """
    k_e, spec_e = radial_spectrum(ens_omega.reshape(-1, *ens_omega.shape[-2:]))
    k_t, spec_t = radial_spectrum(truth_omega)
    e_mean = spec_e.mean(dim=0)
    t_mean = spec_t.mean(dim=0)
    if log_domain:
        eps = 1e-20
        diff = (torch.log(e_mean.clamp_min(eps)) - torch.log(t_mean.clamp_min(eps)))
    else:
        diff = e_mean - t_mean
    # exclude k=0 (DC)
    return diff[1:].pow(2).mean().sqrt().item()


def enstrophy_per_snapshot(omega):
    """omega: (..., H, W) -> (...,) enstrophy 0.5 * <omega^2>."""
    return 0.5 * (omega.pow(2)).mean(dim=(-1, -2))


def _w2_1d(a, b):
    a, b = a.reshape(-1).sort()[0], b.reshape(-1).sort()[0]
    if a.numel() == b.numel():
        return (a - b).pow(2).mean().sqrt().item()
    n = 1024
    q = torch.linspace(0.0, 1.0, n + 2, device=a.device)[1:-1]
    qa = torch.quantile(a.float(), q)
    qb = torch.quantile(b.float(), q)
    return (qa - qb).pow(2).mean().sqrt().item()


def enstrophy_w2(ens_omega, truth_omega):
    """W2 of per-snapshot enstrophy distribution, forecast vs truth."""
    e_f = enstrophy_per_snapshot(ens_omega.reshape(-1, *ens_omega.shape[-2:]))
    e_t = enstrophy_per_snapshot(truth_omega)
    return _w2_1d(e_f, e_t)


def vorticity_pdf_w2(ens_omega, truth_omega, max_n=200_000):
    """W2 of pooled pointwise vorticity distribution."""
    ens = ens_omega.reshape(-1)
    tru = truth_omega.reshape(-1)
    ens = ens[torch.randperm(ens.numel())[:max_n]]
    tru = tru[torch.randperm(tru.numel())[:max_n]]
    return _w2_1d(ens, tru)
