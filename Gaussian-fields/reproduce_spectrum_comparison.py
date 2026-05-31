"""
Reproduce spectrum accuracy comparison: standard vs scaled interpolation schedule.

Based on arXiv:2509.01629 — the scaled schedule uses frequency-dependent
exponential interpolation coefficients derived from the spectral ratio
between source and target measures, whereas the standard schedule uses
linear α(t)=1-t, β(t)=t.

This script:
1. Recomputes everything from scratch (no saved .pt needed)
2. Compares per-frequency-band relative error for both schedules
3. Tests across resolutions 32, 64, 128
"""

import numpy as np
import torch
import torch.nn as nn
import math
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─── Matérn GP sampling ────────────────────────────────────────────────

def generate_matern_laplace(num_samples, grid_size, sigma_sq=1.0, length_scale=1.0, s=2.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    freq = torch.fft.fftfreq(grid_size) * 2 * math.pi * grid_size
    freq_x, freq_y = torch.meshgrid(freq, freq, indexing='ij')
    laplacian = freq_x**2 + freq_y**2
    spectral_density = sigma_sq * (laplacian + length_scale**2)**(-s)
    spectral_density[0, 0] = 0
    spectral_density = spectral_density.unsqueeze(0)
    noise = torch.randn(num_samples, grid_size, grid_size) + 1j * torch.randn(num_samples, grid_size, grid_size)
    spectral_sample = torch.sqrt(spectral_density) * noise
    sample = torch.fft.ifft2(spectral_sample, norm='forward').real
    return sample

# ─── Spectrum computation ──────────────────────────────────────────────

def get_Fourier_spectrum(data):
    data_hat = torch.fft.fftn(data, dim=(1, 2), norm="forward")
    fourier_amplitudes = np.abs(data_hat)**2
    fourier_amplitudes = fourier_amplitudes.mean(dim=0)
    npix = data_hat.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins_w, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
    Abins_w *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins_w

# ─── ODE integrator (manual RK4, no torchdiffeq needed) ───────────────

def integrate_ode(drift_fn, z0, param0, param1, grid_size, T_min, T_max, steps):
    """Fixed-step RK4 integration."""
    tgrid = torch.linspace(T_min, T_max, steps).type_as(z0)
    dt_vals = tgrid[1:] - tgrid[:-1]
    zt = z0.clone()
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i].item()
        dt = dt_vals[i].item()
        k1 = drift_fn(zt, t_i, param0, param1, grid_size)
        k2 = drift_fn(zt + 0.5 * dt * k1, t_i + 0.5 * dt, param0, param1, grid_size)
        k3 = drift_fn(zt + 0.5 * dt * k2, t_i + 0.5 * dt, param0, param1, grid_size)
        k4 = drift_fn(zt + dt * k3, t_i + dt, param0, param1, grid_size)
        zt = zt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return zt

# ─── Standard schedule ─────────────────────────────────────────────────

def make_standard_drift():
    def alpha(t): return 1.0 - t
    def alpha_dot(t): return -1.0
    def beta(t): return t
    def beta_dot(t): return 1.0

    def drift_b(z, t, param0, param1, grid_size):
        freq = torch.fft.fftfreq(grid_size) * 2 * math.pi * grid_size
        freq_x, freq_y = torch.meshgrid(freq, freq, indexing='ij')
        laplacian = freq_x**2 + freq_y**2
        sigma_sq0, ls0, s0 = param0
        sigma_sq1, ls1, s1 = param1
        sd0 = sigma_sq0 * (laplacian + ls0**2)**(-s0)
        sd1 = sigma_sq1 * (laplacian + ls1**2)**(-s1)
        a, da, b, db = alpha(t), alpha_dot(t), beta(t), beta_dot(t)
        ratio = (a * da * sd0[None] + b * db * sd1[None]) / (a**2 * sd0[None] + b**2 * sd1[None])
        xfft = torch.fft.fft2(z, dim=(1, 2), norm='forward')
        return torch.fft.ifft2(ratio * xfft, dim=(1, 2), norm='forward').real
    return drift_b

# ─── Scaled schedule (arXiv:2509.01629) ───────────────────────────────

def make_scaled_drift(param0, param1, grid_size):
    """
    The key idea: define the interpolation schedule using the spectral ratio
    at the highest resolved frequency. This makes the schedule adapt to the
    enormous dynamic range between source and target spectra.

    r = S_1(k_max) / S_0(k_max)   (ratio at Nyquist)

    α(t) = sqrt( (r - r^t) / (r - 1) )
    β(t) = sqrt( (r^t - 1) / (r - 1) )

    These satisfy α(0)=1, α(1)=0, β(0)=0, β(1)=1 and
    α²(t)S_0 + β²(t)S_1 interpolates geometrically across scales.
    """
    freq = torch.fft.fftfreq(grid_size) * 2 * math.pi * grid_size
    freq_x, freq_y = torch.meshgrid(freq, freq, indexing='ij')
    laplacian = freq_x**2 + freq_y**2
    sigma_sq0, ls0, s0 = param0
    sigma_sq1, ls1, s1 = param1
    sd0 = sigma_sq0 * (laplacian + ls0**2)**(-s0)
    sd1 = sigma_sq1 * (laplacian + ls1**2)**(-s1)
    # ratio at Nyquist frequency (grid_size//2, grid_size//2)
    r = (sd1[grid_size // 2, grid_size // 2] / sd0[grid_size // 2, grid_size // 2]).item()

    def alpha(t):
        return np.sqrt((r - r**t) / (r - 1))
    def alpha_dot(t):
        return -0.5 / alpha(t) * (r**t) * np.log(r) / (r - 1)
    def beta(t):
        return np.sqrt((r**t - 1) / (r - 1))
    def beta_dot(t):
        return 0.5 / beta(t) * (r**t) * np.log(r) / (r - 1)

    def drift_b(z, t, param0, param1, grid_size):
        freq = torch.fft.fftfreq(grid_size) * 2 * math.pi * grid_size
        freq_x, freq_y = torch.meshgrid(freq, freq, indexing='ij')
        laplacian = freq_x**2 + freq_y**2
        sigma_sq0, ls0, s0 = param0
        sigma_sq1, ls1, s1 = param1
        sd0 = sigma_sq0 * (laplacian + ls0**2)**(-s0)
        sd1 = sigma_sq1 * (laplacian + ls1**2)**(-s1)
        a, da, b, db = alpha(t), alpha_dot(t), beta(t), beta_dot(t)
        ratio = (a * da * sd0[None] + b * db * sd1[None]) / (a**2 * sd0[None] + b**2 * sd1[None])
        xfft = torch.fft.fft2(z, dim=(1, 2), norm='forward')
        return torch.fft.ifft2(ratio * xfft, dim=(1, 2), norm='forward').real

    return drift_b, r

# ─── Run experiment ────────────────────────────────────────────────────

def run_experiment(grid_size, num_samples=2000, rk_steps_standard=80, rk_steps_scaled=20):
    print(f"\n{'='*60}")
    print(f"Resolution: {grid_size}x{grid_size}")
    print(f"Standard: RK4 with {rk_steps_standard} steps | Scaled: RK4 with {rk_steps_scaled} steps")
    print(f"{'='*60}")

    # Source: white noise (s=0), Target: Matérn with s=3
    ls0, s0 = 1.0, 0.0
    sigma_sq0 = ((2 * np.pi)**2 + ls0**2)**s0
    param0 = [sigma_sq0, ls0, s0]

    ls1, s1 = 1.0, 3.0
    sigma_sq1 = ((2 * np.pi)**2 + ls1**2)**s1
    param1 = [sigma_sq1, ls1, s1]

    # Generate source and target samples
    z0 = generate_matern_laplace(num_samples, grid_size, sigma_sq0, ls0, s0, seed=42)
    z1 = generate_matern_laplace(num_samples, grid_size, sigma_sq1, ls1, s1, seed=45)

    t_min, t_max = 1e-4, 1 - 1e-4

    # Standard schedule
    print("  Running standard schedule...")
    drift_std = make_standard_drift()
    out_std = integrate_ode(drift_std, z0, param0, param1, grid_size, t_min, t_max, rk_steps_standard)

    # Scaled schedule
    print("  Running scaled schedule...")
    drift_sc, ratio = make_scaled_drift(param0, param1, grid_size)
    print(f"  Spectral ratio at Nyquist: {ratio:.4e}")
    out_sc = integrate_ode(drift_sc, z0, param0, param1, grid_size, t_min, t_max, rk_steps_scaled)

    # Compute spectra
    kvals, spec_truth = get_Fourier_spectrum(z1)
    _, spec_std = get_Fourier_spectrum(out_std)
    _, spec_sc = get_Fourier_spectrum(out_sc)

    # Per-band relative error
    rel_err_std = np.abs(spec_std - spec_truth) / spec_truth
    rel_err_sc = np.abs(spec_sc - spec_truth) / spec_truth

    print(f"\n  Per-band relative error |S_gen - S_truth| / S_truth:")
    print(f"  {'k-band':>8}  {'Standard':>12}  {'Scaled':>12}")
    print(f"  {'-'*36}")
    for i in range(len(kvals)):
        print(f"  {kvals[i]:8.1f}  {rel_err_std[i]:12.4e}  {rel_err_sc[i]:12.4e}")

    print(f"\n  Mean rel error — Standard: {rel_err_std.mean():.4e} | Scaled: {rel_err_sc.mean():.4e}")
    print(f"  Max  rel error — Standard: {rel_err_std.max():.4e} | Scaled: {rel_err_sc.max():.4e}")

    return {
        'grid_size': grid_size,
        'kvals': kvals,
        'spec_truth': spec_truth,
        'spec_std': spec_std,
        'spec_sc': spec_sc,
        'rel_err_std': rel_err_std,
        'rel_err_sc': rel_err_sc,
        'rk_steps_standard': rk_steps_standard,
        'rk_steps_scaled': rk_steps_scaled,
        'ratio': ratio,
    }

# ─── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    resolutions = [32, 64, 128]
    # Standard needs many more steps; scaled works with 20
    configs = {32: (80, 20), 64: (80, 20), 128: (80, 20)}

    results = {}
    for res in resolutions:
        rk_std, rk_sc = configs[res]
        results[res] = run_experiment(res, num_samples=2000, rk_steps_standard=rk_std, rk_steps_scaled=rk_sc)

    # ─── Plot: 3 panels (one per resolution) ──────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, res in enumerate(resolutions):
        r = results[res]

        # Top row: spectra comparison
        ax = axes[0, col]
        ax.plot(r['kvals'], r['spec_truth'], 'k-', lw=2, label='Truth')
        ax.plot(r['kvals'], r['spec_std'], 'r--', lw=1.5, marker='s', markersize=4, markevery=max(1, len(r['kvals'])//10),
                label=f'Standard (RK4, {r["rk_steps_standard"]} steps)')
        ax.plot(r['kvals'], r['spec_sc'], 'b--', lw=1.5, marker='o', markersize=4, markevery=max(1, len(r['kvals'])//10),
                label=f'Scaled (RK4, {r["rk_steps_scaled"]} steps)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Energy spectrum')
        ax.set_title(f'Resolution {res}x{res}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: per-band relative error
        ax = axes[1, col]
        ax.plot(r['kvals'], r['rel_err_std'], 'r-o', lw=1.5, markersize=3,
                label=f'Standard (RK4, {r["rk_steps_standard"]} steps)')
        ax.plot(r['kvals'], r['rel_err_sc'], 'b-o', lw=1.5, markersize=3,
                label=f'Scaled (RK4, {r["rk_steps_scaled"]} steps)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Relative error')
        ax.set_title(f'Per-band relative error ({res}x{res})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='1% error')

    plt.tight_layout()
    plt.savefig('spectrum_comparison_standard_vs_scaled.png', dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to spectrum_comparison_standard_vs_scaled.png")
    plt.show()
