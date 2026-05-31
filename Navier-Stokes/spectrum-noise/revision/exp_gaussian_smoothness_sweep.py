"""
Experiment for the revision: noise smoothness sweep on synthetic Gaussian.

Targets Reviewer 2 Major B1: "the paper should focus on different smoothness
levels of the spectrum noise, and specify the smoothness of the data."

We fix the data smoothness at s_1=3 and sweep noise smoothness s_0 in {0,1,2,3}
on a 128x128 Matern-like Gaussian random field.

Drift is closed-form (Section 3.1 of the paper), so no neural net is needed.
The noise spectrum and the data spectrum are diagonal in the Fourier basis with
specifications c_0(m) = sigma_0^2 (4 pi^2 |m|^2 + tau_0^2)^{-s_0} (and similarly
for c_1). The drift in Fourier space is

    tilde b_t(m) = (alpha_t alpha_t' c_0 + beta_t beta_t' c_1) /
                   (alpha_t^2 c_0 + beta_t^2 c_1).

We integrate via fixed-step RK4. Three random seeds, 2000 samples per seed.

Output: relative spectrum error in low/mid/high wavenumber bands at varying
RK4 step counts.

Run from the repo root with:
    python exp_gaussian_smoothness_sweep.py
"""

import math
import os
import time as time_mod
import numpy as np
import torch
import scipy.stats as stats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Sampling Matern-like Gaussian field on [0,1]^2 ───────────────────

def sample_matern(num_samples, grid_size, sigma_sq, length_scale, s, seed):
    """Sample from N(0, sigma_sq * (-Delta + ls^2 I)^{-s}) on the d=2 torus.

    Drawn in Fourier space; matches the formulation in Section 3.1.3 of the paper.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    freq = torch.fft.fftfreq(grid_size, device=DEVICE) * 2 * math.pi * grid_size
    fx, fy = torch.meshgrid(freq, freq, indexing='ij')
    laplacian = fx ** 2 + fy ** 2
    # spectral density per mode
    spec = sigma_sq * (laplacian + length_scale ** 2) ** (-s)
    spec[0, 0] = 0.0  # zero-mean
    spec = spec.unsqueeze(0)
    # complex standard normal of size (num_samples, grid_size, grid_size)
    re = torch.randn(num_samples, grid_size, grid_size, generator=g, device=DEVICE)
    im = torch.randn(num_samples, grid_size, grid_size, generator=g, device=DEVICE)
    noise = re + 1j * im
    sample = torch.fft.ifft2(torch.sqrt(spec) * noise, norm='forward').real
    return sample

# ─── Closed-form Gaussian drift in Fourier space ────────────────────

def compute_spectral_density(grid_size, sigma_sq, ls, s):
    freq = torch.fft.fftfreq(grid_size, device=DEVICE) * 2 * math.pi * grid_size
    fx, fy = torch.meshgrid(freq, freq, indexing='ij')
    lap = fx ** 2 + fy ** 2
    return sigma_sq * (lap + ls ** 2) ** (-s)


def linear_schedule(t):
    return 1.0 - t, -1.0, t, 1.0


def make_drift_linear(sd0, sd1):
    def drift(z, t):
        a, da, b, db = linear_schedule(t)
        zfft = torch.fft.fft2(z, norm='forward')
        ratio = (a * da * sd0 + b * db * sd1) / (a ** 2 * sd0 + b ** 2 * sd1 + 1e-30)
        return torch.fft.ifft2(ratio[None] * zfft, norm='forward').real
    return drift


def integrate_rk4(drift, z0, steps, t_min=1e-3, t_max=1.0 - 1e-3):
    tgrid = torch.linspace(t_min, t_max, steps + 1, device=DEVICE)
    z = z0.clone()
    for i in range(steps):
        ti = tgrid[i].item()
        dt = (tgrid[i + 1] - tgrid[i]).item()
        k1 = drift(z, ti)
        k2 = drift(z + 0.5 * dt * k1, ti + 0.5 * dt)
        k3 = drift(z + 0.5 * dt * k2, ti + 0.5 * dt)
        k4 = drift(z + dt * k3, ti + dt)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z

# ─── Spectrum diagnostics ───────────────────────────────────────────

def radial_spectrum(field):
    """Return (kvals, S(k)) where S is enstrophy/energy spectrum."""
    field_cpu = field.detach().cpu()
    fhat = torch.fft.fftn(field_cpu, dim=(1, 2), norm='forward')
    amp2 = (fhat.abs() ** 2).mean(dim=0).numpy()
    npix = amp2.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kxg, kyg = np.meshgrid(kfreq, kfreq, indexing='ij')
    knrm = np.sqrt(kxg ** 2 + kyg ** 2).flatten()
    amp_flat = amp2.flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, amp_flat, statistic='mean', bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins


def banded_relative_error(kvals, S_gen, S_truth, band):
    lo, hi = band
    mask = (kvals >= lo) & (kvals < hi)
    if mask.sum() == 0:
        return float('nan')
    rel = np.abs(S_gen[mask] - S_truth[mask]) / np.abs(S_truth[mask])
    return float(rel.mean())

# ─── Single experiment ──────────────────────────────────────────────

def run_one(grid_size, s0, s1, sigma_sq0, sigma_sq1, ls0, ls1,
            steps_list, num_samples=2000, seed=0):
    torch.manual_seed(seed)
    sd0 = compute_spectral_density(grid_size, sigma_sq0, ls0, s0)
    sd1 = compute_spectral_density(grid_size, sigma_sq1, ls1, s1)

    z0 = sample_matern(num_samples, grid_size, sigma_sq0, ls0, s0, seed=seed * 7 + 1)
    truth = sample_matern(num_samples, grid_size, sigma_sq1, ls1, s1, seed=seed * 7 + 2)
    drift = make_drift_linear(sd0, sd1)
    kvals_truth, S_truth = radial_spectrum(truth)

    out = {}
    for steps in steps_list:
        t0 = time_mod.time()
        gen = integrate_rk4(drift, z0, steps)
        wall = time_mod.time() - t0
        kvals, S_gen = radial_spectrum(gen)
        out[steps] = {
            'wall': wall,
            'low': banded_relative_error(kvals, S_gen, S_truth, (1, 8)),
            'mid': banded_relative_error(kvals, S_gen, S_truth, (8, 24)),
            'high': banded_relative_error(kvals, S_gen, S_truth, (24, grid_size // 2 + 1)),
            'mean_all': float(np.mean(np.abs(S_gen[1:] - S_truth[1:]) / np.abs(S_truth[1:]))),
        }
    return out

# ─── Main: noise-smoothness sweep at 128x128 with 3 seeds ──────────

def main():
    grid = 128
    s1 = 3
    ls0 = ls1 = 1.0
    sigma_sq1 = ((2 * math.pi) ** 2 + ls1 ** 2) ** s1

    seeds = [0, 1, 2]
    s0_list = [0, 1, 2, 3]
    steps_list = [5, 10, 20, 40, 80]

    print(f"\n=== Noise smoothness sweep, grid={grid}, s1={s1} ===")
    print(f"{'s0':>3}  {'steps':>6}  {'high band':>22}  {'mid band':>22}  {'wall (s)':>10}")
    print('-' * 80)

    results_by_s0 = {}
    for s0 in s0_list:
        sigma_sq0 = ((2 * math.pi) ** 2 + ls0 ** 2) ** s0  # match the convention in main.tex
        # We use sigma_0=1 (white-noise convention) for s0=0; otherwise above formula.
        if s0 == 0:
            sigma_sq0 = 1.0

        per_steps_high = {s: [] for s in steps_list}
        per_steps_mid = {s: [] for s in steps_list}
        per_steps_wall = {s: [] for s in steps_list}
        for seed in seeds:
            res = run_one(grid, s0, s1, sigma_sq0, sigma_sq1, ls0, ls1, steps_list,
                          num_samples=2000, seed=seed)
            for s in steps_list:
                per_steps_high[s].append(res[s]['high'])
                per_steps_mid[s].append(res[s]['mid'])
                per_steps_wall[s].append(res[s]['wall'])

        results_by_s0[s0] = {}
        for s in steps_list:
            high_arr = np.array(per_steps_high[s])
            mid_arr = np.array(per_steps_mid[s])
            wall_arr = np.array(per_steps_wall[s])
            results_by_s0[s0][s] = {
                'high_mean': high_arr.mean(), 'high_std': high_arr.std(),
                'mid_mean': mid_arr.mean(), 'mid_std': mid_arr.std(),
                'wall_mean': wall_arr.mean(),
            }
            print(f"{s0:>3d}  {s:>6d}  "
                  f"{high_arr.mean():>10.3e} ± {high_arr.std():.3e}  "
                  f"{mid_arr.mean():>10.3e} ± {mid_arr.std():.3e}  "
                  f"{wall_arr.mean():>10.3f}")

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'gaussian_smoothness_sweep_results.txt')
    with open(out_path, 'w') as f:
        f.write("# Gaussian smoothness sweep (128x128, s1=3, 3 seeds, 2000 samples)\n")
        f.write("# columns: s0, steps, high_mean, high_std, mid_mean, mid_std, wall(s)\n")
        for s0 in s0_list:
            for s in steps_list:
                r = results_by_s0[s0][s]
                f.write(f"{s0}\t{s}\t{r['high_mean']:.6e}\t{r['high_std']:.6e}\t"
                        f"{r['mid_mean']:.6e}\t{r['mid_std']:.6e}\t{r['wall_mean']:.4f}\n")
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
