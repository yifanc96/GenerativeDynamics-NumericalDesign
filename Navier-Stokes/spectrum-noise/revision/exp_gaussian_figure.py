"""
Generate the smoothness-sweep figure (high-band error vs RK4 steps for s_0 in
{0,1,2,3} at s_1=3, 128x128) and the wavenumber-dep-vs-scalar comparison panel.

These plots accompany the SISC revision (Reviewer 2.B1, Reviewer 1.Q2).
"""

import math
import os
import numpy as np
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_MIN, T_MAX = 1e-4, 1.0 - 1e-4

# ─── shared helpers ─────────────────────────────────────────────────

def sample_matern(num_samples, grid_size, sigma_sq, length_scale, s, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    freq = torch.fft.fftfreq(grid_size, device=DEVICE) * 2 * math.pi * grid_size
    fx, fy = torch.meshgrid(freq, freq, indexing='ij')
    laplacian = fx ** 2 + fy ** 2
    spec = sigma_sq * (laplacian + length_scale ** 2) ** (-s)
    spec[0, 0] = 0.0
    spec = spec.unsqueeze(0)
    re = torch.randn(num_samples, grid_size, grid_size, generator=g, device=DEVICE)
    im = torch.randn(num_samples, grid_size, grid_size, generator=g, device=DEVICE)
    noise = re + 1j * im
    return torch.fft.ifft2(torch.sqrt(spec) * noise, norm='forward').real


def spec_density(grid_size, sigma_sq, ls, s):
    freq = torch.fft.fftfreq(grid_size, device=DEVICE) * 2 * math.pi * grid_size
    fx, fy = torch.meshgrid(freq, freq, indexing='ij')
    sd = sigma_sq * (fx ** 2 + fy ** 2 + ls ** 2) ** (-s)
    sd[0, 0] = 0.0
    return sd


def linear_schedule(t):
    return 1.0 - t, -1.0, t, 1.0


def drift_linear(z, t, sd0, sd1):
    a, da, b, db = linear_schedule(t)
    zfft = torch.fft.fft2(z, norm='forward')
    ratio = (a * da * sd0 + b * db * sd1) / (a ** 2 * sd0 + b ** 2 * sd1 + 1e-40)
    return torch.fft.ifft2(ratio[None] * zfft, norm='forward').real


def scalar_scaled_schedule_factory(lambda_star):
    r = float(lambda_star)
    log_r = math.log(r)
    inv_r_minus_1 = 1.0 / (r - 1.0)

    def alpha_dot_over_alpha(t):
        return -0.5 * (r ** t) * log_r / (r - r ** t + 1e-300)

    def beta_dot_over_beta(t):
        return 0.5 * (r ** t) * log_r / (r ** t - 1.0 + 1e-300)

    def sched(t):
        a2 = max((r - r ** t) * inv_r_minus_1, 1e-30)
        b2 = max((r ** t - 1.0) * inv_r_minus_1, 1e-30)
        a = math.sqrt(a2)
        b = math.sqrt(b2)
        da = a * alpha_dot_over_alpha(t)
        db = b * beta_dot_over_beta(t)
        return a, da, b, db
    return sched


def drift_scalar_scaled_factory(sd0, sd1, lambda_star):
    sched = scalar_scaled_schedule_factory(lambda_star)
    def drift(z, t):
        a, da, b, db = sched(t)
        zfft = torch.fft.fft2(z, norm='forward')
        ratio = (a * da * sd0 + b * db * sd1) / (a ** 2 * sd0 + b ** 2 * sd1 + 1e-40)
        return torch.fft.ifft2(ratio[None] * zfft, norm='forward').real
    return drift


def drift_wavenumber_dep_factory(sd0, sd1):
    eps = 1e-40
    log_ratio = 0.5 * (torch.log(sd1 + eps) - torch.log(sd0 + eps))
    def drift(z, t):
        zfft = torch.fft.fft2(z, norm='forward')
        return torch.fft.ifft2(log_ratio[None] * zfft, norm='forward').real
    return drift


def integrate_rk4(drift, z0, steps, t_min=T_MIN, t_max=T_MAX):
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


def radial_spectrum(field):
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


def banded(kvals, S_gen, S_truth, lo, hi):
    mask = (kvals >= lo) & (kvals < hi)
    rel = np.abs(S_gen[mask] - S_truth[mask]) / np.abs(S_truth[mask])
    return float(rel.mean())


# ─── Figure 1: smoothness sweep ─────────────────────────────────────

def fig_smoothness_sweep():
    grid = 128
    s1 = 3
    ls = 1.0
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1
    sd1 = spec_density(grid, sig1, ls, s1)
    s0_list = [0, 1, 2, 3]
    steps_list = [5, 10, 20, 40, 80, 160]
    seeds = [0, 1, 2]

    high_means = {s0: [] for s0 in s0_list}
    high_stds = {s0: [] for s0 in s0_list}

    print("\n=== Smoothness sweep figure ===")
    for s0 in s0_list:
        sig0 = 1.0 if s0 == 0 else ((2 * math.pi) ** 2 + ls ** 2) ** s0
        sd0 = spec_density(grid, sig0, ls, s0)
        for steps in steps_list:
            highs = []
            for seed in seeds:
                z0 = sample_matern(2000, grid, sig0, ls, s0, seed=seed * 7 + 1)
                truth = sample_matern(2000, grid, sig1, ls, s1, seed=seed * 7 + 2)
                gen = integrate_rk4(lambda z, t: drift_linear(z, t, sd0, sd1), z0, steps=steps)
                kvals, S_truth = radial_spectrum(truth)
                _, S_gen = radial_spectrum(gen)
                highs.append(banded(kvals, S_gen, S_truth, 24, grid // 2 + 1))
            highs = np.array(highs)
            high_means[s0].append(highs.mean())
            high_stds[s0].append(highs.std())
            print(f"s0={s0} steps={steps:4d} high {highs.mean():.3e} ± {highs.std():.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    colors = {0: 'C3', 1: 'C1', 2: 'C2', 3: 'C0'}
    for s0 in s0_list:
        m = np.array(high_means[s0])
        sd = np.array(high_stds[s0])
        label = f'$s_0={s0}$' + (' (white)' if s0 == 0 else (' (matched)' if s0 == 3 else ''))
        ax.errorbar(steps_list, m, yerr=sd, marker='o', ms=7, lw=2.0, capsize=3,
                    label=label, color=colors[s0])
    ax.set_xlabel('RK4 steps')
    ax.set_ylabel(r'Relative error')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title(r'Smoothness sweep ($s_1{=}3$, linear schedule)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xticks(steps_list)
    ax.set_xticklabels([str(x) for x in steps_list])
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'gaussian_smoothness_sweep.pdf')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"saved: {out_path}")


# ─── Figure 2: wavenumber-dep vs scalar ─────────────────────────────

def fig_wavenumber_vs_scalar():
    grid = 128
    s0, s1, ls = 0, 3, 1.0
    sig0 = 1.0
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1
    sd0 = spec_density(grid, sig0, ls, s0)
    sd1 = spec_density(grid, sig1, ls, s1)
    nyq = grid // 2
    lambda_star = float((sd1[nyq, nyq] / sd0[nyq, nyq]).item())
    print(f"\n=== Wavenumber-dep vs scalar at {grid}, lambda*={lambda_star:.3e} ===")

    steps_list = [5, 10, 20, 40, 80, 160]
    seeds = [0, 1, 2]
    means = {'linear': [], 'scalar': [], 'wavenum': []}
    stds = {'linear': [], 'scalar': [], 'wavenum': []}

    for steps in steps_list:
        rec = {'linear': [], 'scalar': [], 'wavenum': []}
        for seed in seeds:
            z0 = sample_matern(2000, grid, sig0, ls, s0, seed=seed * 7 + 1)
            truth = sample_matern(2000, grid, sig1, ls, s1, seed=seed * 7 + 2)
            kvals, S_truth = radial_spectrum(truth)
            for label, drift in [
                ('linear', lambda z, t: drift_linear(z, t, sd0, sd1)),
                ('scalar', drift_scalar_scaled_factory(sd0, sd1, lambda_star)),
                ('wavenum', drift_wavenumber_dep_factory(sd0, sd1)),
            ]:
                gen = integrate_rk4(drift, z0, steps=steps)
                _, S_gen = radial_spectrum(gen)
                rec[label].append(banded(kvals, S_gen, S_truth, 24, grid // 2 + 1))
        for label in rec:
            arr = np.array(rec[label])
            means[label].append(arr.mean())
            stds[label].append(arr.std())
        print(f"steps={steps:4d}: linear {means['linear'][-1]:.3e}, "
              f"scalar {means['scalar'][-1]:.3e}, wavenum {means['wavenum'][-1]:.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    style = {'linear': ('C3', 's', 'Linear schedule'),
             'scalar': ('C0', 'o', r'Scalar designed schedule (single $\lambda^\star$)'),
             'wavenum': ('C2', '^', 'Wavenumber-dependent schedule')}
    for label in ['linear', 'scalar', 'wavenum']:
        c, m, name = style[label]
        ax.errorbar(steps_list, means[label], yerr=stds[label], marker=m, ms=7, lw=2.0, capsize=3,
                    label=name, color=c)
    ax.set_xlabel('RK4 steps')
    ax.set_ylabel(r'Relative error')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title('Wavenumber-dependent vs scalar schedule')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xticks(steps_list)
    ax.set_xticklabels([str(x) for x in steps_list])
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'gaussian_wavenumber_vs_scalar.pdf')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"saved: {out_path}")
    return means, stds


if __name__ == '__main__':
    fig_smoothness_sweep()
    fig_wavenumber_vs_scalar()
