"""
Closed-form Gaussian experiments for the SISC revision.

Targets:
  R2.B1 — Smoothness sweep: vary s_0 in {0,1,2,3} with s_1=3 at 128x128
          (Sec 4 of the paper).
  R2.B2 — NFE / wall-clock comparison across resolutions.
  R1.Q2 — Wavenumber-dependent vs scalar scale-adaptive schedule comparison
          on the Matern-like Gaussian target (Sec 5.1 of the paper).

The Gaussian-target drift is in closed form (Section 3.1 of the paper, eq.
b_t = B(t) x with B(t) computed from C_0, C_1 and the schedule). No neural
network is needed, so the experiments are reproducible from scratch.

We follow the conventions of `reproduce_spectrum_comparison.py` so that our
numbers can be cross-checked: t_min = 1e-4, t_max = 1 - 1e-4; Matern-like
covariance C = sigma^2 (-Delta + tau^2 I)^{-s} on the 2D torus discretized as
the Fourier-truncated GRF; per-band relative error reported on the radial
spectrum. The white-noise prior uses s_0=0, sigma_0 = 1.

We also include a sanity-check that reproduces the existing
white-vs-scaled comparison from `reproduce_spectrum_comparison.py`.

Run from this directory with:
    /home/yifanchen/miniconda3/envs/gpu/bin/python exp_gaussian_revision.py
"""

import math
import os
import time as _t
import numpy as np
import torch
import scipy.stats as stats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] {DEVICE} | torch {torch.__version__}")

T_MIN, T_MAX = 1e-4, 1.0 - 1e-4

# ─── Sampling ───────────────────────────────────────────────────────

def sample_matern(num_samples, grid_size, sigma_sq, length_scale, s, seed):
    """N(0, sigma_sq * (-Delta + ls^2 I)^{-s}) on the torus, drawn in Fourier."""
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
    sd[0, 0] = 0.0  # exclude DC (zero-mean fields)
    return sd

# ─── Schedules ──────────────────────────────────────────────────────

def linear_schedule(t):
    return 1.0 - t, -1.0, t, 1.0


def scalar_scaled_schedule_factory(lambda_star):
    """Schedule \eqref{eqn-high-D-gaussian-alpha-beta} with worst-case lambda_star.

    alpha_t = sqrt((lstar - lstar^t)/(lstar - 1))
    beta_t  = sqrt((lstar^t - 1)/(lstar - 1))
    """
    r = float(lambda_star)
    log_r = math.log(r)
    inv_r_minus_1 = 1.0 / (r - 1.0)

    def alpha_dot_over_alpha(t):
        # alpha = sqrt((r - r^t)/(r - 1))
        # alpha_dot = -0.5 / alpha * r^t * log(r) / (r-1)
        # alpha_dot/alpha = -0.5 * r^t * log(r) / (r - r^t)
        return -0.5 * (r ** t) * log_r / (r - r ** t)

    def beta_dot_over_beta(t):
        # beta = sqrt((r^t - 1)/(r-1))
        # beta_dot = 0.5 / beta * r^t * log(r) / (r-1)
        # beta_dot/beta = 0.5 * r^t * log(r) / (r^t - 1)
        return 0.5 * (r ** t) * log_r / (r ** t - 1.0)

    def sched(t):
        a2 = (r - r ** t) * inv_r_minus_1
        b2 = (r ** t - 1.0) * inv_r_minus_1
        a = math.sqrt(max(a2, 1e-30))
        b = math.sqrt(max(b2, 1e-30))
        # use derivative-over-value forms; rebuild da, db
        da = a * alpha_dot_over_alpha(t)
        db = b * beta_dot_over_beta(t)
        return a, da, b, db

    return sched

# ─── Drifts ─────────────────────────────────────────────────────────

def drift_linear(z, t, sd0, sd1):
    a, da, b, db = linear_schedule(t)
    zfft = torch.fft.fft2(z, norm='forward')
    ratio = (a * da * sd0 + b * db * sd1) / (a ** 2 * sd0 + b ** 2 * sd1 + 1e-40)
    return torch.fft.ifft2(ratio[None] * zfft, norm='forward').real


def drift_scalar_scaled_factory(sd0, sd1, lambda_star):
    sched = scalar_scaled_schedule_factory(lambda_star)

    def drift(z, t):
        a, da, b, db = sched(t)
        zfft = torch.fft.fft2(z, norm='forward')
        ratio = (a * da * sd0 + b * db * sd1) / (a ** 2 * sd0 + b ** 2 * sd1 + 1e-40)
        return torch.fft.ifft2(ratio[None] * zfft, norm='forward').real
    return drift


def drift_wavenumber_dep_factory(sd0, sd1):
    """Wavenumber-dependent schedule: per Fourier mode use the
    scalar_scaled_schedule with lambda_m = c_1(m) / c_0(m).

    Closed-form Fourier-space drift: tilde b_t(m) = (1/2) log(c_1(m)/c_0(m)).

    See Section 5.1 of the paper, "wavenumber-dependent linear interpolation in
    Fourier space".
    """
    # Per-mode log ratio (constant in time).
    eps = 1e-40
    log_ratio = 0.5 * (torch.log(sd1 + eps) - torch.log(sd0 + eps))  # constant in t

    def drift(z, t):
        zfft = torch.fft.fft2(z, norm='forward')
        return torch.fft.ifft2(log_ratio[None] * zfft, norm='forward').real
    return drift

# ─── ODE integrator ─────────────────────────────────────────────────

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

# ─── Spectrum diagnostics ───────────────────────────────────────────

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


def banded_relative_error(kvals, S_gen, S_truth, lo, hi):
    mask = (kvals >= lo) & (kvals < hi)
    if mask.sum() == 0:
        return float('nan')
    rel = np.abs(S_gen[mask] - S_truth[mask]) / np.abs(S_truth[mask])
    return float(rel.mean())


# ─── Sanity check: reproduce existing standard-vs-scaled at 128 ─────

def sanity_reproduce(grid=128, num_samples=2000, seed=0):
    print(f"\n[sanity] reproducing reproduce_spectrum_comparison.py at {grid}x{grid}")
    s0 = 0
    s1 = 3
    ls = 1.0
    sig0 = 1.0  # white noise convention
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1

    torch.manual_seed(seed)
    sd0 = spec_density(grid, sig0, ls, s0)
    sd1 = spec_density(grid, sig1, ls, s1)
    z0 = sample_matern(num_samples, grid, sig0, ls, s0, seed=seed * 7 + 1)
    truth = sample_matern(num_samples, grid, sig1, ls, s1, seed=seed * 7 + 2)

    # standard linear, 80 steps
    g_std = integrate_rk4(lambda z, t: drift_linear(z, t, sd0, sd1), z0, steps=80)
    # scalar scaled with lambda* = sd1/sd0 at Nyquist
    nyq = grid // 2
    lambda_star = float((sd1[nyq, nyq] / sd0[nyq, nyq]).item())
    print(f"[sanity] lambda* (Nyquist) = {lambda_star:.4e}")
    g_sc = integrate_rk4(drift_scalar_scaled_factory(sd0, sd1, lambda_star), z0, steps=20)

    kvals, S_truth = radial_spectrum(truth)
    _, S_std = radial_spectrum(g_std)
    _, S_sc = radial_spectrum(g_sc)
    rel_std = np.abs(S_std - S_truth) / np.abs(S_truth)
    rel_sc = np.abs(S_sc - S_truth) / np.abs(S_truth)
    print(f"[sanity] Standard linear (80 steps): mean rel err = {rel_std.mean():.3e}, max = {rel_std.max():.3e}")
    print(f"[sanity] Scalar scaled (20 steps):   mean rel err = {rel_sc.mean():.3e},  max = {rel_sc.max():.3e}")
    print("[sanity] expected (from reproduce_spectrum_comparison.py 128x128):")
    print("[sanity]   Standard mean ~ 1.49e+02, max ~ 6.80e+02")
    print("[sanity]   Scaled   mean ~ 6.91e-02, max ~ 9.66e-02")
    return rel_std.mean(), rel_sc.mean()

# ─── Smoothness sweep (R2.B1) ───────────────────────────────────────

def smoothness_sweep(grid=128, num_samples=2000, seeds=(0, 1, 2)):
    print(f"\n=== Smoothness sweep, grid={grid}, s1=3 (3 seeds) ===")
    s1 = 3
    ls = 1.0
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1
    s0_list = [0, 1, 2, 3]
    steps_list = [10, 20, 40, 80]

    rows = []
    for s0 in s0_list:
        sig0 = 1.0 if s0 == 0 else ((2 * math.pi) ** 2 + ls ** 2) ** s0  # conventions
        for steps in steps_list:
            highs, mids, walls = [], [], []
            for seed in seeds:
                torch.manual_seed(seed)
                sd0 = spec_density(grid, sig0, ls, s0)
                sd1 = spec_density(grid, sig1, ls, s1)
                z0 = sample_matern(num_samples, grid, sig0, ls, s0, seed=seed * 7 + 1)
                truth = sample_matern(num_samples, grid, sig1, ls, s1, seed=seed * 7 + 2)
                t0 = _t.time()
                gen = integrate_rk4(lambda z, t: drift_linear(z, t, sd0, sd1), z0, steps=steps)
                walls.append(_t.time() - t0)
                kvals, S_truth = radial_spectrum(truth)
                _, S_gen = radial_spectrum(gen)
                highs.append(banded_relative_error(kvals, S_gen, S_truth, 24, grid // 2 + 1))
                mids.append(banded_relative_error(kvals, S_gen, S_truth, 8, 24))
            highs, mids, walls = np.array(highs), np.array(mids), np.array(walls)
            row = (s0, steps, highs.mean(), highs.std(), mids.mean(), mids.std(), walls.mean())
            rows.append(row)
            print(f"s0={s0} | steps={steps:3d} | high {highs.mean():.3e} ± {highs.std():.3e}"
                  f" | mid {mids.mean():.3e} ± {mids.std():.3e}"
                  f" | wall {walls.mean():.3f}s")
    return rows

# ─── NFE / wall-clock comparison (R2.B2) ────────────────────────────

def nfe_wall_clock(grids=(32, 64, 128), num_samples=2000, seeds=(0, 1, 2)):
    print(f"\n=== NFE / wall-clock to high-band err <= 5%, varying resolution ===")
    s1 = 3
    ls = 1.0
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1

    summary = []
    for grid in grids:
        # find the smallest steps for which mean high-band err <= 0.05 over seeds
        # for both white-noise + linear and matched-spectrum + linear
        sd1 = spec_density(grid, sig1, ls, s1)
        truth_per_seed = {}
        for seed in seeds:
            truth_per_seed[seed] = sample_matern(num_samples, grid, sig1, ls, s1, seed=seed * 7 + 2)

        # white noise
        sig0w = 1.0
        sd0w = spec_density(grid, sig0w, ls, 0)
        # matched spectrum
        sd0m = sd1.clone()

        def run(sd0_use, steps_list, label):
            for steps in steps_list:
                highs, walls = [], []
                for seed in seeds:
                    torch.manual_seed(seed)
                    z0 = sample_matern(num_samples, grid, 1.0 if label == 'white' else sig1,
                                       ls, 0 if label == 'white' else s1, seed=seed * 7 + 1)
                    t0 = _t.time()
                    gen = integrate_rk4(lambda z, t: drift_linear(z, t, sd0_use, sd1), z0, steps=steps)
                    walls.append(_t.time() - t0)
                    kvals, S_truth = radial_spectrum(truth_per_seed[seed])
                    _, S_gen = radial_spectrum(gen)
                    highs.append(banded_relative_error(kvals, S_gen, S_truth, 24, grid // 2 + 1))
                if np.mean(highs) <= 0.05:
                    return steps, np.mean(highs), np.mean(walls)
            return None, np.mean(highs), np.mean(walls)

        # search for smallest step count meeting threshold
        steps_grid_white = [5, 10, 20, 40, 80, 160, 320]
        steps_grid_match = [3, 5, 10, 20]

        st_w, err_w, wall_w = run(sd0w, steps_grid_white, 'white')
        st_m, err_m, wall_m = run(sd0m, steps_grid_match, 'match')

        print(f"grid {grid}: white-noise smallest steps for <=5% high err = {st_w} (err {err_w:.3e}, wall {wall_w:.3f}s)")
        print(f"grid {grid}: match-spec  smallest steps for <=5% high err = {st_m} (err {err_m:.3e}, wall {wall_m:.3f}s)")
        summary.append((grid, st_w, wall_w, st_m, wall_m))
    return summary

# ─── Wavenumber-dependent vs scalar (R1.Q2) ────────────────────────

def wavenumber_vs_scalar(grid=128, num_samples=2000, seeds=(0, 1, 2)):
    print(f"\n=== Wavenumber-dependent vs scalar scaled schedule, grid={grid} ===")
    s0 = 0
    s1 = 3
    ls = 1.0
    sig0 = 1.0
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1
    sd0 = spec_density(grid, sig0, ls, s0)
    sd1 = spec_density(grid, sig1, ls, s1)
    nyq = grid // 2
    lambda_star = float((sd1[nyq, nyq] / sd0[nyq, nyq]).item())
    print(f"lambda* (Nyquist) = {lambda_star:.4e}")

    steps_list = [5, 10, 20, 40, 80]
    rows = []
    for steps in steps_list:
        highs_lin, highs_sc, highs_wn = [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            z0 = sample_matern(num_samples, grid, sig0, ls, s0, seed=seed * 7 + 1)
            truth = sample_matern(num_samples, grid, sig1, ls, s1, seed=seed * 7 + 2)
            kvals, S_truth = radial_spectrum(truth)

            gen_lin = integrate_rk4(lambda z, t: drift_linear(z, t, sd0, sd1), z0, steps=steps)
            gen_sc = integrate_rk4(drift_scalar_scaled_factory(sd0, sd1, lambda_star), z0, steps=steps)
            gen_wn = integrate_rk4(drift_wavenumber_dep_factory(sd0, sd1), z0, steps=steps)

            _, S_lin = radial_spectrum(gen_lin)
            _, S_sc = radial_spectrum(gen_sc)
            _, S_wn = radial_spectrum(gen_wn)
            highs_lin.append(banded_relative_error(kvals, S_lin, S_truth, 24, grid // 2 + 1))
            highs_sc.append(banded_relative_error(kvals, S_sc, S_truth, 24, grid // 2 + 1))
            highs_wn.append(banded_relative_error(kvals, S_wn, S_truth, 24, grid // 2 + 1))

        highs_lin = np.array(highs_lin)
        highs_sc = np.array(highs_sc)
        highs_wn = np.array(highs_wn)
        row = (steps, highs_lin.mean(), highs_lin.std(),
               highs_sc.mean(), highs_sc.std(),
               highs_wn.mean(), highs_wn.std())
        rows.append(row)
        print(f"steps={steps:3d} | linear {highs_lin.mean():.3e} ± {highs_lin.std():.3e}"
              f" | scalar {highs_sc.mean():.3e} ± {highs_sc.std():.3e}"
              f" | wavenum {highs_wn.mean():.3e} ± {highs_wn.std():.3e}")
    return rows


# ─── Main ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) sanity reproduce
    sanity_reproduce()

    # 2) smoothness sweep
    rows_sweep = smoothness_sweep()

    # 3) NFE / wall-clock
    summary_nfe = nfe_wall_clock()

    # 4) wavenumber-dep vs scalar
    rows_wn = wavenumber_vs_scalar()

    # save text outputs
    with open(os.path.join(out_dir, 'gaussian_results.txt'), 'w') as f:
        f.write("# === Smoothness sweep (s1=3, 128x128, 3 seeds, 2000 samples) ===\n")
        f.write("# s0  steps  high_mean  high_std  mid_mean  mid_std  wall(s)\n")
        for r in rows_sweep:
            f.write("\t".join(f"{x:.6e}" if isinstance(x, float) else str(x) for x in r) + "\n")
        f.write("\n# === NFE/wall-clock to high-band err <= 5%% ===\n")
        f.write("# grid, white_steps, white_wall(s), match_steps, match_wall(s)\n")
        for r in summary_nfe:
            f.write("\t".join(str(x) for x in r) + "\n")
        f.write("\n# === Wavenumber-dep vs scalar (white-noise prior, 128x128, 3 seeds) ===\n")
        f.write("# steps, lin_mean, lin_std, scalar_mean, scalar_std, wn_mean, wn_std\n")
        for r in rows_wn:
            f.write("\t".join(f"{x:.6e}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"\n[saved] {os.path.join(out_dir, 'gaussian_results.txt')}")
