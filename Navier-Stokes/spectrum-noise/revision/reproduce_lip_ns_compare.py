"""
Reproduce Lip paper (arXiv:2509.01629) result for NS:
Compare NN with standard schedule vs analytical Lip/scaled schedule.

- NN standard: trained flow matching model with α=1-t, β=t, RK4 integration
- Analytical Lip: drift computed from empirical S₀ (white noise) and S₁ (NS data)
  using the scaled interpolation α(t)=sqrt((r-r^t)/(r-1)), β(t)=sqrt((r^t-1)/(r-1))
  where r = S₁(k_max)/S₀(k_max).
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from unet import Unet


# ─── Data loading ─────────────────────────────────────────────────────

def load_ns_data(data_locs, hi_size, train_test_split):
    if isinstance(data_locs, str):
        data_locs = [data_locs]
    avg_pixel_norm = 3.0679163932800293
    all_data = []
    for loc in data_locs:
        data_raw, _ = torch.load(loc, weights_only=False)
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, data_raw.shape[2], data_raw.shape[3])
        if hi_size != data.shape[1]:
            data = nn.functional.interpolate(
                data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear'
            ).squeeze(1)
        all_data.append(data)
    data = torch.cat(all_data, dim=0)[:, None, :, :]
    num_train = int(data.shape[0] * train_test_split)
    return data[:num_train], data[num_train:]


# ─── Spectrum computation ─────────────────────────────────────────────

def get_energy_spectrum(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(1, 2), norm='forward')
    fourier_amp = (torch.abs(fhat)**2 * (2 * np.pi)).mean(dim=0)
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    fourier_flat = fourier_amp.numpy().flatten()
    laplace = knrm**2; laplace[0] = 1.0
    energy_flat = fourier_flat / laplace
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    enstrophy, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    enstrophy *= area_weight
    energy, _, _ = stats.binned_statistic(knrm, energy_flat, statistic='mean', bins=kbins)
    energy *= area_weight
    return kvals, enstrophy, energy


def get_spectral_density_2d(data):
    """Get the full 2D spectral density (not radially averaged)."""
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(1, 2), norm='forward')
    spec = (torch.abs(fhat)**2).mean(dim=0)
    return spec


# ─── NN model ─────────────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self, C, unet_channels, unet_dim_mults):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=C, out_channels=C,
            dim=unet_channels, dim_mults=unet_dim_mults,
            resnet_block_groups=8,
            learned_sinusoidal_cond=True,
            random_fourier_features=False,
            learned_sinusoidal_dim=32,
            attn_dim_head=32, attn_heads=4,
            use_classes=False,
        )
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


# ─── RK4 samplers ────────────────────────────────────────────────────

@torch.no_grad()
def rk4_sample_nn(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
    """Standard schedule with trained NN velocity."""
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0
    for i in range(len(tgrid) - 1):
        t_val = tgrid[i]
        dt = tgrid[i+1] - tgrid[i]
        ones = torch.ones(zt.shape[0], device=zt.device) * t_val
        k1 = model(zt, ones)
        k2 = model(zt + 0.5*dt*k1, ones + 0.5*dt)
        k3 = model(zt + 0.5*dt*k2, ones + 0.5*dt)
        k4 = model(zt + dt*k3, ones + dt)
        zt = zt + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


def make_lip_drift(S0_2d, S1_2d, grid_size):
    """
    Lip/scaled schedule analytical drift using empirical 2D spectra.
    r = S1/S0 at Nyquist; α(t) = sqrt((r - r^t)/(r-1)), β(t) = sqrt((r^t - 1)/(r-1))
    """
    r = (S1_2d[grid_size // 2, grid_size // 2] /
         S0_2d[grid_size // 2, grid_size // 2]).item()

    print(f"  Lip schedule: spectral ratio at Nyquist r = {r:.6e}")

    def alpha(t):
        return np.sqrt((r - r**t) / (r - 1))
    def alpha_dot(t):
        a = alpha(t)
        if abs(a) < 1e-30:
            return 0.0
        return -0.5 / a * (r**t) * np.log(r) / (r - 1)
    def beta(t):
        return np.sqrt((r**t - 1) / (r - 1))
    def beta_dot(t):
        b = beta(t)
        if abs(b) < 1e-30:
            return 0.0
        return 0.5 / b * (r**t) * np.log(r) / (r - 1)

    def drift(z, t):
        a, da = alpha(t), alpha_dot(t)
        b, db = beta(t), beta_dot(t)
        R_k = (a * da * S0_2d + b * db * S1_2d) / (a**2 * S0_2d + b**2 * S1_2d + 1e-30)
        zfft = torch.fft.fft2(z, dim=(2, 3), norm='forward')
        return torch.fft.ifft2(R_k[None, None] * zfft, dim=(2, 3), norm='forward').real

    return drift


def rk4_sample_analytical(drift_fn, z0, steps, t_min=1e-4, t_max=1-1e-4):
    """RK4 with analytical drift."""
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0.clone()
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i].item()
        dt = (tgrid[i+1] - tgrid[i]).item()
        k1 = drift_fn(zt, t_i)
        k2 = drift_fn(zt + 0.5 * dt * k1, t_i + 0.5 * dt)
        k3 = drift_fn(zt + 0.5 * dt * k2, t_i + 0.5 * dt)
        k4 = drift_fn(zt + dt * k3, t_i + dt)
        zt = zt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--ckpt', type=str, default='results/ns_gauss_base/model_final.pt')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    # ── Load data ──
    print("Loading NS data...")
    train_data, test_data = load_ns_data('../NSdata/data_file.pt', 128, 0.9)
    print(f"  train={train_data.shape[0]}, test={test_data.shape[0]}")

    num_eval = min(args.num_eval, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)
    grid_size = 128

    # ── Compute empirical 2D spectral densities ──
    print("Computing empirical spectral densities...")
    torch.manual_seed(123)
    z0_samples = torch.randn(2000, 1, grid_size, grid_size)
    S0_2d = get_spectral_density_2d(z0_samples.squeeze(1))
    S1_2d = get_spectral_density_2d(train_data[:2000].squeeze(1))
    S0_2d[0, 0] = 1e-30
    S1_2d[0, 0] = 1e-30
    print(f"  S0 range: [{S0_2d.min():.4e}, {S0_2d.max():.4e}]")
    print(f"  S1 range: [{S1_2d.min():.4e}, {S1_2d.max():.4e}]")

    # ── Build Lip analytical drift ──
    print("\nBuilding Lip schedule drift...")
    drift_lip = make_lip_drift(S0_2d, S1_2d, grid_size)

    # ── Load NN model ──
    print(f"\nLoading NN model: {args.ckpt}")
    model = Velocity(C=1, unet_channels=32, unet_dim_mults=(1,2,2,2)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()

    # ── Same initial noise for both methods ──
    torch.manual_seed(42)
    z0 = torch.randn(num_eval, 1, grid_size, grid_size)

    step_counts = [5, 10, 20, 50, 100]
    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}

    results_nn = {}
    results_lip = {}

    # ── NN standard schedule ──
    print("\n" + "="*80)
    print("NN standard schedule (trained model, α=1-t, β=t)")
    print("="*80)
    print(f"  {'steps':>6}  {'std_ratio':>10}  {'low':>8}  {'mid':>8}  {'high':>8}  {'mean':>8}")
    print(f"  {'-'*56}")
    for nsteps in step_counts:
        gen = rk4_sample_nn(model, z0.clone().to(device), nsteps)
        gen_sq = gen.squeeze(1).cpu()
        _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)
        std_ratio = gen_sq.std().item() / truth_sq.std().item()
        results_nn[nsteps] = (enst_gen, ener_gen, std_ratio)
        band_err = {bn: np.mean(np.abs(enst_truth[m] - enst_gen[m]) / (np.abs(enst_truth[m]) + 1e-20))
                    for bn, m in bands.items()}
        mean_err = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-20))
        print(f"  {nsteps:>6}  {std_ratio:>10.4f}  {band_err['low']:>8.4f}  {band_err['mid']:>8.4f}  {band_err['high']:>8.4f}  {mean_err:>8.4f}")

    # ── Analytical Lip schedule ──
    print("\n" + "="*80)
    print("Analytical Lip schedule (empirical spectra, scaled α/β)")
    print("="*80)
    print(f"  {'steps':>6}  {'std_ratio':>10}  {'low':>8}  {'mid':>8}  {'high':>8}  {'mean':>8}")
    print(f"  {'-'*56}")
    for nsteps in step_counts:
        gen = rk4_sample_analytical(drift_lip, z0.clone(), nsteps)
        gen_sq = gen.squeeze(1)
        _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)
        std_ratio = gen_sq.std().item() / truth_sq.std().item()
        results_lip[nsteps] = (enst_gen, ener_gen, std_ratio)
        band_err = {bn: np.mean(np.abs(enst_truth[m] - enst_gen[m]) / (np.abs(enst_truth[m]) + 1e-20))
                    for bn, m in bands.items()}
        mean_err = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-20))
        print(f"  {nsteps:>6}  {std_ratio:>10.4f}  {band_err['low']:>8.4f}  {band_err['mid']:>8.4f}  {band_err['high']:>8.4f}  {mean_err:>8.4f}")

    # ── Plots ──────────────────────────────────────────────────────────

    methods = {
        'NN-standard':    (results_nn,  'red',  '--', 's', 'NN standard schedule'),
        'Analytical-Lip': (results_lip, 'blue', '-',  'o', 'Analytical Lip schedule'),
    }

    # Figure 1: Spectra at 20 steps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ref_steps = 20

    ax = axes[0]
    ax.loglog(kvals, enst_truth, 'k-', lw=2.5, label='Truth', zorder=10)
    for key, (res, color, ls, marker, label) in methods.items():
        ax.loglog(kvals, res[ref_steps][0], ls, color=color, lw=1.5, marker=marker,
                 markersize=4, markevery=max(1, len(kvals)//10),
                 label=f'{label} ({ref_steps} steps)')
    ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Enstrophy', fontsize=12)
    ax.set_title(f'Enstrophy Spectrum (RK4, {ref_steps} steps)', fontsize=13)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(kvals, ener_truth, 'k-', lw=2.5, label='Truth', zorder=10)
    for key, (res, color, ls, marker, label) in methods.items():
        ax.loglog(kvals, res[ref_steps][1], ls, color=color, lw=1.5, marker=marker,
                 markersize=4, markevery=max(1, len(kvals)//10),
                 label=f'{label} ({ref_steps} steps)')
    ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'Energy Spectrum (RK4, {ref_steps} steps)', fontsize=13)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for key, (res, color, ls, marker, label) in methods.items():
        rel_err = np.abs(res[ref_steps][0] - enst_truth) / (np.abs(enst_truth) + 1e-20)
        ax.semilogy(kvals, rel_err, ls, color=color, lw=1.5, alpha=0.8, label=label)
    ax.axhline(y=0.1, color='gray', ls=':', alpha=0.5)
    ax.axhline(y=0.01, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Relative enstrophy error', fontsize=12)
    ax.set_title(f'Per-k Error ({ref_steps} steps)', fontsize=13)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(1e-3, 10)

    plt.tight_layout()
    plt.savefig('ns_lip_comparison_spectra.png', dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: ns_lip_comparison_spectra.png")

    # Figure 2: Band-averaged error vs steps
    band_edges = [(1, 8, 'low'), (8, 24, 'mid'), (24, 64, 'high')]
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (k_lo, k_hi, bname) in enumerate(band_edges):
        ax = axes2[ax_idx]
        mask = (kvals >= k_lo) & (kvals < k_hi)
        for key, (res, color, ls, marker, label) in methods.items():
            errs = []
            for ns in step_counts:
                err = np.mean(np.abs(enst_truth[mask] - res[ns][0][mask]) / (np.abs(enst_truth[mask]) + 1e-20))
                errs.append(err)
            ax.semilogy(step_counts, errs, ls, color=color, marker=marker, lw=2,
                       markersize=6, label=label)
        ax.set_xlabel('RK4 steps', fontsize=12)
        ax.set_ylabel('Rel. enstrophy error', fontsize=12)
        ax.set_title(f'{bname} band: k∈[{k_lo},{k_hi})', fontsize=13)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xticks(step_counts)

    plt.tight_layout()
    plt.savefig('ns_lip_comparison_bands.png', dpi=200, bbox_inches='tight')
    print(f"Figure saved: ns_lip_comparison_bands.png")

    # Figure 3: Mean error vs steps
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5.5))
    for key, (res, color, ls, marker, label) in methods.items():
        errs = [np.mean(np.abs(enst_truth - res[ns][0]) / (np.abs(enst_truth) + 1e-20))
                for ns in step_counts]
        ax3.semilogy(step_counts, errs, ls, color=color, marker=marker, lw=2.5,
                    markersize=8, label=label)
    ax3.set_xlabel('Number of RK4 steps', fontsize=13)
    ax3.set_ylabel('Mean relative enstrophy error', fontsize=13)
    ax3.set_title('NS: NN Standard vs Analytical Lip Schedule', fontsize=14)
    ax3.legend(fontsize=11); ax3.grid(True, alpha=0.3)
    ax3.set_xticks(step_counts)
    plt.tight_layout()
    plt.savefig('ns_lip_comparison_mean.png', dpi=200, bbox_inches='tight')
    print(f"Figure saved: ns_lip_comparison_mean.png")
