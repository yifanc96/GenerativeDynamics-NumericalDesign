"""
Reproduce the Lip paper (arXiv:2509.01629) result for Navier-Stokes:
Per-band enstrophy/energy spectrum accuracy for standard FM.

Uses the best checkpoint (Gauss-base 2M) and shows per-frequency-band
relative errors as a function of integration steps.
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
            data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        all_data.append(data)
    data = torch.cat(all_data, dim=0)[:, None, :, :]
    num_train = int(data.shape[0] * train_test_split)
    return data[:num_train], data[num_train:]


def get_enstrophy_energy_perband(data, nbands=None):
    """Return kvals, enstrophy, energy spectra."""
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


@torch.no_grad()
def rk4_sample(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
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


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_eval', type=int, default=500)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    # Load data
    print("Loading NS data...")
    train_data, test_data = load_ns_data('../NSdata/data_file.pt', 128, 0.9)
    print(f"  train={train_data.shape[0]}, test={test_data.shape[0]}")

    # Load model — user specified this is the good-spectrum checkpoint
    ckpt = 'results/ns_gauss_base_5data/model_final.pt'
    print(f"Loading model: {ckpt}")
    model = Velocity(C=1, unet_channels=32, unet_dim_mults=(1,2,2,2)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()

    num_eval = min(args.num_eval, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, enst_truth, ener_truth = get_enstrophy_energy_perband(truth_sq)

    z0 = torch.randn(num_eval, 1, 128, 128, device=device)
    torch.manual_seed(42)

    step_counts = [5, 10, 20, 50, 100]
    all_enst = {}
    all_ener = {}

    for nsteps in step_counts:
        print(f"  Sampling with RK4, {nsteps} steps...")
        gen = rk4_sample(model, z0.clone(), nsteps)
        gen_sq = gen.squeeze(1).cpu()
        _, enst_gen, ener_gen = get_enstrophy_energy_perband(gen_sq)
        all_enst[nsteps] = enst_gen
        all_ener[nsteps] = ener_gen
        std_ratio = gen_sq.std().item() / truth_sq.std().item()
        print(f"    std_ratio={std_ratio:.4f}")

    # ─── Per-band relative error ──────────────────────────────────────

    print("\n" + "="*80)
    print("PER-BAND ENSTROPHY RELATIVE ERROR  |S_gen(k) - S_truth(k)| / S_truth(k)")
    print("="*80)
    print(f"{'k':>6}", end='')
    for ns in step_counts:
        print(f"  {'RK4-'+str(ns):>10}", end='')
    print()
    print("-" * (6 + 12 * len(step_counts)))

    for i, k in enumerate(kvals):
        print(f"{k:6.1f}", end='')
        for ns in step_counts:
            err = abs(all_enst[ns][i] - enst_truth[i]) / (abs(enst_truth[i]) + 1e-20)
            print(f"  {err:10.4f}", end='')
        print()

    # ─── Band-averaged errors ─────────────────────────────────────────

    band_edges = [(1, 4, 'very-low'), (4, 8, 'low'), (8, 16, 'mid-low'),
                  (16, 24, 'mid'), (24, 40, 'mid-high'), (40, 64, 'high')]

    print("\n" + "="*80)
    print("BAND-AVERAGED ENSTROPHY RELATIVE ERROR")
    print("="*80)
    print(f"{'band':>12}", end='')
    for ns in step_counts:
        print(f"  {'RK4-'+str(ns):>10}", end='')
    print()
    print("-" * (12 + 12 * len(step_counts)))

    band_errors = {ns: {} for ns in step_counts}
    for k_lo, k_hi, bname in band_edges:
        mask = (kvals >= k_lo) & (kvals < k_hi)
        if mask.sum() == 0:
            continue
        print(f"{bname+f' [{k_lo},{k_hi})':>12}", end='')
        for ns in step_counts:
            err = np.mean(np.abs(all_enst[ns][mask] - enst_truth[mask]) / (np.abs(enst_truth[mask]) + 1e-20))
            band_errors[ns][bname] = err
            print(f"  {err:10.4f}", end='')
        print()

    # ─── Figure 1: Spectra at different step counts ───────────────────

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Enstrophy
    ax = axes[0]
    ax.loglog(kvals, enst_truth, 'k-', lw=2.5, label='Truth', zorder=10)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(step_counts)))
    for i, ns in enumerate(step_counts):
        ax.loglog(kvals, all_enst[ns], '--', color=colors[i], lw=1.5,
                 label=f'RK4 {ns} steps')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Enstrophy spectrum', fontsize=12)
    ax.set_title('NS Enstrophy Spectrum', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Energy
    ax = axes[1]
    ax.loglog(kvals, ener_truth, 'k-', lw=2.5, label='Truth', zorder=10)
    for i, ns in enumerate(step_counts):
        ax.loglog(kvals, all_ener[ns], '--', color=colors[i], lw=1.5,
                 label=f'RK4 {ns} steps')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Energy spectrum', fontsize=12)
    ax.set_title('NS Energy Spectrum', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Per-band relative error
    ax = axes[2]
    for i, ns in enumerate(step_counts):
        rel_err = np.abs(all_enst[ns] - enst_truth) / (np.abs(enst_truth) + 1e-20)
        ax.semilogy(kvals, rel_err, '-', color=colors[i], lw=1.5, alpha=0.8,
                   label=f'RK4 {ns} steps')
    ax.axhline(y=0.1, color='gray', ls=':', alpha=0.6)
    ax.axhline(y=0.01, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Relative enstrophy error', fontsize=12)
    ax.set_title('Per-k Relative Error', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-3, 10)

    plt.tight_layout()
    plt.savefig('ns_lip_standard_schedule.png', dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: ns_lip_standard_schedule.png")

    # ─── Figure 2: Band-averaged error vs steps ──────────────────────

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5.5))
    band_colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(band_edges)))
    for j, (k_lo, k_hi, bname) in enumerate(band_edges):
        errs = [band_errors[ns].get(bname, np.nan) for ns in step_counts]
        ax2.semilogy(step_counts, errs, '-o', color=band_colors[j], lw=2, markersize=6,
                    label=f'k∈[{k_lo},{k_hi})')
    ax2.set_xlabel('Number of RK4 steps', fontsize=12)
    ax2.set_ylabel('Band-averaged relative enstrophy error', fontsize=12)
    ax2.set_title('NS Standard FM: Error vs Integration Steps per Band', fontsize=13)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(step_counts)

    plt.tight_layout()
    plt.savefig('ns_lip_band_vs_steps.png', dpi=200, bbox_inches='tight')
    print(f"Figure saved: ns_lip_band_vs_steps.png")
