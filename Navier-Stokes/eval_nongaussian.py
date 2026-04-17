"""
Non-Gaussian evaluation metrics for NS flow matching:
1. Pointwise vorticity PDF
2. Per-sample skewness & kurtosis distributions
3. Structure functions S_p(r) for p=2,3,4 and flatness S_4/S_2^2
4. Per-sample enstrophy distribution
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


# ─── Data / model / samplers (same as eval_lip_transfer.py) ──────────

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
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    enstrophy, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    enstrophy *= area_weight
    return kvals, enstrophy


class Velocity(nn.Module):
    def __init__(self, C, unet_channels, unet_dim_mults):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=C, out_channels=C,
            dim=unet_channels, dim_mults=unet_dim_mults,
            resnet_block_groups=8, learned_sinusoidal_cond=True,
            random_fourier_features=False, learned_sinusoidal_dim=32,
            attn_dim_head=32, attn_heads=4, use_classes=False,
        )
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


@torch.no_grad()
def rk4_standard(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0; ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = model(zt, ta); k2 = model(zt+.5*dt*k1, ta+.5*dt)
        k3 = model(zt+.5*dt*k2, ta+.5*dt); k4 = model(zt+dt*k3, ta+dt)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


@torch.no_grad()
def rk4_lip(model, z0, steps, ratio, t_min=1e-3, t_max=1-1e-3):
    r = ratio; log_r = math.log(r)
    def alpha(t): return torch.sqrt((r - r**t)/(r - 1)) * torch.ones_like(t)
    def alpha_dot(t): return -0.5/alpha(t) * (r**t) * log_r/(r - 1)
    def beta(t): return torch.sqrt((r**t - 1)/(r - 1)) * torch.ones_like(t)
    def beta_dot(t): return 0.5/beta(t) * (r**t) * log_r/(r - 1)
    def drift(zt, ta):
        bt = (alpha_dot(ta)/alpha(ta))[:, None, None, None] * zt
        coef = (beta_dot(ta) - alpha_dot(ta)*beta(ta)/alpha(ta))[:, None, None, None]
        orig_t = 1/(1 + alpha(ta)/beta(ta))
        orig_x = orig_t[:, None, None, None]/(beta(ta)[:, None, None, None]) * zt
        orig_bt = model(orig_x, orig_t)
        bt += coef * ((1 - orig_t[:, None, None, None]) * orig_bt + orig_x)
        return bt
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0; ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = drift(zt, ta); k2 = drift(zt+.5*dt*k1, (t_i+.5*dt)*ones)
        k3 = drift(zt+.5*dt*k2, (t_i+.5*dt)*ones); k4 = drift(zt+dt*k3, (t_i+dt)*ones)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Non-Gaussian metrics ────────────────────────────────────────────

def compute_pointwise_stats(samples):
    """Per-sample mean, std, skewness, kurtosis of pixel values."""
    # samples: (N, H, W)
    N = samples.shape[0]
    flat = samples.reshape(N, -1)
    mu = flat.mean(dim=1)
    std = flat.std(dim=1)
    centered = flat - mu[:, None]
    skew = (centered**3).mean(dim=1) / (std**3 + 1e-20)
    kurt = (centered**4).mean(dim=1) / (std**4 + 1e-20)
    return mu.numpy(), std.numpy(), skew.numpy(), kurt.numpy()


def compute_structure_functions(samples, max_r=None, num_r=20):
    """
    Structure functions S_p(r) = <|omega(x+r) - omega(x)|^p> for p=2,3,4.
    Averaged over spatial directions and samples.
    samples: (N, H, W) numpy or torch
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    N, H, W = samples.shape
    if max_r is None:
        max_r = H // 4
    rs = np.unique(np.linspace(1, max_r, num_r).astype(int))

    S2 = np.zeros(len(rs))
    S3 = np.zeros(len(rs))
    S4 = np.zeros(len(rs))

    for idx, r in enumerate(rs):
        # Average over x and y shifts
        diffs = []
        if r < H:
            diffs.append(samples[:, r:, :] - samples[:, :-r, :])  # y-shift
        if r < W:
            diffs.append(samples[:, :, r:] - samples[:, :, :-r])  # x-shift
        diff = np.concatenate([d.reshape(-1) for d in diffs])
        S2[idx] = np.mean(np.abs(diff)**2)
        S3[idx] = np.mean(np.abs(diff)**3)
        S4[idx] = np.mean(np.abs(diff)**4)

    return rs, S2, S3, S4


def compute_per_sample_enstrophy(samples):
    """Total enstrophy per sample: integral |omega|^2 dx."""
    # samples: (N, H, W)
    return (samples**2).sum(dim=(1, 2)).numpy()


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--lip_r', type=float, default=1e-5)
    p.add_argument('--noise_strength', type=float, default=10.0)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--tag', type=str, default='')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    hi = args.hi_size
    ns = args.noise_strength

    print(f"{'='*70}")
    print(f"  Non-Gaussian eval: {hi}x{hi}, noise={ns}, lip_r={args.lip_r:.0e}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"{'='*70}")

    # Load data
    _, test_data = load_ns_data('../NSdata/data_file.pt', hi, 0.9)
    num_eval = min(args.num_eval, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)  # (N, H, W)
    print(f"  Test samples: {num_eval}, resolution: {hi}x{hi}")

    # Load model
    model = Velocity(C=1, unet_channels=32, unet_dim_mults=(1, 2, 2, 2)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()

    # Generate
    torch.manual_seed(42)
    z0 = ns * torch.randn(num_eval, 1, hi, hi, device=device)

    step_counts = [10, 20]
    generated = {}  # (method, steps) -> (N, H, W) tensor

    for nsteps in step_counts:
        print(f"  Generating: Standard RK4-{nsteps}...")
        generated[('Standard', nsteps)] = rk4_standard(model, z0.clone(), nsteps).squeeze(1).cpu()
        print(f"  Generating: Lip r={args.lip_r:.0e} RK4-{nsteps}...")
        generated[('Lip', nsteps)] = rk4_lip(model, z0.clone(), nsteps, args.lip_r).squeeze(1).cpu()

    # ─── Compute metrics ──────────────────────────────────────────────

    # Truth metrics
    truth_mu, truth_std, truth_skew, truth_kurt = compute_pointwise_stats(truth_sq)
    truth_rs, truth_S2, truth_S3, truth_S4 = compute_structure_functions(truth_sq)
    truth_enst = compute_per_sample_enstrophy(truth_sq)
    kvals, truth_spec = get_energy_spectrum(truth_sq)

    methods_steps = [('Standard', 10), ('Standard', 20), ('Lip', 10), ('Lip', 20)]
    all_metrics = {}

    for key in methods_steps:
        gen = generated[key]
        mu, std, skew, kurt = compute_pointwise_stats(gen)
        rs, S2, S3, S4 = compute_structure_functions(gen)
        enst = compute_per_sample_enstrophy(gen)
        _, spec = get_energy_spectrum(gen)
        all_metrics[key] = dict(mu=mu, std=std, skew=skew, kurt=kurt,
                                rs=rs, S2=S2, S3=S3, S4=S4, enst=enst, spec=spec)

    # ─── Print summary ────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {hi}x{hi}, noise={ns}")
    print(f"{'='*70}")
    print(f"  Truth: std={truth_std.mean():.4f}, skew={truth_skew.mean():.4f}, "
          f"kurt={truth_kurt.mean():.4f}, enstrophy={truth_enst.mean():.2f}")

    print(f"\n  {'Method':<16} {'steps':>5} {'std':>8} {'skew':>8} {'kurt':>8} "
          f"{'enst_mean':>10} {'enst_std':>10} {'spec_err':>10}")
    print(f"  {'-'*78}")
    print(f"  {'Truth':<16} {'':>5} {truth_std.mean():>8.4f} {truth_skew.mean():>8.4f} "
          f"{truth_kurt.mean():>8.4f} {truth_enst.mean():>10.2f} {truth_enst.std():>10.2f} {'':>10}")

    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} r={args.lip_r:.0e}" if key[0] == 'Lip' else key[0]
        spec_err = np.mean(np.abs(m['spec'] - truth_spec) / (np.abs(truth_spec) + 1e-20))
        print(f"  {label:<16} {key[1]:>5} {m['std'].mean():>8.4f} {m['skew'].mean():>8.4f} "
              f"{m['kurt'].mean():>8.4f} {m['enst'].mean():>10.2f} {m['enst'].std():>10.2f} "
              f"{spec_err:>10.4f}")

    # ─── Plots ────────────────────────────────────────────────────────

    colors = {'Standard': 'red', 'Lip': 'blue'}
    lstyles = {10: '--', 20: '-'}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    tag = f"{hi}x{hi}_noise{int(ns)}"

    # (0,0) Enstrophy spectrum
    ax = axes[0, 0]
    ax.loglog(kvals, truth_spec, 'k-', lw=2.5, label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} {key[1]}s"
        ax.loglog(kvals, m['spec'], ls=lstyles[key[1]], color=colors[key[0]], lw=1.5, label=label)
    ax.set_xlabel('k'); ax.set_ylabel('Enstrophy spectrum')
    ax.set_title('Enstrophy Spectrum'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) Pointwise vorticity PDF
    ax = axes[0, 1]
    bins = np.linspace(-4, 4, 100)
    ax.hist(truth_sq.numpy().flatten(), bins=bins, density=True, alpha=0.5, color='black', label='Truth')
    for key in methods_steps:
        gen = generated[key]
        label = f"{key[0]} {key[1]}s"
        ax.hist(gen.numpy().flatten(), bins=bins, density=True, alpha=0.3,
                color=colors[key[0]], linestyle=lstyles[key[1]], histtype='step', lw=1.5, label=label)
    ax.set_xlabel('Vorticity'); ax.set_ylabel('Density')
    ax.set_title('Pointwise Vorticity PDF'); ax.legend(fontsize=8); ax.set_yscale('log')
    ax.set_ylim(1e-4, 10)

    # (0,2) Per-sample kurtosis distribution
    ax = axes[0, 2]
    ax.hist(truth_kurt, bins=30, density=True, alpha=0.5, color='black', label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} {key[1]}s"
        ax.hist(m['kurt'], bins=30, density=True, alpha=0.3, color=colors[key[0]],
                histtype='step', lw=1.5, label=label)
    ax.set_xlabel('Kurtosis'); ax.set_ylabel('Density')
    ax.set_title('Per-sample Kurtosis'); ax.legend(fontsize=8)

    # (0,3) Per-sample enstrophy distribution
    ax = axes[0, 3]
    ax.hist(truth_enst, bins=30, density=True, alpha=0.5, color='black', label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} {key[1]}s"
        ax.hist(m['enst'], bins=30, density=True, alpha=0.3, color=colors[key[0]],
                histtype='step', lw=1.5, label=label)
    ax.set_xlabel('Enstrophy'); ax.set_ylabel('Density')
    ax.set_title('Per-sample Enstrophy'); ax.legend(fontsize=8)

    # (1,0) Structure function S2(r)
    ax = axes[1, 0]
    ax.loglog(truth_rs, truth_S2, 'k-', lw=2.5, label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} {key[1]}s"
        ax.loglog(m['rs'], m['S2'], ls=lstyles[key[1]], color=colors[key[0]], lw=1.5, label=label)
    ax.set_xlabel('r (pixels)'); ax.set_ylabel('S₂(r)')
    ax.set_title('Structure Function S₂(r)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,1) Structure function S4(r)
    ax = axes[1, 1]
    ax.loglog(truth_rs, truth_S4, 'k-', lw=2.5, label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} {key[1]}s"
        ax.loglog(m['rs'], m['S4'], ls=lstyles[key[1]], color=colors[key[0]], lw=1.5, label=label)
    ax.set_xlabel('r (pixels)'); ax.set_ylabel('S₄(r)')
    ax.set_title('Structure Function S₄(r)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,2) Flatness S4/S2^2
    ax = axes[1, 2]
    truth_flat = truth_S4 / (truth_S2**2 + 1e-20)
    ax.semilogx(truth_rs, truth_flat, 'k-', lw=2.5, label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        flatness = m['S4'] / (m['S2']**2 + 1e-20)
        label = f"{key[0]} {key[1]}s"
        ax.semilogx(m['rs'], flatness, ls=lstyles[key[1]], color=colors[key[0]], lw=1.5, label=label)
    ax.axhline(y=3.0, color='gray', ls=':', alpha=0.5, label='Gaussian (=3)')
    ax.set_xlabel('r (pixels)'); ax.set_ylabel('S₄/S₂²')
    ax.set_title('Flatness (Intermittency)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,3) Per-sample skewness distribution
    ax = axes[1, 3]
    ax.hist(truth_skew, bins=30, density=True, alpha=0.5, color='black', label='Truth')
    for key in methods_steps:
        m = all_metrics[key]
        label = f"{key[0]} {key[1]}s"
        ax.hist(m['skew'], bins=30, density=True, alpha=0.3, color=colors[key[0]],
                histtype='step', lw=1.5, label=label)
    ax.set_xlabel('Skewness'); ax.set_ylabel('Density')
    ax.set_title('Per-sample Skewness'); ax.legend(fontsize=8)

    plt.suptitle(f'Non-Gaussian Metrics: {hi}x{hi}, noise={int(ns)}, Lip r={args.lip_r:.0e}', fontsize=14)
    plt.tight_layout()
    out_name = f'ns_nongaussian_{tag}_lipR{args.lip_r:.0e}.png'
    plt.savefig(out_name, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {out_name}")
