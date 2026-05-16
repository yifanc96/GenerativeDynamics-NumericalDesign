"""
Compare standard vs Lip schedule using the TRANSFER FORMULA on existing checkpoints.

Transfer formula (from Lip paper): given NN trained with standard schedule,
the Lip schedule drift is:
  b(z,t) = (α̇/α)·z + (β̇ - α̇β/α)·E[z₁|z_t]
where E[z₁|z_t] is obtained by mapping to equivalent standard time:
  orig_t = 1/(1 + α(t)/β(t))
  orig_x = orig_t/β(t) · z_t
  E[z₁|z_t] = (1 - orig_t)·v_nn(orig_x, orig_t) + orig_x
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


# ─── Standard schedule sampler ────────────────────────────────────────

@torch.no_grad()
def rk4_standard(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
    """N grid points → N-1 RK4 steps."""
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0
    ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]
        dt = tgrid[i + 1] - tgrid[i]
        t_arr = t_i * ones
        k1 = model(zt, t_arr)
        k2 = model(zt + 0.5*dt*k1, t_arr + 0.5*dt)
        k3 = model(zt + 0.5*dt*k2, t_arr + 0.5*dt)
        k4 = model(zt + dt*k3, t_arr + dt)
        zt = zt + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Lip schedule sampler (transfer formula) ─────────────────────────

@torch.no_grad()
def rk4_lip(model, z0, steps, ratio, t_min=1e-3, t_max=1-1e-3):
    """
    Lip schedule using transfer formula with trained NN.
    Matches the notebook: NSunconditional_training-and-inference-white-noise.ipynb
    """
    r = ratio
    log_r = math.log(r)

    # Matching notebook exactly
    def alpha(t):
        return torch.sqrt((r - r**t) / (r - 1)) * torch.ones_like(t)
    def alpha_dot(t):
        return -0.5 / alpha(t) * (r**t) * log_r / (r - 1)
    def beta(t):
        return torch.sqrt((r**t - 1) / (r - 1)) * torch.ones_like(t)
    def beta_dot(t):
        return 0.5 / beta(t) * (r**t) * log_r / (r - 1)

    def lip_drift(zt, t_arr):
        # Matching notebook Cell 10 exactly
        bt = (alpha_dot(t_arr) / alpha(t_arr))[:, None, None, None] * zt
        coef = beta_dot(t_arr) - alpha_dot(t_arr) * beta(t_arr) / alpha(t_arr)
        coef = coef[:, None, None, None]
        orig_t = 1 / (1 + alpha(t_arr) / beta(t_arr))
        orig_x = orig_t[:, None, None, None] / (beta(t_arr)[:, None, None, None]) * zt
        orig_bt = model(orig_x, orig_t)
        bt += coef * ((1 - orig_t[:, None, None, None]) * orig_bt + orig_x)
        return bt

    # EM integration (matching notebook style, simpler than RK4 for stability)
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0
    ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]
        dt = tgrid[i + 1] - tgrid[i]
        t_arr = t_i * ones
        k1 = lip_drift(zt, t_arr)
        k2 = lip_drift(zt + 0.5*dt*k1, (t_i + 0.5*dt) * ones)
        k3 = lip_drift(zt + 0.5*dt*k2, (t_i + 0.5*dt) * ones)
        k4 = lip_drift(zt + dt*k3, (t_i + dt) * ones)
        zt = zt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lip_rs', type=float, nargs='+', default=[1e-5])
    p.add_argument('--num_eval', type=int, default=200)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--noise_strength', type=float, default=10.0)
    p.add_argument('--hi_size', type=int, default=128)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    hi = args.hi_size
    # Load data
    print(f"Loading NS data at {hi}x{hi}...")
    train_data, test_data = load_ns_data('../NSdata/data_file.pt', hi, 0.9)
    print(f"  train={train_data.shape[0]}, test={test_data.shape[0]}")

    num_eval = min(args.num_eval, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)

    ns = args.noise_strength
    print(f"  noise_strength={ns}, lip_rs={args.lip_rs}, hi_size={hi}")

    # Noise spectrum
    torch.manual_seed(123)
    noise_sq = ns * torch.randn(num_eval, hi, hi)
    _, enst_noise, _ = get_energy_spectrum(noise_sq)

    # Fixed z0
    torch.manual_seed(42)
    z0 = ns * torch.randn(num_eval, 1, hi, hi, device=device)

    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}
    step_counts = [5, 10, 20, 50]

    # Checkpoints to evaluate
    if args.ckpt:
        configs = [("Custom", args.ckpt, 32, (1,2,2,2))]
    else:
        configs = [("Gauss-base 2M", "results/ns_gauss_base/model_final.pt", 32, (1,2,2,2))]

    for name, ckpt, channels, dim_mults in configs:
        if not os.path.exists(ckpt):
            print(f"SKIP {name}: {ckpt} not found")
            continue

        print(f"\n{'='*80}")
        print(f"  {name}: {ckpt}")
        print(f"  Lip ratios r = {args.lip_rs}")
        print(f"{'='*80}")

        model = Velocity(C=1, unet_channels=channels, unet_dim_mults=dim_mults).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()

        results = {}  # (method, steps) -> (enst, std_ratio)

        methods = [('Standard', lambda z, n: rk4_standard(model, z, n))]
        for lr in args.lip_rs:
            methods.append((f'Lip r={lr:.0e}', lambda z, n, _r=lr: rk4_lip(model, z, n, _r)))

        for method_name, sample_fn in methods:
            print(f"\n  {method_name} schedule:")
            print(f"  {'steps':>6}  {'std_ratio':>10}  {'low':>8}  {'mid':>8}  {'high':>8}  {'mean':>8}")
            print(f"  {'-'*56}")
            for nsteps in step_counts:
                gen = sample_fn(z0.clone(), nsteps)
                gen_sq = gen.squeeze(1).cpu()
                _, enst_gen, _ = get_energy_spectrum(gen_sq)
                std_ratio = gen_sq.std().item() / truth_sq.std().item()
                results[(method_name, nsteps)] = (enst_gen, std_ratio)

                band_err = {bn: np.mean(np.abs(enst_truth[m] - enst_gen[m]) / (np.abs(enst_truth[m]) + 1e-20))
                            for bn, m in bands.items()}
                mean_err = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-20))
                print(f"  {nsteps:>6}  {std_ratio:>10.4f}  {band_err['low']:>8.4f}  {band_err['mid']:>8.4f}  {band_err['high']:>8.4f}  {mean_err:>8.4f}")

        # ── Plot ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        ref = 20
        lip_colors = ['blue', 'green', 'purple', 'orange']

        # Spectrum at ref steps
        ax = axes[0]
        ax.loglog(kvals, enst_noise, 'gray', lw=1.5, alpha=0.6, label='Noise (z0)')
        ax.loglog(kvals, enst_truth, 'k-', lw=2.5, label='Target (data)')
        if ('Standard', ref) in results:
            ax.loglog(kvals, results[('Standard', ref)][0], 'r--', lw=1.5, label=f'Standard ({ref} steps)')
        for i, lr in enumerate(args.lip_rs):
            key = f'Lip r={lr:.0e}'
            if (key, ref) in results:
                ax.loglog(kvals, results[(key, ref)][0], '-', color=lip_colors[i % len(lip_colors)],
                         lw=1.5, label=f'{key} ({ref} steps)')
        ax.set_xlabel('k'); ax.set_ylabel('Enstrophy')
        ax.set_title(f'Enstrophy ({name}, {hi}x{hi})')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Per-k error at ref steps
        ax = axes[1]
        if ('Standard', ref) in results:
            rel_err = np.abs(results[('Standard', ref)][0] - enst_truth) / (np.abs(enst_truth) + 1e-20)
            ax.semilogy(kvals, rel_err, 'r--', lw=1.5, label='Standard')
        for i, lr in enumerate(args.lip_rs):
            key = f'Lip r={lr:.0e}'
            if (key, ref) in results:
                rel_err = np.abs(results[(key, ref)][0] - enst_truth) / (np.abs(enst_truth) + 1e-20)
                ax.semilogy(kvals, rel_err, '-', color=lip_colors[i % len(lip_colors)], lw=1.5, label=key)
        ax.axhline(y=0.1, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('k'); ax.set_ylabel('Relative error')
        ax.set_title(f'Per-k Error ({ref} steps)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(1e-3, 10)

        # Mean error vs steps
        ax = axes[2]
        if all(('Standard', n) in results for n in step_counts):
            errs = [np.mean(np.abs(enst_truth - results[('Standard', n)][0]) / (np.abs(enst_truth) + 1e-20))
                    for n in step_counts]
            ax.semilogy(step_counts, errs, 'r--s', lw=2, markersize=6, label='Standard')
        for i, lr in enumerate(args.lip_rs):
            key = f'Lip r={lr:.0e}'
            if all((key, n) in results for n in step_counts):
                errs = [np.mean(np.abs(enst_truth - results[(key, n)][0]) / (np.abs(enst_truth) + 1e-20))
                        for n in step_counts]
                ax.semilogy(step_counts, errs, '-o', color=lip_colors[i % len(lip_colors)],
                           lw=2, markersize=6, label=key)
        ax.set_xlabel('RK4 steps'); ax.set_ylabel('Mean rel. enstrophy error')
        ax.set_title('Error vs Steps')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(step_counts)

        plt.tight_layout()
        out_name = f'ns_lip_transfer_{hi}x{hi}_noise{int(ns)}.png'
        plt.savefig(out_name, dpi=200, bbox_inches='tight')
        print(f"\nFigure saved: {out_name}")
