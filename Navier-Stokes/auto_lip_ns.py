"""
Auto-tuned (sigma, r) for Lip transfer on NS.

Algorithm:
  sigma = c * sqrt(max_k(S_data(k) / S_noise(k)))
  r     = S_data(k_Nyquist) / (sigma^2 * S_noise(k_Nyquist))

Tests on existing checkpoints:
  - noise=1 checkpoint via noise-scaled affine transfer
  - noise=10 checkpoint directly (already has large noise)
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


# ─── Data / spectrum ─────────────────────────────────────────────────

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


def get_enstrophy_spectrum(data):
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


# ─── Model / samplers ────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1, dim=32,
                        dim_mults=(1,2,2,2), resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4, use_classes=False)
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
def rk4_lip_noisescaled(model, z0_unit, steps, ratio, sigma, t_min=1e-3, t_max=1-1e-3):
    """Combined noise-scaling + Lip transfer. z0_unit has std=1."""
    r = ratio; log_r = math.log(r); s = sigma
    def alpha(t): return torch.sqrt((r - r**t)/(r - 1)) * torch.ones_like(t)
    def alpha_dot(t): return -0.5/alpha(t) * (r**t) * log_r/(r - 1)
    def beta(t): return torch.sqrt((r**t - 1)/(r - 1)) * torch.ones_like(t)
    def beta_dot(t): return 0.5/beta(t) * (r**t) * log_r/(r - 1)
    def drift(zt, ta):
        a = alpha(ta); ad = alpha_dot(ta)
        b = beta(ta); bd = beta_dot(ta)
        denom = (s*a + b)
        orig_t = b / denom
        orig_x = zt / denom[:, None, None, None]
        orig_bt = model(orig_x, orig_t)
        coef_x = (s*ad + bd)[:, None, None, None]
        coef_v = (-s*ad*orig_t + bd*(1 - orig_t))[:, None, None, None]
        return coef_x * orig_x + coef_v * orig_bt
    zt = s * z0_unit
    tgrid = torch.linspace(t_min, t_max, steps).type_as(zt)
    ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = drift(zt, ta); k2 = drift(zt+.5*dt*k1, (t_i+.5*dt)*ones)
        k3 = drift(zt+.5*dt*k2, (t_i+.5*dt)*ones); k4 = drift(zt+dt*k3, (t_i+dt)*ones)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


@torch.no_grad()
def rk4_lip_direct(model, z0_scaled, steps, ratio, t_min=1e-3, t_max=1-1e-3):
    """Lip transfer without noise scaling (z0 already scaled at training time)."""
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
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0_scaled)
    zt = z0_scaled; ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = drift(zt, ta); k2 = drift(zt+.5*dt*k1, (t_i+.5*dt)*ones)
        k3 = drift(zt+.5*dt*k2, (t_i+.5*dt)*ones); k4 = drift(zt+dt*k3, (t_i+dt)*ones)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Auto-select (sigma, r) ──────────────────────────────────────────

def auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=1.5):
    eps = 1e-30
    ratio_per_k = S_data / (S_noise_unit + eps)
    sigma_min = float(np.sqrt(ratio_per_k.max()))
    sigma = margin * sigma_min
    r = float(S_data[-1] / (sigma**2 * S_noise_unit[-1] + eps))
    return sigma, r, dict(sigma_min=sigma_min, k_argmax=kvals[np.argmax(ratio_per_k)])


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--training_noise', type=float, default=1.0, help='Noise strength used during training')
    p.add_argument('--margins', type=float, nargs='+', default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--steps', type=int, nargs='+', default=[10, 20, 50])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    hi = args.hi_size; tn = args.training_noise

    print(f"{'='*70}")
    print(f"  NS Auto-Lip: {hi}x{hi}, training_noise={tn}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Margins: {args.margins}, Steps: {args.steps}")
    print(f"{'='*70}")

    _, test_data = load_ns_data('../NSdata/data_file.pt', hi, 0.9)
    num_eval = min(args.num_eval, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, S_data = get_enstrophy_spectrum(truth_sq)

    # Noise spectrum (unit variance)
    torch.manual_seed(123)
    noise_unit = torch.randn(2000, hi, hi)
    _, S_noise_unit = get_enstrophy_spectrum(noise_unit)

    print(f"\n  S_data range: [{S_data.min():.4e}, {S_data.max():.4e}]")
    print(f"  S_noise(unit) range: [{S_noise_unit.min():.4e}, {S_noise_unit.max():.4e}]")
    print(f"  S_data/S_noise max: {(S_data/S_noise_unit).max():.4e} at k={kvals[np.argmax(S_data/S_noise_unit)]:.1f}")

    # Auto (sigma, r) for each margin
    print(f"\n  {'margin':>8} {'sigma':>10} {'r':>15} {'sigma_min':>10}")
    for m in args.margins:
        s, r, diag = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=m)
        print(f"  {m:>8.2f} {s:>10.4f} {r:>15.6e} {diag['sigma_min']:>10.4f}")

    model = Velocity().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()

    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}

    # Build methods
    methods = []
    for ns in args.steps:
        methods.append(('Std (baseline)', ns, 'std_baseline', tn, None))
    for margin in args.margins:
        sigma, r, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=margin)
        for ns in args.steps:
            if tn == 1.0:
                # noise=1 checkpoint: use affine noise-scaled transfer
                methods.append((f'Lip c={margin} σ={sigma:.1f} r={r:.1e}', ns, 'lip_scaled', sigma, r))
            else:
                # noise=10 checkpoint: use direct Lip transfer with auto-r
                # but also scale sigma relative to training noise
                r_direct = S_data[-1] / (tn**2 * S_noise_unit[-1] + 1e-30)
                r_effective = r_direct / margin**2  # extra margin
                methods.append((f'Lip c={margin} r={r_effective:.1e}', ns, 'lip_direct', tn, r_effective))

    all_results = {(m[0], m[1]): [] for m in methods}

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx * 1000
        torch.manual_seed(seed)
        z0_unit = torch.randn(num_eval, 1, hi, hi)

        for label, nsteps, kind, sigma, r in methods:
            print(f"  seed={seed}, {label} RK4-{nsteps}...", flush=True)
            z0_dev = z0_unit.to(device)
            if kind == 'std_baseline':
                gen = rk4_standard(model, sigma * z0_dev, nsteps)
            elif kind == 'lip_scaled':
                gen = rk4_lip_noisescaled(model, z0_dev, nsteps, r, sigma)
            elif kind == 'lip_direct':
                gen = rk4_lip_direct(model, sigma * z0_dev, nsteps, r)
            gen_sq = gen.squeeze(1).cpu()
            _, enst_gen = get_enstrophy_spectrum(gen_sq)
            std_ratio = gen_sq.std().item() / truth_sq.std().item()

            band_err = {bn: np.mean(np.abs(S_data[m] - enst_gen[m]) / (np.abs(S_data[m]) + 1e-20))
                        for bn, m in bands.items()}
            mean_err = np.mean(np.abs(S_data - enst_gen) / (np.abs(S_data) + 1e-20))

            all_results[(label, nsteps)].append(dict(
                mean=mean_err, low=band_err['low'], mid=band_err['mid'],
                high=band_err['high'], std_ratio=std_ratio, enst_gen=enst_gen))

    # ─── Print ────────────────────────────────────────────────────────

    def agg(vals):
        a = np.array(vals); return a.mean(), a.std()

    print(f"\n{'='*70}")
    print(f"  RESULTS (mean ± std, {args.num_seeds} seeds)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<30} {'steps':>5} {'low':>12} {'mid':>12} {'high':>12} {'mean':>14} {'std_r':>8}")
    print(f"  {'-'*100}")
    for label, nsteps, _, _, _ in methods:
        key = (label, nsteps)
        ml, sl = agg([m['low'] for m in all_results[key]])
        mm, sm = agg([m['mid'] for m in all_results[key]])
        mh, sh = agg([m['high'] for m in all_results[key]])
        ma, sa = agg([m['mean'] for m in all_results[key]])
        mr, sr = agg([m['std_ratio'] for m in all_results[key]])
        print(f"  {label:<30} {nsteps:>5} {ml:.4f}±{sl:.4f} {mm:.4f}±{sm:.4f} {mh:.4f}±{sh:.4f} {ma:.4f}±{sa:.4f} {mr:>8.4f}")
