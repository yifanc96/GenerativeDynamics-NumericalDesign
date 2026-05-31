"""
Evaluate CelebA-HQ: Standard vs Lip schedule.
Metrics: per-band spectrum, FID, gradient kurtosis, pixel stats.
"""
import os, sys, math, tempfile, shutil
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from unet import Unet


# ─── Data ─────────────────────────────────────────────────────────────

def load_celeba_data(hi_size=128):
    cache = f'celeba_data/celeba_hq_{hi_size}.pt'
    if os.path.exists(cache):
        data = torch.load(cache, weights_only=False)
        print(f"  Loaded cached data: {data.shape}")
    else:
        raise FileNotFoundError(f"Cache not found: {cache}")
    # data is (N, 3, H, W) in [0, 1] — normalize to [-1, 1]
    if data.min() >= 0:
        data = data * 2 - 1
        print(f"  Normalized to [-1, 1]: range [{data.min():.2f}, {data.max():.2f}]")
    num_train = int(data.shape[0] * 0.95)
    return data[:num_train], data[num_train:]


# ─── Spectrum ─────────────────────────────────────────────────────────

def get_spectrum_per_channel(data):
    """Radially averaged power spectrum, averaged over channels and samples."""
    # data: (N, C, H, W)
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(2, 3), norm='forward')
    power = (torch.abs(fhat)**2).mean(dim=(0, 1))  # avg over samples and channels
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    power_flat = power.numpy().flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    spec, _, _ = stats.binned_statistic(knrm, power_flat, statistic='mean', bins=kbins)
    spec *= area_weight
    return kvals, spec


# ─── Model ────────────────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self, C=3, dim=64, dim_mults=(1, 2, 4, 4)):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=C, out_channels=C,
            dim=dim, dim_mults=dim_mults, resnet_block_groups=8,
            learned_sinusoidal_cond=True, random_fourier_features=False,
            learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4,
            use_classes=False,
        )
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


# ─── Samplers ─────────────────────────────────────────────────────────

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


@torch.no_grad()
def rk4_lip_noisescaled(model, z0_unit, steps, ratio, sigma, t_min=1e-3, t_max=1-1e-3):
    """
    Lip schedule + noise scaling, both via affine transfer of the noise=1 checkpoint.

    Effective interpolant: z_t = sigma*alpha_lip(t)*z0 + beta_lip(t)*z1
    Transfer:
      orig_t = beta_lip / (sigma*alpha_lip + beta_lip)
      orig_x = z_t / (sigma*alpha_lip + beta_lip)
      b(z,t) = orig_x*(sigma*alpha_dot + beta_dot)
             + v_nn(orig_x, orig_t)*(-sigma*alpha_dot*orig_t + beta_dot*(1-orig_t))

    z0_unit: noise with std=1 (will be scaled by sigma internally)
    """
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
        # b(z,t) = orig_x * (s*ad + bd) + v_nn * (-s*ad*orig_t + bd*(1-orig_t))
        coef_x = (s*ad + bd)[:, None, None, None]
        coef_v = (-s*ad*orig_t + bd*(1 - orig_t))[:, None, None, None]
        return coef_x * orig_x + coef_v * orig_bt

    # Initial condition is sigma * z0_unit (start at z_{t_min} ~= sigma*z0)
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
def rk4_standard_noisescaled(model, z0_unit, steps, sigma, t_min=1e-3, t_max=1-1e-3):
    """
    Standard schedule with noise scaling: z_t = sigma*(1-t)*z0 + t*z1 (alpha=1-t)
    Special case of rk4_lip_noisescaled with alpha=1-t, beta=t (no Lip).
    Equivalent transfer: orig_t = t/(sigma*(1-t) + t), orig_x = z/(sigma*(1-t)+t)
    """
    s = sigma
    def drift(zt, ta):
        # alpha=1-t, alpha_dot=-1, beta=t, beta_dot=1
        denom = s*(1 - ta) + ta
        orig_t = ta / denom
        orig_x = zt / denom[:, None, None, None]
        orig_bt = model(orig_x, orig_t)
        coef_x = (-s + 1)  # s*(-1) + 1
        coef_v = (s*orig_t + (1 - orig_t))  # -s*(-1)*orig_t + 1*(1-orig_t)
        return coef_x * orig_x + coef_v[:, None, None, None] * orig_bt

    zt = s * z0_unit
    tgrid = torch.linspace(t_min, t_max, steps).type_as(zt)
    ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = drift(zt, ta); k2 = drift(zt+.5*dt*k1, (t_i+.5*dt)*ones)
        k3 = drift(zt+.5*dt*k2, (t_i+.5*dt)*ones); k4 = drift(zt+dt*k3, (t_i+dt)*ones)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Image metrics ───────────────────────────────────────────────────

def compute_image_metrics(truth, generated):
    """
    Compute image-based metrics.
    truth, generated: (N, 3, H, W) in [-1, 1]
    """
    metrics = {}

    # 1. Per-channel mean and std
    for c, name in enumerate(['R', 'G', 'B']):
        metrics[f'mean_{name}'] = generated[:, c].mean().item()
        metrics[f'std_{name}'] = generated[:, c].std().item()

    # 2. Per-sample pixel statistics
    flat = generated.reshape(generated.shape[0], -1)
    sample_mean = flat.mean(dim=1)
    sample_std = flat.std(dim=1)
    sample_kurt = ((flat - sample_mean[:, None])**4).mean(dim=1) / (sample_std**4 + 1e-20)
    metrics['pixel_mean'] = sample_mean.mean().item()
    metrics['pixel_std'] = sample_std.mean().item()
    metrics['pixel_kurt'] = sample_kurt.mean().item()

    # 3. Spectrum relative error
    kvals, spec_truth = get_spectrum_per_channel(truth)
    _, spec_gen = get_spectrum_per_channel(generated)
    metrics['spec_err_mean'] = np.mean(np.abs(spec_truth - spec_gen) / (np.abs(spec_truth) + 1e-20))

    # Per-band errors
    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}
    for bname, mask in bands.items():
        if mask.sum() > 0:
            metrics[f'spec_err_{bname}'] = np.mean(np.abs(spec_truth[mask] - spec_gen[mask]) / (np.abs(spec_truth[mask]) + 1e-20))

    # 4. SSIM-like: per-sample correlation with nearest truth sample
    # (simplified: just variance of generated vs truth)
    metrics['total_var_ratio'] = generated.var().item() / (truth.var().item() + 1e-20)

    # 5. Gradient magnitude distribution (texture sharpness)
    dx = generated[:, :, :, 1:] - generated[:, :, :, :-1]
    dy = generated[:, :, 1:, :] - generated[:, :, :-1, :]
    grad_mag = torch.cat([dx.reshape(-1), dy.reshape(-1)])
    metrics['grad_std'] = grad_mag.std().item()
    metrics['grad_kurt'] = (((grad_mag - grad_mag.mean())**4).mean() / (grad_mag.std()**4 + 1e-20)).item()

    return metrics, kvals, spec_truth, spec_gen


# ─── FID computation ──────────────────────────────────────────────────

def save_images_to_dir(tensor, dirpath):
    """Save (N, 3, H, W) tensor in [-1,1] as PNG images."""
    os.makedirs(dirpath, exist_ok=True)
    imgs = ((tensor.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    for i, img in enumerate(imgs):
        Image.fromarray(img).save(os.path.join(dirpath, f'{i:05d}.png'))


def compute_fid(truth_tensor, gen_tensor, device_str='cuda'):
    """Compute FID using pytorch-fid (plain PyTorch InceptionV3)."""
    from pytorch_fid.fid_score import calculate_fid_given_paths
    truth_dir = tempfile.mkdtemp(prefix='fid_truth_')
    gen_dir = tempfile.mkdtemp(prefix='fid_gen_')
    try:
        save_images_to_dir(truth_tensor, truth_dir)
        save_images_to_dir(gen_tensor, gen_dir)
        score = calculate_fid_given_paths(
            [truth_dir, gen_dir], batch_size=50, device=device_str,
            dims=2048, num_workers=0,
        )
        return score
    finally:
        shutil.rmtree(truth_dir)
        shutil.rmtree(gen_dir)


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ckpt', type=str, default='results/celeba_gauss_200k/model_final.pt')
    p.add_argument('--lip_rs', type=float, nargs='+', default=[1e-2, 5e-3, 2e-3])
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--steps', type=int, nargs='+', default=[10, 20, 50])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    print(f"{'='*70}")
    print(f"  CelebA-HQ eval: Standard vs Lip (r={args.lip_rs})")
    print(f"  Steps: {args.steps}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  {args.num_eval} samples, {args.num_seeds} seeds")
    print(f"{'='*70}")

    # Data
    _, test_data = load_celeba_data(128)
    num_eval = min(args.num_eval, test_data.shape[0])
    truth = test_data[:num_eval]
    print(f"  Test data: {truth.shape}, range [{truth.min():.2f}, {truth.max():.2f}]")

    # Model
    model = Velocity(C=3, dim=64, dim_mults=(1, 2, 4, 4)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Methods: Standard + each Lip r, at each step count
    methods = []
    for ns in args.steps:
        methods.append(('Standard', ns))
    for lr in args.lip_rs:
        for ns in args.steps:
            methods.append((f'Lip r={lr:.0e}', ns))

    # Truth metrics
    truth_m, kvals, truth_spec, _ = compute_image_metrics(truth, truth)

    # Run across seeds
    all_results = {key: [] for key in methods}
    all_specs = {key: [] for key in methods}
    all_fids = {key: [] for key in methods}

    batch_size = 50  # generate in batches to avoid OOM

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx * 1000
        torch.manual_seed(seed)
        z0_all = torch.randn(num_eval, 3, 128, 128)  # on CPU

        for method_label, nsteps in methods:
            print(f"  seed={seed}, {method_label} RK4-{nsteps}...", flush=True)
            gen_chunks = []
            for start in range(0, num_eval, batch_size):
                end = min(start + batch_size, num_eval)
                z0_batch = z0_all[start:end].to(device)
                if method_label == 'Standard':
                    chunk = rk4_standard(model, z0_batch, nsteps)
                else:
                    r_val = float(method_label.split('=')[1])
                    chunk = rk4_lip(model, z0_batch, nsteps, r_val)
                gen_chunks.append(chunk.cpu())
                torch.cuda.empty_cache()
            gen_cpu = torch.cat(gen_chunks, dim=0)

            m, _, _, spec_gen = compute_image_metrics(truth, gen_cpu)
            all_results[(method_label, nsteps)].append(m)
            all_specs[(method_label, nsteps)].append(spec_gen)

            # FID — skip if Inception fails (nvrtc issue)
            try:
                print(f"    Computing FID...", flush=True)
                fid_score = compute_fid(truth, gen_cpu)
                all_fids[(method_label, nsteps)].append(fid_score)
                print(f"    FID = {fid_score:.2f}")
            except Exception as e:
                print(f"    FID failed: {e}")
                all_fids[(method_label, nsteps)].append(float('nan'))

    # ─── Print results ────────────────────────────────────────────────

    def agg(vals):
        arr = np.array(vals)
        return arr.mean(), arr.std()

    print(f"\n{'='*70}")
    print(f"  RESULTS (mean ± std over {args.num_seeds} seeds)")
    print(f"{'='*70}")

    # FID
    print(f"\n--- FID (lower = better) ---")
    print(f"  {'Method':<20} {'steps':>5} {'FID':>16}")
    print(f"  {'-'*44}")
    for key in methods:
        mu, sd = agg(all_fids[key])
        print(f"  {key[0]:<20} {key[1]:>5} {mu:>8.2f} ± {sd:>5.2f}")

    # Spectrum errors
    print(f"\n--- Spectrum relative error by band ---")
    print(f"  {'Method':<20} {'steps':>5} {'low':>12} {'mid':>12} {'high':>12} {'mean':>12}")
    print(f"  {'-'*65}")
    for key in methods:
        vals_low = [m.get('spec_err_low', 0) for m in all_results[key]]
        vals_mid = [m.get('spec_err_mid', 0) for m in all_results[key]]
        vals_high = [m.get('spec_err_high', 0) for m in all_results[key]]
        vals_mean = [m['spec_err_mean'] for m in all_results[key]]
        mu_l, _ = agg(vals_low); mu_m, _ = agg(vals_mid)
        mu_h, _ = agg(vals_high); mu_a, sd_a = agg(vals_mean)
        print(f"  {key[0]:<20} {key[1]:>5} {mu_l:>12.4f} {mu_m:>12.4f} {mu_h:>12.4f} {mu_a:.4f}±{sd_a:.4f}")

    # Image quality
    print(f"\n--- Image quality metrics ---")
    print(f"  Truth: pixel_std={truth_m['pixel_std']:.4f}  grad_kurt={truth_m['grad_kurt']:.4f}")
    print(f"  {'Method':<20} {'steps':>5} {'pixel_std':>10} {'grad_kurt':>10} {'var_ratio':>10}")
    print(f"  {'-'*58}")
    for key in methods:
        ps = [m['pixel_std'] for m in all_results[key]]
        gk = [m['grad_kurt'] for m in all_results[key]]
        vr = [m['total_var_ratio'] for m in all_results[key]]
        print(f"  {key[0]:<20} {key[1]:>5} {agg(ps)[0]:>10.4f} {agg(gk)[0]:>10.4f} {agg(vr)[0]:>10.4f}")

    # ─── Plots ────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    all_colors = {'Standard': 'red'}
    lip_colors = ['blue', 'green', 'purple', 'orange']
    for i, lr in enumerate(args.lip_rs):
        all_colors[f'Lip r={lr:.0e}'] = lip_colors[i % len(lip_colors)]

    # Spectrum at max steps
    ref = max(args.steps)
    ax = axes[0]
    ax.loglog(kvals, truth_spec, 'k-', lw=2.5, label='Truth')
    for key in methods:
        if key[1] == ref:
            specs = all_specs[key]
            mean_spec = np.mean(specs, axis=0)
            c = all_colors.get(key[0], 'gray')
            ls = '--' if 'Standard' in key[0] else '-'
            ax.loglog(kvals, mean_spec, ls, color=c, lw=1.5, label=f'{key[0]}')
    ax.set_xlabel('k'); ax.set_ylabel('Power spectrum')
    ax.set_title(f'Power Spectrum ({ref} steps)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # FID vs steps
    ax = axes[1]
    for method_name in ['Standard'] + [f'Lip r={lr:.0e}' for lr in args.lip_rs]:
        step_fids = []
        for ns in args.steps:
            if (method_name, ns) in all_fids:
                mu, sd = agg(all_fids[(method_name, ns)])
                step_fids.append((ns, mu, sd))
        if step_fids:
            ss, ms, sds = zip(*step_fids)
            c = all_colors.get(method_name, 'gray')
            ls = '--' if 'Standard' in method_name else '-'
            marker = 's' if 'Standard' in method_name else 'o'
            ax.errorbar(ss, ms, yerr=sds, fmt=ls+marker, color=c, lw=2, markersize=6,
                       capsize=4, label=method_name)
    ax.set_xlabel('RK4 steps'); ax.set_ylabel('FID')
    ax.set_title('FID vs Steps'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xticks(args.steps)

    # Per-k error at max steps
    ax = axes[2]
    for key in methods:
        if key[1] == ref:
            specs = all_specs[key]
            mean_spec = np.mean(specs, axis=0)
            rel_err = np.abs(mean_spec - truth_spec) / (np.abs(truth_spec) + 1e-20)
            c = all_colors.get(key[0], 'gray')
            ls = '--' if 'Standard' in key[0] else '-'
            ax.semilogy(kvals, rel_err, ls, color=c, lw=1.5, label=f'{key[0]}')
    ax.axhline(y=0.1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('k'); ax.set_ylabel('Relative error')
    ax.set_title(f'Spectrum Error ({ref} steps)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle('CelebA-HQ: Standard vs Lip Schedule', fontsize=14)
    plt.tight_layout()
    plt.savefig('celeba_lip_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: celeba_lip_comparison.png")
