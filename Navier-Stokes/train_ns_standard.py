"""
Train standard FM for NS (Gaussian base). Periodic eval with wandb.
After training, run eval_ns_lip_compare.py for Lip schedule comparison.
"""
import os, sys, math, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import wandb
from time import time as timer
import scipy.stats as stats

from unet import Unet

# ─── Data ────────────────────────────────────────────────────────────

def load_ns_data(data_locs, hi_size, batch_size, train_test_split):
    if isinstance(data_locs, str):
        data_locs = [data_locs]
    avg_pixel_norm = 3.0679163932800293
    all_data = []
    for loc in data_locs:
        data_raw, _ = torch.load(loc, weights_only=False)
        Ntj, Nts, Nx, Ny = data_raw.shape
        print(f"  {loc}: {Ntj}x{Nts}x{Nx}x{Ny}")
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, Nx, Ny)
        if hi_size != Nx:
            data = nn.functional.interpolate(
                data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear'
            ).squeeze(1)
        all_data.append(data)
    data = torch.cat(all_data, dim=0)[:, None, :, :]
    num_train = int(data.shape[0] * train_test_split)
    print(f"  Total: {data.shape[0]}, train={num_train}, test={data.shape[0]-num_train}, std={data.std():.4f}")
    train_loader = DataLoader(TensorDataset(data[:num_train]), batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, data[:num_train], data[num_train:]

# ─── Spectrum ────────────────────────────────────────────────────────

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

# ─── Network ─────────────────────────────────────────────────────────

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
        print(f"[Network] {sum(p.numel() for p in self.parameters()):,} params")
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)

# ─── Samplers ────────────────────────────────────────────────────────

@torch.no_grad()
def rk4_sample(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
    """Standard schedule: α=1-t, β=t. N grid points → N-1 RK4 steps."""
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


def make_lip_drift(model, ratio):
    """
    Transfer formula (from Lip paper / notebook Cell 9-10):
    Convert standard-schedule NN to Lip schedule drift.

    b(z,t) = (α̇/α)·z + (β̇ - α̇β/α)·E[z₁|z_t=z]
    where:
      orig_t = 1/(1 + α/β)          — equivalent standard time
      orig_x = orig_t/β · z_t       — rescaled state
      E[z₁|z_t] = (1-orig_t)·v_nn(orig_x, orig_t) + orig_x
    """
    r = ratio
    log_r = math.log(r)

    # Matching notebook exactly: torch.ones_like(t) ensures correct shape
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

    return lip_drift


@torch.no_grad()
def rk4_sample_lip(model, z0, steps, ratio, t_min=1e-3, t_max=1-1e-3):
    """Lip schedule using transfer formula with trained NN (RK4)."""
    lip_drift = make_lip_drift(model, ratio)
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

# ─── Eval ────────────────────────────────────────────────────────────

def evaluate(model, test_data, z0_fixed, kvals, enst_truth, enst_noise,
             global_step, device, use_wandb, lip_rs=[1e-5, 1e-6, 1e-7]):
    model.eval()
    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}
    step_counts = [10, 20, 50]
    log_dict = {}
    truth_std = test_data[:z0_fixed.shape[0]].squeeze(1).std().item()

    enst_gens = {}  # (method_label, nsteps) -> enst_gen

    # Standard schedule
    methods = [('std', lambda z, n: rk4_sample(model, z, n))]
    # Lip schedules for each r
    for lr in lip_rs:
        label = f'lip_r{lr:.0e}'
        methods.append((label, lambda z, n, _r=lr: rk4_sample_lip(model, z, n, _r)))

    for method_name, sample_fn in methods:
        print(f"  [Eval step {global_step}] {method_name}:")
        for nsteps in step_counts:
            gen = sample_fn(z0_fixed.clone().to(device), nsteps)
            gen_sq = gen.squeeze(1).cpu()
            _, enst_gen, _ = get_energy_spectrum(gen_sq)
            std_ratio = gen_sq.std().item() / truth_std
            enst_gens[(method_name, nsteps)] = enst_gen

            band_err = {bn: np.mean(np.abs(enst_truth[m] - enst_gen[m]) / (np.abs(enst_truth[m]) + 1e-20))
                        for bn, m in bands.items()}
            mean_err = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-20))

            print(f"    RK4-{nsteps:3d}: std={std_ratio:.4f}  low={band_err['low']:.4f}  mid={band_err['mid']:.4f}  high={band_err['high']:.4f}  mean={mean_err:.4f}")

            log_dict[f"{method_name}/enst_mean_RK{nsteps}"] = mean_err
            log_dict[f"{method_name}/enst_low_RK{nsteps}"] = band_err['low']
            log_dict[f"{method_name}/enst_mid_RK{nsteps}"] = band_err['mid']
            log_dict[f"{method_name}/enst_high_RK{nsteps}"] = band_err['high']
            log_dict[f"{method_name}/std_ratio_RK{nsteps}"] = std_ratio

    # Spectrum plot: noise, target, std, and all Lip variants at 20 steps
    ref = 20
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(kvals, enst_noise, 'gray', lw=1.5, alpha=0.6, label='Noise (z0)')
    ax.loglog(kvals, enst_truth, 'k-', lw=2.5, label='Target (data)')
    if ('std', ref) in enst_gens:
        ax.loglog(kvals, enst_gens[('std', ref)], 'r--', lw=1.5, label=f'Standard ({ref} steps)')
    lip_colors = ['blue', 'green', 'purple']
    for i, lr in enumerate(lip_rs):
        label = f'lip_r{lr:.0e}'
        if (label, ref) in enst_gens:
            ax.loglog(kvals, enst_gens[(label, ref)], '-', color=lip_colors[i % len(lip_colors)],
                     lw=1.5, label=f'Lip r={lr:.0e} ({ref} steps)')
    ax.set_xlabel('k'); ax.set_ylabel('Enstrophy')
    ax.set_title(f'Enstrophy Spectrum (step {global_step})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if use_wandb:
        log_dict["spectrum"] = wandb.Image(fig)
    plt.close()

    if use_wandb:
        wandb.log(log_dict, step=global_step)

# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_steps', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_dataset', type=int, default=5)
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--test_every', type=int, default=5000)
    p.add_argument('--noise_strength', type=float, default=1.0,
                   help='Noise scaling: z0 = noise_strength * randn. '
                        'Use 1.0 for standard Gaussian, 10.0 for scaled noise.')
    p.add_argument('--lip_rs', type=float, nargs='+', default=[1e-5, 1e-6, 1e-7],
                   help='Lip schedule ratio parameters to eval. '
                        'Multiple values compared during training.')
    p.add_argument('--save_dir', type=str, default='results/ns_standard_fm')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Print setup clearly ──
    print("="*70)
    print(f"  SETUP: noise_strength={args.noise_strength}, lip_rs={args.lip_rs}")
    print(f"  z0 = {args.noise_strength} * N(0,I)  →  Var(z0) = {args.noise_strength**2:.1f}")
    print(f"  Lip schedule ratios r = {args.lip_rs}")
    print(f"  lr={args.lr}, {args.num_dataset} datasets, {args.max_steps} steps")
    print("="*70)

    # Data
    suffixes = ['', '02', '03', '04', '05']
    data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes[:args.num_dataset]]
    print(f"\nLoading {args.num_dataset} NS datasets at {args.hi_size}x{args.hi_size}:")
    train_loader, train_data, test_data = load_ns_data(data_locs, args.hi_size, args.batch_size, 0.9)

    # Precompute spectra
    num_eval = min(200, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, enst_truth, _ = get_energy_spectrum(truth_sq)

    # Noise spectrum (with noise_strength scaling)
    torch.manual_seed(123)
    noise_sq = args.noise_strength * torch.randn(num_eval, args.hi_size, args.hi_size)
    _, enst_noise, _ = get_energy_spectrum(noise_sq)
    print(f"  Noise std: {noise_sq.std():.4f}, Data std: {truth_sq.std():.4f}")

    # Fixed z0 for eval (with noise_strength scaling)
    torch.manual_seed(42)
    z0_fixed = args.noise_strength * torch.randn(num_eval, 1, args.hi_size, args.hi_size)

    # Model
    model = Velocity(C=1, unet_channels=32, unet_dim_mults=(1, 2, 2, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    time_dist = torch.distributions.Uniform(low=1e-3, high=1-1e-3)

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        date = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_name = f"NS_gauss_noise{args.noise_strength}_hi{args.hi_size}_lr{args.lr}_{date}"
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))
        print(f"[wandb] {run_name}")

    # Train
    global_step = 0
    t_start = timer()
    ns = args.noise_strength
    print(f"\n[Training] max_steps={args.max_steps}, lr={args.lr}, noise_strength={ns}, lip_rs={args.lip_rs}")
    evaluate(model, test_data, z0_fixed, kvals, enst_truth, enst_noise,
             0, device, use_wandb, lip_rs=args.lip_rs)

    while global_step < args.max_steps:
        for (batch_data,) in train_loader:
            if global_step >= args.max_steps:
                break
            batch_data = batch_data.to(device)
            z0 = ns * torch.randn_like(batch_data)
            z1 = batch_data
            t = time_dist.sample((z1.shape[0],)).to(device)
            tw = t[:, None, None, None]
            zt = (1 - tw) * z0 + tw * z1
            target = z1 - z0

            model.train()
            loss = (model(zt, t) - target).pow(2).sum(dim=(1, 2, 3)).mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e4)
            optimizer.step()
            scheduler.step()

            if global_step % 500 == 0:
                elapsed = (timer() - t_start) / 60
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step}/{args.max_steps}  loss={loss.item():.4f}  lr={lr_now:.2e}  [{elapsed:.1f}m]")
                if use_wandb:
                    wandb.log({"loss": loss.item(), "grad_norm": grad_norm, "lr": lr_now}, step=global_step)

            if global_step > 0 and global_step % args.test_every == 0:
                evaluate(model, test_data, z0_fixed, kvals, enst_truth, enst_noise,
                         global_step, device, use_wandb, lip_rs=args.lip_rs)
            global_step += 1

    print("\n[Training] Done.")
    evaluate(model, test_data, z0_fixed, kvals, enst_truth, enst_noise,
             global_step, device, use_wandb, lip_rs=args.lip_rs)
    save_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] {save_path}")
    if use_wandb:
        wandb.finish()
