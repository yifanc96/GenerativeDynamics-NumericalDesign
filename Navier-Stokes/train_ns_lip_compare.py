"""
Train standard FM for NS (Gaussian base, lr=1e-4, 5 datasets, 50k steps).
Evaluate with both standard and Lip schedules during training.
Plot spectra of noise, target, and generated samples.
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

# ─── Data loading ────────────────────────────────────────────────────

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
            resnet_block_groups=8,
            learned_sinusoidal_cond=True,
            random_fourier_features=False,
            learned_sinusoidal_dim=32,
            attn_dim_head=32, attn_heads=4,
            use_classes=False,
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Network] {n_params:,} parameters")

    def forward(self, zt, t):
        return self.net(zt, t, classes=None)

# ─── Samplers ────────────────────────────────────────────────────────

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


def make_lip_sampler(r):
    """
    Build Lip/scaled schedule RK4 sampler with given spectral ratio parameter r.
    α(t) = sqrt((r - r^t)/(r-1)), β(t) = sqrt((r^t-1)/(r-1))
    """
    log_r = np.log(r)

    def alpha(t):
        return np.sqrt((r - r**t) / (r - 1))
    def alpha_dot(t):
        a = alpha(t)
        if abs(a) < 1e-30: return 0.0
        return -0.5 / a * (r**t) * log_r / (r - 1)
    def beta(t):
        return np.sqrt((r**t - 1) / (r - 1))
    def beta_dot(t):
        b = beta(t)
        if abs(b) < 1e-30: return 0.0
        return 0.5 / b * (r**t) * log_r / (r - 1)

    @torch.no_grad()
    def rk4_lip(model, z0, steps, t_min=1e-4, t_max=1-1e-4):
        """
        Lip schedule sampler: reparameterize the NN velocity.

        The NN was trained with standard schedule: v_nn(z, t) ≈ E[z1 - z0 | z_t].
        For the Lip schedule, the ODE is dz/ds = b_lip(z, s) where s is the
        reparameterized time. We map s -> t(s) so that α_lip(s) = α_std(t(s)):
          t(s) = α_lip(s) gives the "equivalent standard time" but this isn't
          quite right either.

        Actually, for standard FM: v_nn(z,t) learns the velocity dz/dt for the
        ODE path z_t = (1-t)z0 + t*z1. The Lip schedule defines a DIFFERENT path
        z_s = α(s)z0 + β(s)z1 with velocity dz/ds = α'(s)z0 + β'(s)z1.

        Since the NN approximates v(z,t) = E[z1-z0|z_t=z] = E[R_t|z_t=z],
        and for the Lip schedule the conditional velocity is:
          b(z,s) = [α'(s) + β'(s)] * E[z1-z0|z_s=z] + ... (not simply v_nn)

        The clean approach: use the analytical Gaussian drift with the Lip schedule,
        which gets the 2-point statistics right. The NN handles the non-Gaussian part.

        Here we use the analytical Lip drift with empirical spectra.
        """
        # This function is replaced by the analytical version below
        raise NotImplementedError("Use analytical_lip_sample instead")

    return alpha, alpha_dot, beta, beta_dot


def get_spectral_density_2d(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(1, 2), norm='forward')
    return (torch.abs(fhat)**2).mean(dim=0)


@torch.no_grad()
def analytical_lip_sample(z0, S0_2d, S1_2d, r, steps, t_min=1e-4, t_max=1-1e-4):
    """Analytical Lip schedule drift using empirical spectra and given r."""
    log_r = np.log(r)

    def alpha(t):
        return np.sqrt((r - r**t) / (r - 1))
    def alpha_dot(t):
        a = alpha(t)
        if abs(a) < 1e-30: return 0.0
        return -0.5 / a * (r**t) * log_r / (r - 1)
    def beta(t):
        return np.sqrt((r**t - 1) / (r - 1))
    def beta_dot(t):
        b = beta(t)
        if abs(b) < 1e-30: return 0.0
        return 0.5 / b * (r**t) * log_r / (r - 1)

    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0.clone()
    S0 = S0_2d[None, None]  # (1, 1, H, W)
    S1 = S1_2d[None, None]

    for i in range(len(tgrid) - 1):
        t_i = tgrid[i].item()
        dt = (tgrid[i+1] - tgrid[i]).item()

        def drift(z, t):
            a, da = alpha(t), alpha_dot(t)
            b, db = beta(t), beta_dot(t)
            R_k = (a*da*S0 + b*db*S1) / (a**2*S0 + b**2*S1 + 1e-30)
            zfft = torch.fft.fft2(z, dim=(2, 3), norm='forward')
            return torch.fft.ifft2(R_k * zfft, dim=(2, 3), norm='forward').real

        k1 = drift(zt, t_i)
        k2 = drift(zt + 0.5*dt*k1, t_i + 0.5*dt)
        k3 = drift(zt + 0.5*dt*k2, t_i + 0.5*dt)
        k4 = drift(zt + dt*k3, t_i + dt)
        zt = zt + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return zt

# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate(model, test_data, z0_fixed, S0_2d, S1_2d, lip_r, kvals,
             enst_truth, ener_truth, enst_noise, ener_noise,
             global_step, device, use_wandb, save_dir):
    model.eval()
    num_eval = z0_fixed.shape[0]
    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}

    step_counts_nn = [10, 20, 50]
    step_counts_lip = [10, 20, 50]

    log_dict = {}

    # ── NN standard schedule ──
    print(f"  [Eval step {global_step}] NN standard schedule:")
    for nsteps in step_counts_nn:
        gen = rk4_sample(model, z0_fixed.clone().to(device), nsteps)
        gen_sq = gen.squeeze(1).cpu()
        _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)
        std_ratio = gen_sq.std().item() / test_data[:num_eval].squeeze(1).std().item()

        band_err = {bn: np.mean(np.abs(enst_truth[m] - enst_gen[m]) / (np.abs(enst_truth[m]) + 1e-20))
                    for bn, m in bands.items()}
        mean_err = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-20))

        print(f"    RK4-{nsteps:3d}: std={std_ratio:.4f}  low={band_err['low']:.4f}  mid={band_err['mid']:.4f}  high={band_err['high']:.4f}  mean={mean_err:.4f}")

        log_dict[f"nn_std/enst_mean_RK{nsteps}"] = mean_err
        log_dict[f"nn_std/enst_low_RK{nsteps}"] = band_err['low']
        log_dict[f"nn_std/enst_mid_RK{nsteps}"] = band_err['mid']
        log_dict[f"nn_std/enst_high_RK{nsteps}"] = band_err['high']
        log_dict[f"nn_std/std_ratio_RK{nsteps}"] = std_ratio

        # Save spectrum for the 50-step case for plotting
        if nsteps == 50:
            enst_gen_nn50 = enst_gen
            ener_gen_nn50 = ener_gen

    # ── Analytical Lip schedule ──
    print(f"  [Eval step {global_step}] Analytical Lip schedule (r={lip_r:.1e}):")
    for nsteps in step_counts_lip:
        gen = analytical_lip_sample(z0_fixed.clone(), S0_2d, S1_2d, lip_r, nsteps)
        gen_sq = gen.squeeze(1).cpu()
        _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)
        std_ratio = gen_sq.std().item() / test_data[:num_eval].squeeze(1).std().item()

        band_err = {bn: np.mean(np.abs(enst_truth[m] - enst_gen[m]) / (np.abs(enst_truth[m]) + 1e-20))
                    for bn, m in bands.items()}
        mean_err = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-20))

        print(f"    RK4-{nsteps:3d}: std={std_ratio:.4f}  low={band_err['low']:.4f}  mid={band_err['mid']:.4f}  high={band_err['high']:.4f}  mean={mean_err:.4f}")

        log_dict[f"lip/enst_mean_RK{nsteps}"] = mean_err
        log_dict[f"lip/enst_low_RK{nsteps}"] = band_err['low']
        log_dict[f"lip/enst_mid_RK{nsteps}"] = band_err['mid']
        log_dict[f"lip/enst_high_RK{nsteps}"] = band_err['high']
        log_dict[f"lip/std_ratio_RK{nsteps}"] = std_ratio

        if nsteps == 50:
            enst_gen_lip50 = enst_gen
            ener_gen_lip50 = ener_gen

    # ── Spectrum plot: noise / target / NN-generated / Lip-generated ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.loglog(kvals, enst_noise, 'gray', lw=1.5, alpha=0.6, label='Noise (z0)')
    ax.loglog(kvals, enst_truth, 'k-', lw=2.5, label='Target (data)')
    ax.loglog(kvals, enst_gen_nn50, 'r--', lw=1.5, marker='s', markersize=3,
             markevery=max(1, len(kvals)//10), label='NN standard (50 steps)')
    ax.loglog(kvals, enst_gen_lip50, 'b-', lw=1.5, marker='o', markersize=3,
             markevery=max(1, len(kvals)//10), label='Analytical Lip (50 steps)')
    ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Enstrophy', fontsize=12)
    ax.set_title(f'Enstrophy Spectrum (step {global_step})', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(kvals, ener_noise, 'gray', lw=1.5, alpha=0.6, label='Noise (z0)')
    ax.loglog(kvals, ener_truth, 'k-', lw=2.5, label='Target (data)')
    ax.loglog(kvals, ener_gen_nn50, 'r--', lw=1.5, marker='s', markersize=3,
             markevery=max(1, len(kvals)//10), label='NN standard (50 steps)')
    ax.loglog(kvals, ener_gen_lip50, 'b-', lw=1.5, marker='o', markersize=3,
             markevery=max(1, len(kvals)//10), label='Analytical Lip (50 steps)')
    ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'Energy Spectrum (step {global_step})', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'spectrum_step{global_step}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    if use_wandb:
        log_dict["spectrum"] = wandb.Image(fig)
    plt.close()

    if use_wandb:
        wandb.log(log_dict, step=global_step)

# ─── Training ────────────────────────────────────────────────────────

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
    p.add_argument('--lip_r', type=float, default=1e-5, help='Lip schedule spectral ratio parameter')
    p.add_argument('--save_dir', type=str, default='results/ns_lip_compare')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ──
    suffixes = ['', '02', '03', '04', '05']
    data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes[:args.num_dataset]]
    print(f"Loading {args.num_dataset} NS datasets at {args.hi_size}x{args.hi_size}:")
    train_loader, train_data, test_data = load_ns_data(data_locs, args.hi_size, args.batch_size, 0.9)

    # ── Precompute spectra ──
    num_eval = min(200, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)

    # Noise spectrum
    torch.manual_seed(123)
    z0_noise = torch.randn(num_eval, 1, args.hi_size, args.hi_size)
    noise_sq = z0_noise.squeeze(1)
    _, enst_noise, ener_noise = get_energy_spectrum(noise_sq)

    # Empirical 2D spectral densities for Lip schedule
    S0_2d = get_spectral_density_2d(noise_sq)
    S1_2d = get_spectral_density_2d(train_data[:2000].squeeze(1))
    S0_2d[0, 0] = 1e-30
    S1_2d[0, 0] = 1e-30

    # Fixed z0 for eval (same across all evals)
    torch.manual_seed(42)
    z0_fixed = torch.randn(num_eval, 1, args.hi_size, args.hi_size)

    # ── Model ──
    model = Velocity(C=1, unet_channels=32, unet_dim_mults=(1, 2, 2, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    time_dist = torch.distributions.Uniform(low=1e-3, high=1-1e-3)

    # ── Wandb ──
    use_wandb = not args.no_wandb
    if use_wandb:
        date = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_name = f"NS_gaussbase_hi{args.hi_size}_lr{args.lr}_{date}"
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))
        print(f"[wandb] {run_name}")

    # ── Train ──
    global_step = 0
    t_start = timer()
    print(f"\n[Training] max_steps={args.max_steps}, lr={args.lr}, batch={args.batch_size}")

    # Initial eval
    evaluate(model, test_data, z0_fixed, S0_2d, S1_2d, args.lip_r,
             kvals, enst_truth, ener_truth, enst_noise, ener_noise,
             global_step, device, use_wandb, args.save_dir)

    while global_step < args.max_steps:
        for (batch_data,) in train_loader:
            if global_step >= args.max_steps:
                break

            batch_data = batch_data.to(device)
            z0 = torch.randn_like(batch_data)
            z1 = batch_data
            t = time_dist.sample((z1.shape[0],)).to(device)

            # Interpolant: z_t = (1-t)*z0 + t*z1, target: z1 - z0
            tw = t[:, None, None, None]
            zt = (1 - tw) * z0 + tw * z1
            target = z1 - z0

            model.train()
            pred = model(zt, t)
            loss = (pred - target).pow(2).sum(dim=(1, 2, 3)).mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e4)
            optimizer.step()
            scheduler.step()

            if global_step % 500 == 0:
                elapsed = (timer() - t_start) / 60
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step}/{args.max_steps}  loss={loss.item():.4f}  grad={grad_norm:.2f}  lr={lr_now:.2e}  [{elapsed:.1f}m]")
                if use_wandb:
                    wandb.log({"loss": loss.item(), "grad_norm": grad_norm, "lr": lr_now}, step=global_step)

            if global_step > 0 and global_step % args.test_every == 0:
                evaluate(model, test_data, z0_fixed, S0_2d, S1_2d, args.lip_r,
                         kvals, enst_truth, ener_truth, enst_noise, ener_noise,
                         global_step, device, use_wandb, args.save_dir)

            global_step += 1

    # ── Final eval and save ──
    print("\n[Training] Done. Final evaluation:")
    evaluate(model, test_data, z0_fixed, S0_2d, S1_2d, args.lip_r,
             kvals, enst_truth, ener_truth, enst_noise, ener_noise,
             global_step, device, use_wandb, args.save_dir)

    save_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] {save_path}")

    if use_wandb:
        wandb.finish()
