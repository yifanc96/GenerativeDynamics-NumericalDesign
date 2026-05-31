"""
Flow matching for 2D Navier-Stokes (unconditional) with data-dependent noise.

Noise construction: given data {x_i} from the next batch,
  z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j},  xi_{i,j} ~ N(0,1)

This gives noise with the empirical covariance of the data, making the
learned drift more well-conditioned — so coarse integration (few steps) suffices.
"""

import os, sys, math, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import wandb
from time import time as timer
import scipy.stats as stats

from unet import Unet

# ─── Data loading ────────────────────────────────────────────────────────────

def load_ns_data(data_locs, hi_size, batch_size, train_test_split):
    """
    Load NS vorticity data from one or more files.
    Each file contains (data, time) tuple with data: (num_traj, num_snapshots, Nx, Ny).
    For unconditional generation, flatten trajectories into individual snapshots.
    """
    if isinstance(data_locs, str):
        data_locs = [data_locs]

    avg_pixel_norm = 3.0679163932800293  # fixed across datasets

    all_data = []
    for loc in data_locs:
        data_raw, _ = torch.load(loc)
        Ntj, Nts, Nx, Ny = data_raw.shape
        print(f"[Data] {loc}: {Ntj} traj x {Nts} snapshots x {Nx}x{Ny}")
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, Nx, Ny)
        if hi_size != Nx:
            data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        all_data.append(data)

    data = torch.cat(all_data, dim=0)[:, None, :, :]  # (N, 1, H, W)
    print(f"[Data] total: {data.shape}, std={data.std():.4f}")

    num_train = int(data.shape[0] * train_test_split)
    print(f"[Data] train={num_train}, test={data.shape[0] - num_train}")

    train_loader = DataLoader(TensorDataset(data[:num_train]), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(data[num_train:]), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, data[:num_train], data[num_train:], avg_pixel_norm


# ─── 2D energy spectrum ─────────────────────────────────────────────────────

def get_energy_spectrum(data):
    """
    Compute radially-averaged enstrophy and energy spectra.
    data: (N, H, W) numpy or torch tensor.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(1, 2), norm='forward')
    fourier_amp = (torch.abs(fhat)**2 * (2 * np.pi)).mean(dim=0)
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()

    fourier_flat = fourier_amp.numpy().flatten()
    laplace = knrm**2
    laplace[0] = 1.0
    energy_flat = fourier_flat / laplace

    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    enstrophy, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    enstrophy *= area_weight

    energy, _, _ = stats.binned_statistic(knrm, energy_flat, statistic='mean', bins=kbins)
    energy *= area_weight

    return kvals, enstrophy, energy


# ─── Data-dependent noise (2D) ──────────────────────────────────────────────

def make_data_dependent_noise(next_batch, batch_size):
    """
    z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j},  xi ~ N(0,1)
    next_batch: (N, 1, H, W),  returns: (batch_size, 1, H, W)
    """
    N = next_batch.shape[0]
    centered = next_batch - next_batch.mean(dim=0, keepdim=True)  # (N, 1, H, W)
    xi = torch.randn(N, batch_size, device=next_batch.device)     # (N, batch_size)
    noise = torch.einsum('nchw,nb->bchw', centered, xi) / math.sqrt(N)
    return noise


# ─── Network ─────────────────────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=config.C, out_channels=config.C,
            dim=config.unet_channels, dim_mults=config.unet_dim_mults,
            resnet_block_groups=config.unet_resnet_block_groups,
            learned_sinusoidal_cond=config.unet_learned_sinusoidal_cond,
            random_fourier_features=config.unet_random_fourier_features,
            learned_sinusoidal_dim=config.unet_learned_sinusoidal_dim,
            attn_dim_head=config.unet_attn_dim_head,
            attn_heads=config.unet_attn_heads,
            use_classes=False,
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Network] {n_params:,} parameters")

    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


# ─── Interpolants ────────────────────────────────────────────────────────────

class Interpolants:
    """I_t = (1-t)*z0 + t*z1,  R_t = -z0 + z1"""
    @staticmethod
    def It(z0, z1, t):
        tw = t[:, None, None, None]
        return (1 - tw) * z0 + tw * z1

    @staticmethod
    def Rt(z0, z1):
        return -z0 + z1


# ─── Sampler ─────────────────────────────────────────────────────────────────

class Sampler:
    def __init__(self, config):
        self.config = config

    def _f(self, model, zt, t_val):
        ones = torch.ones(zt.shape[0], device=zt.device) * t_val
        return model(zt, ones)

    @torch.no_grad()
    def EM(self, z0, model, steps):
        tgrid = torch.linspace(self.config.t_min_sample, self.config.t_max_sample, steps).type_as(z0)
        zt = z0
        for i in range(len(tgrid) - 1):
            t_val = tgrid[i]
            dt = tgrid[i+1] - tgrid[i]
            zt = zt + self._f(model, zt, t_val) * dt
        return zt

    @torch.no_grad()
    def RK4(self, z0, model, steps):
        """Hand-written classic RK4."""
        tgrid = torch.linspace(self.config.t_min_sample, self.config.t_max_sample, steps).type_as(z0)
        zt = z0
        for i in range(len(tgrid) - 1):
            t_val = tgrid[i]
            dt = tgrid[i+1] - tgrid[i]
            k1 = self._f(model, zt, t_val)
            k2 = self._f(model, zt + 0.5 * dt * k1, t_val + 0.5 * dt)
            k3 = self._f(model, zt + 0.5 * dt * k2, t_val + 0.5 * dt)
            k4 = self._f(model, zt + dt * k3, t_val + dt)
            zt = zt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return zt


# ─── Logger ──────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, config):
        date = str(datetime.datetime.now())
        self.log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        floss_tag = '_floss' if config.fourier_loss else ''
        self.name = f"NS_datanoise_hi{config.hi_size}_lr{config.base_lr}{floss_tag}_{self.log_base}"

    def setup_wandb(self, config):
        if config.use_wandb:
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=self.name)
            for key, val in vars(config).items():
                if isinstance(val, (int, float, bool, str, list, tuple)):
                    setattr(wandb.config, key, val)
            print("[wandb] setup done")


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_data()
        self.model = Velocity(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_steps)
        self.sampler = Sampler(config)
        self.time_dist = torch.distributions.Uniform(low=config.t_min_train, high=config.t_max_train)
        self.global_step = 0
        self.prev_batch = None
        if config.fourier_loss:
            self._precompute_fourier_weights()

    def _precompute_fourier_weights(self):
        """Compute per-mode inverse-variance weights from training data.
        w(k) = 1/Sigma(k)^alpha, where alpha controls reweighting strength.
        """
        alpha = self.config.fourier_loss_alpha
        data = self.train_data.squeeze(1)  # (N, L, L)
        fhat = torch.fft.fftn(data, dim=(1, 2), norm='ortho')
        sigma_k = (fhat.abs() ** 2).mean(dim=0)  # (L, L)
        w = 1.0 / torch.clamp(sigma_k, min=1e-6).pow(alpha)
        L = data.shape[-1]
        w = w * (L * L) / w.sum()
        self.fourier_weights = w[None, None, :, :].to(self.device)
        print(f"[FourierLoss] alpha={alpha}, weight range: [{w.min():.4e}, {w.max():.4e}], "
              f"sigma_k range: [{sigma_k.min():.4e}, {sigma_k.max():.4e}]")

    def prepare_data(self):
        cfg = self.config
        self.train_loader, self.test_loader, self.train_data, self.test_data, self.avg_pixel_norm = \
            load_ns_data(cfg.data_locs, cfg.hi_size, cfg.batch_size, cfg.train_test_split)

    def get_noise(self, batch_data):
        B = batch_data.shape[0]
        if self.prev_batch is None:
            z0 = torch.randn(B, 1, self.config.hi_size, self.config.hi_size, device=self.device)
        else:
            z0 = make_data_dependent_noise(self.prev_batch, B)
        self.prev_batch = batch_data.clone()
        return z0

    def loss_function(self, z0, z1, t):
        zt = Interpolants.It(z0, z1, t)
        target = Interpolants.Rt(z0, z1)
        pred = self.model(zt, t)
        err = pred - target
        if self.config.fourier_loss:
            err_fft = torch.fft.fftn(err, dim=(2, 3), norm='ortho')
            weighted = self.fourier_weights * err_fft.abs().pow(2)
            return weighted.sum(dim=(1, 2, 3)).mean()
        else:
            return err.pow(2).sum(dim=(1, 2, 3)).mean()

    def fit(self):
        cfg = self.config
        t_start = timer()
        print(f"[Training] max_steps={cfg.max_steps}, batch={cfg.batch_size}, lr={cfg.base_lr}")
        self.test_model()

        while self.global_step < cfg.max_steps:
            for (batch_data,) in self.train_loader:
                if self.global_step >= cfg.max_steps:
                    break

                batch_data = batch_data.to(self.device)
                z0 = self.get_noise(batch_data)
                z1 = batch_data
                t = self.time_dist.sample((z1.shape[0],)).to(self.device)

                self.model.train()
                loss = self.loss_function(z0, z1, t)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e4)
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % cfg.print_loss_every == 0:
                    elapsed = (timer() - t_start) / 60
                    lr_now = self.scheduler.get_last_lr()[0]
                    print(f"  step {self.global_step}/{cfg.max_steps}  loss={loss.item():.4f}  grad={grad_norm:.2f}  lr={lr_now:.2e}  [{elapsed:.1f}m]")
                    if cfg.use_wandb:
                        wandb.log({"loss": loss.item(), "grad_norm": grad_norm, "lr": lr_now}, step=self.global_step)

                if self.global_step > 0 and self.global_step % cfg.test_every == 0:
                    self.test_model()

                self.global_step += 1

            # adjust LR at epoch boundary
            if cfg.cosine_scheduler:
                scale = self.global_step / cfg.max_steps
                lr = cfg.base_lr * 0.5 * (1. + math.cos(math.pi * scale))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

        print("[Training] Done.")
        self.test_model()
        save_path = os.path.join(cfg.save_dir, 'model_final.pt')
        torch.save(self.model.state_dict(), save_path)
        print(f"[Saved] {save_path}")

    # ── Evaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def test_model(self):
        cfg = self.config
        self.model.eval()

        num_eval = min(200, self.test_data.shape[0])
        truth = self.test_data[:num_eval]
        truth_sq = truth.squeeze(1)  # (N, H, W)
        kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)

        # Data-dependent noise: each chunk uses a fresh random batch
        z0_chunks = []
        remaining = num_eval
        while remaining > 0:
            chunk = min(cfg.batch_size, remaining)
            perm = torch.randperm(self.train_data.shape[0])
            noise_src = self.train_data[perm[:cfg.batch_size]].to(self.device)
            z0_chunks.append(make_data_dependent_noise(noise_src, chunk))
            remaining -= chunk
        z0 = torch.cat(z0_chunks, dim=0)

        step_counts = cfg.eval_step_counts
        results = {}

        for nsteps in step_counts:
            for tag, sample_fn in [('RK', lambda z: self.sampler.RK4(z, self.model, nsteps)),
                                   ('EM', lambda z: self.sampler.EM(z, self.model, nsteps))]:
                gen = sample_fn(z0.clone())
                gen_sq = gen.squeeze(1).cpu()
                _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)

                std_ratio = gen_sq.std().item() / (truth_sq.std().item() + 1e-12)

                # Per-band relative errors: low (k<8), mid (8<=k<24), high (k>=24)
                bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}
                band_metrics = {}
                for bname, mask in bands.items():
                    if mask.sum() == 0:
                        continue
                    enst_err = np.mean(np.abs(enst_truth[mask] - enst_gen[mask]) / (np.abs(enst_truth[mask]) + 1e-12))
                    ener_err = np.mean(np.abs(ener_truth[mask] - ener_gen[mask]) / (np.abs(ener_truth[mask]) + 1e-12))
                    band_metrics[bname] = (enst_err, ener_err)

                # Overall (for backward compat)
                enst_rel = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-12))
                ener_rel = np.mean(np.abs(ener_truth - ener_gen) / (np.abs(ener_truth) + 1e-12))

                key = f"{tag}{nsteps}"
                results[key] = dict(enst_gen=enst_gen, ener_gen=ener_gen, gen_sq=gen_sq,
                                    std_ratio=std_ratio, enst_rel=enst_rel, ener_rel=ener_rel,
                                    band_metrics=band_metrics)

                band_str = '  '.join(f"{b}: enst={v[0]:.4f} ener={v[1]:.4f}" for b, v in band_metrics.items())
                print(f"    {tag} steps={nsteps:3d}:  std_ratio={std_ratio:.4f}  {band_str}")
                if cfg.use_wandb:
                    log_dict = {
                        f"enstrophy_relErr/{tag}_steps{nsteps}": enst_rel,
                        f"energy_relErr/{tag}_steps{nsteps}": ener_rel,
                        f"std_ratio/{tag}_steps{nsteps}": std_ratio,
                    }
                    for bname, (enst_err, ener_err) in band_metrics.items():
                        log_dict[f"enstrophy_relErr_{bname}/{tag}_steps{nsteps}"] = enst_err
                        log_dict[f"energy_relErr_{bname}/{tag}_steps{nsteps}"] = ener_err
                    wandb.log(log_dict, step=self.global_step)

        # ── Spectrum + sample plot ──
        import seaborn as sns
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Enstrophy spectrum
        ax = axes[0]
        ax.loglog(kvals, enst_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'RK{ns}']['enst_gen'], '--', color=colors[i], label=f'RK4 {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('Enstrophy'); ax.legend(fontsize=6); ax.set_title('Enstrophy Spectrum')

        # Energy spectrum
        ax = axes[1]
        ax.loglog(kvals, ener_truth, 'k-', lw=2, label='truth')
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'RK{ns}']['ener_gen'], '--', color=colors[i], label=f'RK4 {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('Energy'); ax.legend(fontsize=6); ax.set_title('Energy Spectrum')

        # Sample visualization
        nmax = step_counts[-1]
        vmax = max(abs(truth_sq[0].min().item()), abs(truth_sq[0].max().item()))
        combined = torch.cat([truth_sq[0], results[f'RK{step_counts[0]}']['gen_sq'][0],
                              results[f'RK{nmax}']['gen_sq'][0]], dim=1).numpy()
        axes[2].imshow(combined, cmap=sns.cm.icefire, vmin=-2, vmax=2)
        axes[2].set_title(f'Truth | RK4 {step_counts[0]} | RK4 {nmax}'); axes[2].axis('off')

        plt.tight_layout()
        if cfg.use_wandb:
            wandb.log({"spectrum_comparison": wandb.Image(fig)}, step=self.global_step)
        plt.close()


# ─── Config ──────────────────────────────────────────────────────────────────

class Config:
    def __init__(self):
        self.use_wandb = True
        self.wandb_project = 'interpolants-design'
        self.wandb_entity = 'yifanc96'

        # data
        self.data_locs = ['../NSdata/data_file.pt']
        self.hi_size = 128
        self.C = 1
        self.batch_size = 100
        self.train_test_split = 0.9

        # training
        self.base_lr = 2e-4
        self.max_steps = 50000
        self.cosine_scheduler = True
        self.t_min_train = 1e-3
        self.t_max_train = 1 - 1e-3
        self.t_min_sample = 1e-3
        self.t_max_sample = 1 - 1e-3
        self.print_loss_every = 50
        self.test_every = 2000
        self.eval_step_counts = [2, 5, 10, 20, 50]

        # architecture (medium, matching existing)
        self.unet_channels = 32
        self.unet_dim_mults = (1, 2, 2, 2)
        self.unet_resnet_block_groups = 8
        self.unet_learned_sinusoidal_dim = 32
        self.unet_attn_dim_head = 32
        self.unet_attn_heads = 4
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False

        self.save_dir = 'results/ns_data_dep_noise'
        self.fourier_loss = False
        self.fourier_loss_alpha = 1.0


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--max_steps', type=int, default=50000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--test_every', type=int, default=2000)
    p.add_argument('--data_locs', type=str, nargs='+', default=['../NSdata/data_file.pt'])
    p.add_argument('--num_dataset', type=int, default=None, help='shortcut: use first N of the 5 NS data files')
    p.add_argument('--save_dir', type=str, default='results/ns_data_dep_noise')
    p.add_argument('--unet_channels', type=int, default=32)
    p.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 2, 2])
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--fourier_loss', action='store_true')
    p.add_argument('--fourier_loss_alpha', type=float, default=1.0)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    config.hi_size = args.hi_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.test_every = args.test_every
    if args.num_dataset is not None:
        suffixes = ['', '02', '03', '04', '05']
        config.data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes[:args.num_dataset]]
    else:
        config.data_locs = args.data_locs
    config.save_dir = args.save_dir
    config.unet_channels = args.unet_channels
    config.unet_dim_mults = tuple(args.unet_dim_mults)
    config.use_wandb = not args.no_wandb
    config.fourier_loss = args.fourier_loss
    config.fourier_loss_alpha = args.fourier_loss_alpha

    os.makedirs(config.save_dir, exist_ok=True)

    logger = Logger(config)
    logger.setup_wandb(config)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
