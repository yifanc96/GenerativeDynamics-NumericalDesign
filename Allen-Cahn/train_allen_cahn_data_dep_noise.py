"""
Flow matching for 1D Allen-Cahn with data-dependent noise.

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

from unet1D import Unet1D

# ─── Data loading ────────────────────────────────────────────────────────────

def load_allen_cahn_data(loc, grid_size, batch_size, train_test_split):
    """Load Allen-Cahn samples from .npy file."""
    data_raw = np.load(loc)
    print(f"[Data] raw shape={data_raw.shape}, dtype={data_raw.dtype}")
    torch_data = torch.from_numpy(data_raw).float()
    torch_data = torch_data.reshape(-1, torch_data.shape[-1])
    print(f"[Data] flattened shape={torch_data.shape}")
    norm_per_pixel = torch.norm(torch_data, dim=1, p='fro').mean() / torch_data.shape[-1]
    print(f"[Data] norm per pixel={norm_per_pixel:.4f}")

    # (N, 1, grid_size)
    torch_data = torch_data[:, None, :grid_size]
    num_train = int(torch_data.shape[0] * train_test_split)
    print(f"[Data] train={num_train}, test={torch_data.shape[0] - num_train}")

    train_loader = DataLoader(TensorDataset(torch_data[:num_train]), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(torch_data[num_train:]), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, torch_data[:num_train], torch_data[num_train:]


# ─── 1D energy spectrum ─────────────────────────────────────────────────────

def get_energy_spectrum1d(data):
    """Compute 1D energy spectrum. data: (N, L) or (N, 1, L)."""
    if data.dim() == 3:
        data = data.squeeze(1)
    fhat = torch.fft.fft(data, dim=1, norm='forward')
    power = (fhat.abs()**2).mean(dim=0)
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    return np.abs(kfreq), power.cpu().numpy()


# ─── Data-dependent noise (1D) ──────────────────────────────────────────────

def make_data_dependent_noise_1d(next_batch, batch_size):
    """
    z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j},  xi ~ N(0,1)
    next_batch: (N, 1, L),  returns: (batch_size, 1, L)
    """
    N = next_batch.shape[0]
    centered = next_batch - next_batch.mean(dim=0, keepdim=True)  # (N, 1, L)
    xi = torch.randn(N, batch_size, device=next_batch.device)     # (N, batch_size)
    noise = torch.einsum('ncl,nb->bcl', centered, xi) / math.sqrt(N)
    return noise


# ─── Network ─────────────────────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = Unet1D(
            config.unet_channels,
            dim_mults=config.unet_dim_mults,
            channels=config.C,
            out_dim=config.C,
            learned_sinusoidal_cond=config.unet_learned_sinusoidal_cond,
            random_fourier_features=config.unet_random_fourier_features,
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Network] {n_params:,} parameters")

    def forward(self, zt, t):
        return self.net(zt, t)


# ─── Interpolants ────────────────────────────────────────────────────────────

class Interpolants:
    """I_t = (1-t)*z0 + t*z1,  R_t = -z0 + z1"""
    @staticmethod
    def It(z0, z1, t):
        tw = t[:, None, None]
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
        dt = tgrid[1] - tgrid[0]
        zt = z0
        for t_val in tgrid:
            zt = zt + self._f(model, zt, t_val) * dt
        return zt

    @torch.no_grad()
    def RK4(self, z0, model, steps):
        """Hand-written classic RK4."""
        tgrid = torch.linspace(self.config.t_min_sample, self.config.t_max_sample, steps).type_as(z0)
        dt = tgrid[1] - tgrid[0]
        zt = z0
        for t_val in tgrid:
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
        self.name = f"AllenCahn_datanoise_grid{config.grid_size}_lr{config.base_lr}_{self.log_base}"

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

    def prepare_data(self):
        cfg = self.config
        self.train_loader, self.test_loader, self.train_data, self.test_data = \
            load_allen_cahn_data(cfg.data_loc, cfg.grid_size, cfg.batch_size, cfg.train_test_split)

    def get_noise(self, batch_data):
        """Data-dependent noise from previous batch; fallback to iid Gaussian."""
        B = batch_data.shape[0]
        if self.prev_batch is None:
            z0 = torch.randn(B, 1, self.config.grid_size, device=self.device)
        else:
            z0 = make_data_dependent_noise_1d(self.prev_batch, B)
        self.prev_batch = batch_data.clone()
        return z0

    def loss_function(self, z0, z1, t):
        zt = Interpolants.It(z0, z1, t)
        target = Interpolants.Rt(z0, z1)
        pred = self.model(zt, t)
        return (pred - target).pow(2).sum(dim=(1, 2)).mean()

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
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e5)
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

        # Use a batch from test data
        num_eval = min(500, self.test_data.shape[0])
        truth = self.test_data[:num_eval].to(self.device)
        truth_np = truth.cpu()
        kvals, spec_truth = get_energy_spectrum1d(truth_np)

        # Data-dependent noise from training data
        perm = torch.randperm(self.train_data.shape[0])
        noise_src = self.train_data[perm[:cfg.batch_size]].to(self.device)
        z0 = make_data_dependent_noise_1d(noise_src, num_eval)

        step_counts = cfg.eval_step_counts
        results = {}

        for nsteps in step_counts:
            for tag, sample_fn in [('RK', lambda z: self.sampler.RK4(z, self.model, nsteps)),
                                   ('EM', lambda z: self.sampler.EM(z, self.model, nsteps))]:
                gen = sample_fn(z0.clone())
                gen_np = gen.cpu()
                _, spec_gen = get_energy_spectrum1d(gen_np)

                std_ratio = gen_np.std().item() / (truth_np.std().item() + 1e-12)
                # Only compare positive-k part
                half = len(spec_truth) // 2
                spec_rel = np.mean(np.abs(spec_truth[:half] - spec_gen[:half]) / (np.abs(spec_truth[:half]) + 1e-12))
                spec_l1 = np.mean(np.abs(spec_truth[:half] - spec_gen[:half]))
                key = f"{tag}{nsteps}"
                results[key] = dict(spec_gen=spec_gen, gen_np=gen_np,
                                    std_ratio=std_ratio, spec_rel=spec_rel, spec_l1=spec_l1)
                print(f"    {tag} steps={nsteps:3d}:  spec_relErr={spec_rel:.4f}  spec_L1={spec_l1:.6f}  std_ratio={std_ratio:.4f}")
                if cfg.use_wandb:
                    wandb.log({
                        f"spec_relErr/{tag}_steps{nsteps}": spec_rel,
                        f"spec_L1/{tag}_steps{nsteps}": spec_l1,
                        f"std_ratio/{tag}_steps{nsteps}": std_ratio,
                    }, step=self.global_step)

        # ── Spectrum plot ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        half = len(kvals) // 2
        ax = axes[0]
        ax.semilogy(kvals[:half], spec_truth[:half], 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.semilogy(kvals[:half], results[f'RK{ns}']['spec_gen'][:half], '--', color=colors[i], label=f'RK4 {ns}')
            ax.semilogy(kvals[:half], results[f'EM{ns}']['spec_gen'][:half], ':', color=colors[i], alpha=0.5, label=f'EM {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)'); ax.legend(fontsize=6); ax.set_title('Energy Spectrum')

        # Sample comparison
        ax = axes[1]
        nmax = step_counts[-1]
        idx = 0
        x = np.arange(cfg.grid_size)
        ax.plot(x, truth_np[idx, 0].numpy(), 'k-', lw=2, label='truth')
        ax.plot(x, results[f'RK{step_counts[0]}']['gen_np'][idx, 0].numpy(), '--', label=f'RK4 {step_counts[0]}')
        ax.plot(x, results[f'RK{nmax}']['gen_np'][idx, 0].numpy(), ':', label=f'RK4 {nmax}')
        ax.legend(fontsize=8); ax.set_title('Sample comparison')
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
        self.data_loc = 'data/AllenCahn1D_grid64_samples.npy'
        self.grid_size = 64
        self.C = 1
        self.batch_size = 1000
        self.train_test_split = 0.9

        # training
        self.base_lr = 2e-4
        self.max_steps = 50000
        self.t_min_train = 0
        self.t_max_train = 1
        self.t_min_sample = 0
        self.t_max_sample = 1
        self.print_loss_every = 50
        self.test_every = 2000
        self.eval_step_counts = [2, 5, 10, 20, 50]

        # architecture (medium, matching existing)
        self.unet_channels = 32
        self.unet_dim_mults = (1, 2, 2, 2)
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False

        self.save_dir = 'results/allen_cahn_data_dep_noise'


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--max_steps', type=int, default=50000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--test_every', type=int, default=2000)
    p.add_argument('--data_loc', type=str, default='data/AllenCahn1D_grid64_samples.npy')
    p.add_argument('--save_dir', type=str, default='results/allen_cahn_data_dep_noise')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    config.grid_size = args.grid_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.test_every = args.test_every
    config.data_loc = args.data_loc
    config.save_dir = args.save_dir
    config.use_wandb = not args.no_wandb

    os.makedirs(config.save_dir, exist_ok=True)

    logger = Logger(config)
    logger.setup_wandb(config)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
