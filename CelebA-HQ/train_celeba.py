"""
Flow matching for CelebA-HQ 128x128 (unconditional).
Supports both standard Gaussian base and data-dependent noise.

Usage:
  python train_celeba.py --gpu 0 --noise_type gauss --save_dir results/celeba_gauss
  python train_celeba.py --gpu 0 --noise_type data_dep --save_dir results/celeba_data_dep
"""

import os, sys, math, datetime, argparse

# Parse args before importing torch so CUDA_VISIBLE_DEVICES takes effect
p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--noise_type', type=str, default='gauss', choices=['gauss', 'data_dep'])
p.add_argument('--hi_size', type=int, default=128)
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--max_steps', type=int, default=50000)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--test_every', type=int, default=2000)
p.add_argument('--unet_channels', type=int, default=64)
p.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 4])
p.add_argument('--save_dir', type=str, default='results/celeba')
p.add_argument('--no_wandb', action='store_true')
p.add_argument('--data_cache', type=str, default='./celeba_data', help='where to cache HF dataset')
args = p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

def load_celeba_data(hi_size, batch_size, train_test_split=0.95, cache_dir='./celeba_data'):
    """
    Load CelebA-HQ from HuggingFace, resize to hi_size, normalize to [-1, 1].
    Returns train_loader, test_loader, train_data, test_data.
    """
    from datasets import load_dataset
    from torchvision import transforms

    cache_pt = os.path.join(cache_dir, f'celeba_hq_{hi_size}.pt')
    if os.path.exists(cache_pt):
        print(f"[Data] Loading cached tensor from {cache_pt}")
        data = torch.load(cache_pt)
    else:
        print("[Data] Downloading CelebA-HQ from HuggingFace...")
        ds = load_dataset("nielsr/CelebA-faces", split="train", cache_dir=cache_dir)
        transform = transforms.Compose([
            transforms.Resize(hi_size),
            transforms.CenterCrop(hi_size),
            transforms.ToTensor(),  # [0, 1]
        ])
        imgs = []
        for i, item in enumerate(ds):
            img = transform(item['image'])
            imgs.append(img)
            if (i + 1) % 5000 == 0:
                print(f"  processed {i+1}/{len(ds)}")
        data = torch.stack(imgs)  # (N, 3, H, W) in [0, 1]
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(data, cache_pt)
        print(f"[Data] Saved cache to {cache_pt}")

    # Normalize to [-1, 1]
    data = data * 2 - 1
    print(f"[Data] shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}], std={data.std():.4f}")

    num_train = int(data.shape[0] * train_test_split)
    print(f"[Data] train={num_train}, test={data.shape[0] - num_train}")

    train_loader = DataLoader(TensorDataset(data[:num_train]), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(data[num_train:]), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, data[:num_train], data[num_train:]


# ─── 2D power spectrum ──────────────────────────────────────────────────────

def get_power_spectrum(data):
    """
    Radially-averaged power spectrum for (N, C, H, W) images.
    Average over channels, then radially bin.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    # Average spectrum over channels
    fhat = torch.fft.fftn(data, dim=(2, 3), norm='forward')
    fourier_amp = (torch.abs(fhat)**2).mean(dim=(0, 1))  # avg over batch and channels
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    fourier_flat = fourier_amp.numpy().flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    Abins *= area_weight
    return kvals, Abins


# ─── Data-dependent noise ───────────────────────────────────────────────────

def make_data_dependent_noise(batch, num_samples):
    """
    z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j}
    batch: (N, C, H, W), returns: (num_samples, C, H, W)
    """
    N = batch.shape[0]
    centered = batch - batch.mean(dim=0, keepdim=True)
    xi = torch.randn(N, num_samples, device=batch.device)
    noise = torch.einsum('nchw,nb->bchw', centered, xi) / math.sqrt(N)
    return noise


# ─── Network ────────────────────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=config.C, out_channels=config.C,
            dim=config.unet_channels, dim_mults=config.unet_dim_mults,
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


# ─── Interpolants & Sampler ─────────────────────────────────────────────────

class Interpolants:
    @staticmethod
    def It(z0, z1, t):
        tw = t[:, None, None, None]
        return (1 - tw) * z0 + tw * z1

    @staticmethod
    def Rt(z0, z1):
        return -z0 + z1


class Sampler:
    def __init__(self, config):
        self.config = config

    def _f(self, model, zt, t_val):
        ones = torch.ones(zt.shape[0], device=zt.device) * t_val
        return model(zt, ones)

    @torch.no_grad()
    def RK4(self, z0, model, steps):
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


# ─── Logger ─────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, config):
        date = str(datetime.datetime.now())
        self.log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.name = f"CelebA_{config.noise_type}_ch{config.unet_channels}_lr{config.base_lr}_{self.log_base}"

    def setup_wandb(self, config):
        if config.use_wandb:
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=self.name)
            for key, val in vars(config).items():
                if isinstance(val, (int, float, bool, str, list, tuple)):
                    setattr(wandb.config, key, val)
            print("[wandb] setup done")


# ─── Trainer ────────────────────────────────────────────────────────────────

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
            load_celeba_data(cfg.hi_size, cfg.batch_size, cache_dir=cfg.data_cache)

    def get_noise(self, batch_data):
        B = batch_data.shape[0]
        if self.config.noise_type == 'gauss':
            return torch.randn_like(batch_data)
        else:
            # data-dependent noise from previous batch
            if self.prev_batch is None:
                z0 = torch.randn_like(batch_data)
            else:
                z0 = make_data_dependent_noise(self.prev_batch, B)
            self.prev_batch = batch_data.clone()
            return z0

    def loss_function(self, z0, z1, t):
        zt = Interpolants.It(z0, z1, t)
        target = Interpolants.Rt(z0, z1)
        pred = self.model(zt, t)
        return (pred - target).pow(2).sum(dim=(1, 2, 3)).mean()

    def fit(self):
        cfg = self.config
        t_start = timer()
        print(f"[Training] noise_type={cfg.noise_type}, max_steps={cfg.max_steps}, batch={cfg.batch_size}, lr={cfg.base_lr}")
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

        num_eval = min(100, self.test_data.shape[0])
        truth = self.test_data[:num_eval]
        kvals, spec_truth = get_power_spectrum(truth)

        # Generate noise
        if cfg.noise_type == 'gauss':
            z0 = torch.randn(num_eval, cfg.C, cfg.hi_size, cfg.hi_size, device=self.device)
        else:
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
        bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}

        for nsteps in step_counts:
            gen = self.sampler.RK4(z0.clone(), self.model, nsteps)
            gen_cpu = gen.cpu()
            _, spec_gen = get_power_spectrum(gen_cpu)

            std_ratio = gen_cpu.std().item() / (truth.std().item() + 1e-12)

            band_metrics = {}
            for bname, mask in bands.items():
                if mask.sum() == 0:
                    continue
                rel = np.mean(np.abs(spec_truth[mask] - spec_gen[mask]) / (np.abs(spec_truth[mask]) + 1e-12))
                band_metrics[bname] = rel

            spec_l1 = np.mean(np.abs(spec_truth - spec_gen))
            results[nsteps] = dict(spec_gen=spec_gen, gen_cpu=gen_cpu,
                                   std_ratio=std_ratio, band_metrics=band_metrics, spec_l1=spec_l1)

            band_str = '  '.join(f"{b}={v:.4f}" for b, v in band_metrics.items())
            print(f"    RK4 steps={nsteps:3d}:  std_ratio={std_ratio:.4f}  L1={spec_l1:.6f}  {band_str}")
            if cfg.use_wandb:
                log_dict = {f"std_ratio/RK_steps{nsteps}": std_ratio, f"spec_L1/RK_steps{nsteps}": spec_l1}
                for bname, val in band_metrics.items():
                    log_dict[f"spec_relErr_{bname}/RK_steps{nsteps}"] = val
                wandb.log(log_dict, step=self.global_step)

        # ── Spectrum + sample plot ──
        import seaborn as sns
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[ns]['spec_gen'], '--', color=colors[i], label=f'RK4 {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('Power'); ax.legend(fontsize=6); ax.set_title('Power Spectrum')

        # Sample images
        nmax = step_counts[-1]
        truth_imgs = (truth[:4].clamp(-1, 1) + 1) / 2  # back to [0,1]
        gen_imgs = (results[nmax]['gen_cpu'][:4].clamp(-1, 1) + 1) / 2

        for idx in range(min(4, num_eval)):
            ax = axes[1] if idx < 2 else axes[2]
            row = idx % 2
            combined = torch.cat([truth_imgs[idx], gen_imgs[idx]], dim=2)  # (C, H, 2W)
            combined = combined.permute(1, 2, 0).numpy()
            if row == 0:
                ax.imshow(combined)
                ax.set_title(f'Truth | RK4 {nmax}')
                ax.axis('off')

        plt.tight_layout()
        if cfg.use_wandb:
            wandb.log({"spectrum_and_samples": wandb.Image(fig)}, step=self.global_step)
        plt.close()


# ─── Config ─────────────────────────────────────────────────────────────────

class Config:
    def __init__(self):
        self.use_wandb = True
        self.wandb_project = 'interpolants-design'
        self.wandb_entity = 'yifanc96'

        self.noise_type = 'gauss'
        self.hi_size = 128
        self.C = 3
        self.batch_size = 32
        self.data_cache = './celeba_data'

        self.base_lr = 2e-4
        self.max_steps = 50000
        self.t_min_train = 1e-3
        self.t_max_train = 1 - 1e-3
        self.t_min_sample = 1e-3
        self.t_max_sample = 1 - 1e-3
        self.print_loss_every = 50
        self.test_every = 2000
        self.eval_step_counts = [2, 5, 10, 20, 50]

        self.unet_channels = 64
        self.unet_dim_mults = (1, 2, 4, 4)

        self.save_dir = 'results/celeba'


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = Config()
    config.noise_type = args.noise_type
    config.hi_size = args.hi_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.test_every = args.test_every
    config.unet_channels = args.unet_channels
    config.unet_dim_mults = tuple(args.unet_dim_mults)
    config.save_dir = args.save_dir
    config.data_cache = args.data_cache
    config.use_wandb = not args.no_wandb

    os.makedirs(config.save_dir, exist_ok=True)

    logger = Logger(config)
    logger.setup_wandb(config)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
