"""
Mean-flow training for 2D Navier-Stokes with data-dependent noise.

Same mean-flow algorithm as the Gaussian-base version, but noise z0 is
constructed from the empirical covariance of the data:
  z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j},  xi ~ N(0,1)

Convention (same as existing NS scripts):
  s=0: noise,  s=1: data
  z_s = (1-s)*z0 + s*z1,  z0=noise, z1=data
  v = z1 - z0

Mean-flow loss (adapted from arXiv:2505.13447):
  w, dw_ds = JVP(model, (z,s,r), (v,1,0))
  w_tgt = v + (r-s) * dw_ds
  loss = adaptive_L2(w - stopgrad(w_tgt))
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
from functools import partial

from unet import Unet, RandomOrLearnedSinusoidalPosEmb

# ─── Data loading ────────────────────────────────────────────────────────────

def load_ns_data(data_loc, hi_size, batch_size, train_test_split):
    """
    Load NS vorticity data. File contains (data, time) tuple.
    data: (num_traj, num_snapshots, Nx, Ny)
    For unconditional generation, flatten trajectories into individual snapshots.
    """
    data_raw, time_raw = torch.load(data_loc)
    Ntj, Nts, Nx, Ny = data_raw.shape
    print(f"[Data] raw: {Ntj} trajectories x {Nts} snapshots x {Nx}x{Ny}")

    avg_pixel_norm = torch.norm(data_raw, dim=(2, 3), p='fro').mean() / np.sqrt(Nx * Ny)
    print(f"[Data] avg pixel norm (original): {avg_pixel_norm:.4f}")
    data_raw = data_raw / avg_pixel_norm

    # Flatten to (Ntj*Nts, Nx, Ny) and downsample
    data = data_raw.reshape(-1, Nx, Ny)
    if hi_size != Nx:
        data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
    data = data[:, None, :, :]  # (N, 1, H, W)
    print(f"[Data] flattened & resized: {data.shape}, std={data.std():.4f}")

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
    centered = next_batch - next_batch.mean(dim=0, keepdim=True)
    xi = torch.randn(N, batch_size, device=next_batch.device)
    noise = torch.einsum('nchw,nb->bchw', centered, xi) / math.sqrt(N)
    return noise


# ─── Adaptive L2 loss ───────────────────────────────────────────────────────

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    L_adap = stopgrad(w) * ||error||^2,  w = 1/(||error||^2 + c)^(1-gamma)
    Weights errors inversely by magnitude for better training dynamics.
    """
    delta_sq = error.pow(2).mean(dim=(1, 2, 3))  # (B,)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq.detach() + c).pow(p)
    return (w * delta_sq).mean()


# ─── Network ─────────────────────────────────────────────────────────────────

class MeanFlowVelocity(nn.Module):
    """
    UNet conditioned on both current time s and target time r.
    s-embedding uses the UNet's built-in time_mlp.
    r-embedding uses a separate learned sinusoidal MLP (same architecture).
    Combined via addition: emb = time_mlp(s) + r_mlp(r).
    """
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
        # r-time embedding (mirrors the UNet's time_mlp)
        time_dim = config.unet_channels * 4
        sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(config.unet_learned_sinusoidal_dim, is_random=False)
        fourier_dim = config.unet_learned_sinusoidal_dim + 1
        self.r_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # Zero-init final r_mlp projection so r conditioning starts as no-op
        # (matches DiT zero-init pattern for the conditioning side branch).
        nn.init.zeros_(self.r_mlp[-1].weight)
        nn.init.zeros_(self.r_mlp[-1].bias)
        # Time-scale factor: matches reference DiT (t * 1000) so sinusoidal
        # embeddings can resolve close (s, r) pairs.
        self.time_scale = config.time_scale
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Network] MeanFlowVelocity: {n_params:,} parameters")

    def forward(self, zt, s, r):
        """
        zt: (B, C, H, W)
        s:  (B,)  current time
        r:  (B,)  target time
        """
        net = self.net
        s_scaled = s * self.time_scale
        r_scaled = r * self.time_scale
        s_emb = net.time_mlp(s_scaled)   # (B, time_dim)
        r_emb = self.r_mlp(r_scaled)     # (B, time_dim)
        t_emb = s_emb + r_emb            # combined conditioning

        # ── UNet forward with combined embedding ──
        x = net.init_conv(zt)
        r_skip = x.clone()
        h = []

        for block1, block2, attn, downsample in net.downs:
            x = block1(x, t_emb, None)
            h.append(x)
            x = block2(x, t_emb, None)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = net.mid_block1(x, t_emb, None)
        x = net.mid_attn(x)
        x = net.mid_block2(x, t_emb, None)

        for block1, block2, attn, upsample in net.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_emb, None)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_emb, None)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r_skip), dim=1)
        x = net.final_res_block(x, t_emb, None)
        return net.final_conv(x)


# ─── JVP computation ────────────────────────────────────────────────────────

def compute_jvp(model, z, s, r, v):
    """
    Compute w = model(z, s, r) and its total s-derivative along the flow:
      dw/ds_total = (∂w/∂z)·v + ∂w/∂s
    using forward-mode AD (torch.func.jvp).

    Returns: (w, dw_ds_total), both shape (B, C, H, W).
    """
    def fn(z_, s_, r_):
        return model(z_, s_, r_)

    primals = (z, s, r)
    tangents = (v, torch.ones_like(s), torch.zeros_like(r))
    w, dw_ds = torch.func.jvp(fn, primals, tangents)
    return w, dw_ds


# ─── Sampler ─────────────────────────────────────────────────────────────────

class Sampler:
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def mean_flow_sample(self, z0, model, steps):
        """
        K-step mean-flow sampling: s goes from 0 (noise) to 1 (data).
        z_{k+1} = z_k + (r - s) * w(z_k, s_k, r_k)
        """
        s_vals = torch.linspace(self.config.t_min, self.config.t_max, steps + 1).type_as(z0)
        zt = z0
        for k in range(steps):
            s_k = s_vals[k]
            r_k = s_vals[k + 1]
            s_batch = torch.full((zt.shape[0],), s_k, device=zt.device)
            r_batch = torch.full((zt.shape[0],), r_k, device=zt.device)
            w = model(zt, s_batch, r_batch)
            zt = zt + (r_k - s_k) * w
        return zt


# ─── Logger ──────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, config):
        date = str(datetime.datetime.now())
        self.log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.name = f"NS_meanflow_datanoise_hi{config.hi_size}_lr{config.base_lr}_{self.log_base}"

    def setup_wandb(self, config):
        if config.use_wandb:
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=self.name)
            for key, val in vars(config).items():
                if isinstance(val, (int, float, bool, str, list, tuple)):
                    setattr(wandb.config, key, val)
            print("[wandb] setup done")


# ─── EMA helper ──────────────────────────────────────────────────────────────

import copy

class EMA:
    """Exponential moving average of model weights for stable evaluation."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # also update buffers (BN, etc.)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_data()
        self.model = MeanFlowVelocity(config).to(self.device)
        self.ema = EMA(self.model, decay=config.ema_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_steps)
        self.sampler = Sampler(config)
        self.global_step = 0
        self.prev_batch = None
        self.best_metric = float('inf')
        self.best_step = 0

    def prepare_data(self):
        cfg = self.config
        self.train_loader, self.test_loader, self.train_data, self.test_data, self.avg_pixel_norm = \
            load_ns_data(cfg.data_loc, cfg.hi_size, cfg.batch_size, cfg.train_test_split)

    def get_noise(self, batch_data):
        """Data-dependent noise from previous batch's empirical covariance."""
        B = batch_data.shape[0]
        if self.prev_batch is None:
            z0 = torch.randn(B, 1, self.config.hi_size, self.config.hi_size, device=self.device)
        else:
            z0 = make_data_dependent_noise(self.prev_batch, B)
        self.prev_batch = batch_data.clone()
        return z0

    def sample_s_r(self, batch_size):
        """
        Sample (s, r) pairs. s < r for mean-flow; s == r for flow-matching.
        flow_ratio fraction of the batch uses s == r (standard flow matching).
        """
        cfg = self.config

        if cfg.time_dist == 'uniform':
            t1 = torch.rand(batch_size, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min
            t2 = torch.rand(batch_size, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min
        elif cfg.time_dist == 'lognorm':
            # sigmoid(N(mu, sigma)) — concentrates near 0 and 1 (matches reference)
            t1 = torch.sigmoid(torch.randn(batch_size, device=self.device) * cfg.time_sigma + cfg.time_mu)
            t2 = torch.sigmoid(torch.randn(batch_size, device=self.device) * cfg.time_sigma + cfg.time_mu)
        else:
            raise ValueError(f"Unknown time_dist: {cfg.time_dist}")

        s = torch.minimum(t1, t2)  # earlier time (closer to noise)
        r = torch.maximum(t1, t2)  # later time (closer to data)

        # flow_ratio fraction: set r = s (standard flow matching, no JVP needed)
        flow_mask = torch.rand(batch_size, device=self.device) < cfg.flow_ratio
        r = torch.where(flow_mask, s, r)

        return s, r

    def loss_function(self, z0, z1, s, r):
        """
        Mean-flow loss with JVP.
        z_s = (1-s)*z0 + s*z1,  v = z1 - z0
        w, dw_ds = JVP(model, (z_s, s, r), (v, 1, 0))
        w_tgt = v + (r - s) * dw_ds
        loss = adaptive_L2(w - stopgrad(w_tgt))
        """
        sw = s[:, None, None, None]
        z_s = (1.0 - sw) * z0 + sw * z1
        v = z1 - z0  # target velocity

        # Identify which samples need JVP (r != s)
        needs_jvp = (r - s).abs() > 1e-8

        if needs_jvp.any():
            # Full JVP computation
            w, dw_ds = compute_jvp(self.model, z_s, s, r, v)
            dr = (r - s)[:, None, None, None]
            w_tgt = v + dr * dw_ds
        else:
            # Pure flow-matching mode (all r == s)
            w = self.model(z_s, s, r)
            w_tgt = v

        error = w - w_tgt.detach()
        return adaptive_l2_loss(error, gamma=self.config.adaptive_gamma)

    def fit(self):
        cfg = self.config
        t_start = timer()
        print(f"[Training] max_steps={cfg.max_steps}, batch={cfg.batch_size}, lr={cfg.base_lr}")
        print(f"[Training] flow_ratio={cfg.flow_ratio}, time_dist={cfg.time_dist}, adaptive_gamma={cfg.adaptive_gamma}")
        self.test_model()

        while self.global_step < cfg.max_steps:
            for (batch_data,) in self.train_loader:
                if self.global_step >= cfg.max_steps:
                    break

                batch_data = batch_data.to(self.device)
                z0 = self.get_noise(batch_data)
                z1 = batch_data
                s, r = self.sample_s_r(z1.shape[0])

                self.model.train()
                loss = self.loss_function(z0, z1, s, r)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update(self.model)

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
        # Use EMA weights for evaluation (mean-flow stability trick)
        eval_model = self.ema.ema_model
        eval_model.eval()

        num_eval = min(200, self.test_data.shape[0])
        truth = self.test_data[:num_eval]
        truth_sq = truth.squeeze(1)  # (N, H, W)
        kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)

        # Data-dependent noise from training data
        perm = torch.randperm(self.train_data.shape[0])
        noise_src = self.train_data[perm[:cfg.batch_size]].to(self.device)
        z0 = make_data_dependent_noise(noise_src, num_eval)

        step_counts = cfg.eval_step_counts
        results = {}

        for nsteps in step_counts:
            gen = self.sampler.mean_flow_sample(z0.clone(), eval_model, nsteps)
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

            enst_rel = np.mean(np.abs(enst_truth - enst_gen) / (np.abs(enst_truth) + 1e-12))
            ener_rel = np.mean(np.abs(ener_truth - ener_gen) / (np.abs(ener_truth) + 1e-12))

            tag = f"MF{nsteps}"
            results[tag] = dict(enst_gen=enst_gen, ener_gen=ener_gen, gen_sq=gen_sq,
                                std_ratio=std_ratio, enst_rel=enst_rel, ener_rel=ener_rel,
                                band_metrics=band_metrics)

            band_str = '  '.join(f"{b}: enst={v[0]:.4f} ener={v[1]:.4f}" for b, v in band_metrics.items())
            print(f"    MF steps={nsteps:3d}:  std_ratio={std_ratio:.4f}  {band_str}")
            if cfg.use_wandb:
                log_dict = {
                    f"enstrophy_relErr/MF_steps{nsteps}": enst_rel,
                    f"energy_relErr/MF_steps{nsteps}": ener_rel,
                    f"std_ratio/MF_steps{nsteps}": std_ratio,
                }
                for bname, (enst_err, ener_err) in band_metrics.items():
                    log_dict[f"enstrophy_relErr_{bname}/MF_steps{nsteps}"] = enst_err
                    log_dict[f"energy_relErr_{bname}/MF_steps{nsteps}"] = ener_err
                wandb.log(log_dict, step=self.global_step)

        # ── Spectrum + sample plot ──
        import seaborn as sns
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Enstrophy spectrum
        ax = axes[0]
        ax.loglog(kvals, enst_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'MF{ns}']['enst_gen'], '--', color=colors[i], label=f'MF {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('Enstrophy'); ax.legend(fontsize=6); ax.set_title('Enstrophy Spectrum')

        # Energy spectrum
        ax = axes[1]
        ax.loglog(kvals, ener_truth, 'k-', lw=2, label='truth')
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'MF{ns}']['ener_gen'], '--', color=colors[i], label=f'MF {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('Energy'); ax.legend(fontsize=6); ax.set_title('Energy Spectrum')

        # Sample visualization
        nmax = step_counts[-1]
        combined = torch.cat([truth_sq[0], results[f'MF{step_counts[0]}']['gen_sq'][0],
                              results[f'MF{nmax}']['gen_sq'][0]], dim=1).numpy()
        axes[2].imshow(combined, cmap=sns.cm.icefire, vmin=-2, vmax=2)
        axes[2].set_title(f'Truth | MF {step_counts[0]}-step | MF {nmax}-step'); axes[2].axis('off')

        plt.tight_layout()
        if cfg.use_wandb:
            wandb.log({"spectrum_comparison": wandb.Image(fig)}, step=self.global_step)
        plt.close()

        # ── Track best EMA snapshot by 4-step mid-band enstrophy error ──
        # Use mid-band as the primary metric (most reliable for NS).
        if 4 in step_counts:
            primary = results['MF4']['band_metrics']['mid'][0]  # mid enst err
        else:
            primary = results[f'MF{step_counts[0]}']['enst_rel']
        if primary < self.best_metric:
            self.best_metric = primary
            self.best_step = self.global_step
            best_path = os.path.join(cfg.save_dir, 'model_best.pt')
            torch.save(eval_model.state_dict(), best_path)
            print(f"    [BEST] step={self.global_step}  metric={primary:.4f}  -> {best_path}")
            if cfg.use_wandb:
                wandb.log({"best/metric": primary, "best/step": self.global_step}, step=self.global_step)


# ─── Config ──────────────────────────────────────────────────────────────────

class Config:
    def __init__(self):
        self.use_wandb = True
        self.wandb_project = 'interpolants-design'
        self.wandb_entity = 'yifanc96'

        # data
        self.data_loc = '../NSdata/data_file.pt'
        self.hi_size = 128
        self.C = 1
        self.batch_size = 100
        self.train_test_split = 0.9

        # training
        self.base_lr = 2e-4
        self.max_steps = 200000   # mean flow needs much more training than pure FM
        self.cosine_scheduler = True
        self.print_loss_every = 100
        self.test_every = 5000
        self.eval_step_counts = [1, 2, 4, 8, 16]

        # mean-flow specific (matches reference defaults)
        self.t_min = 0.0
        self.t_max = 1.0
        self.flow_ratio = 0.5       # fraction of batch using r=s (reference default)
        self.grad_clip = 1.0        # gradient clipping norm
        self.ema_decay = 0.9999     # EMA decay for evaluation weights
        self.time_dist = 'uniform'  # uniform; lognormal hurt in toy ablation
        self.time_mu = -0.4
        self.time_sigma = 1.0
        self.time_scale = 1.0       # no scaling — UNet's learned sinusoidal handles t∈[0,1]
        self.adaptive_gamma = 0.5

        # architecture (medium, matching existing)
        self.unet_channels = 32
        self.unet_dim_mults = (1, 2, 2, 2)
        self.unet_resnet_block_groups = 8
        self.unet_learned_sinusoidal_dim = 32
        self.unet_attn_dim_head = 32
        self.unet_attn_heads = 4
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False

        self.save_dir = 'results/ns_meanflow_data_dep_noise'


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--max_steps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test_every', type=int, default=5000)
    p.add_argument('--data_loc', type=str, default='../NSdata/data_file.pt')
    p.add_argument('--save_dir', type=str, default='results/ns_meanflow_data_dep_noise')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--t_min', type=float, default=0.0)
    p.add_argument('--t_max', type=float, default=1.0)
    p.add_argument('--flow_ratio', type=float, default=0.5)
    p.add_argument('--time_scale', type=float, default=1.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--time_dist', type=str, default='uniform', choices=['uniform', 'lognorm'])
    p.add_argument('--adaptive_gamma', type=float, default=0.5)
    p.add_argument('--unet_channels', type=int, default=32)
    p.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 2, 2])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    config.hi_size = args.hi_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.test_every = args.test_every
    config.data_loc = args.data_loc
    config.save_dir = args.save_dir
    config.use_wandb = not args.no_wandb
    config.t_min = args.t_min
    config.t_max = args.t_max
    config.flow_ratio = args.flow_ratio
    config.time_scale = args.time_scale
    config.grad_clip = args.grad_clip
    config.ema_decay = args.ema_decay
    config.time_dist = args.time_dist
    config.adaptive_gamma = args.adaptive_gamma
    config.unet_channels = args.unet_channels
    config.unet_dim_mults = tuple(args.unet_dim_mults)

    os.makedirs(config.save_dir, exist_ok=True)

    logger = Logger(config)
    logger.setup_wandb(config)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
