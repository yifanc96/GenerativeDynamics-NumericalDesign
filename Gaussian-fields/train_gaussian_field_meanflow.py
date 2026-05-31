"""
Mean-flow training for 2D Matern Gaussian fields (sanity check).

Both base distributions supported:
  --noise gauss      : z0 ~ N(0,I)
  --noise data_dep   : z0 from empirical covariance of previous batch

This is a sanity check for the mean-flow algorithm: the data is exactly
Gaussian, so the optimal velocity field is linear and the mean-flow loss
should converge cleanly. If the algorithm works on Gaussian data, then
NS difficulties are due to data complexity, not the algorithm.

Mean-flow loss (NS convention, s=0 noise → s=1 data):
  z_s = (1-s)*z0 + s*z1,  v = z1 - z0
  w, dw_ds = JVP(model, (z_s, s, r), (v, 1, 0))
  w_tgt = v + (r - s) * dw_ds
  loss = adaptive_L2(w - stopgrad(w_tgt))
"""

import os, sys, math, datetime
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import wandb
from time import time as timer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb

# ─── Data generation ─────────────────────────────────────────────────────────

def precompute_matern_amplitude(grid_size, sigma_sq, length_scale, s):
    """Precompute spectral amplitude for fast on-the-fly sampling."""
    freq = torch.fft.fftfreq(grid_size) * 2 * math.pi * grid_size
    kx, ky = torch.meshgrid(freq, freq, indexing='ij')
    k_sq = kx**2 + ky**2
    spectral_density = sigma_sq * (k_sq + length_scale**2) ** (-s)
    spectral_density[0, 0] = 0  # zero mean mode
    return torch.sqrt(spectral_density)


def sample_matern_batch(amplitude, batch_size, device='cpu'):
    """Generate a batch of Matérn samples on-the-fly using precomputed amplitude."""
    G = amplitude.shape[0]
    amp = amplitude.to(device).unsqueeze(0)
    noise = torch.complex(torch.randn(batch_size, G, G, device=device),
                          torch.randn(batch_size, G, G, device=device))
    return torch.fft.ifft2(amp * noise, norm='forward').real


def get_fourier_spectrum(data):
    """Compute radially-averaged energy spectrum."""
    from scipy.stats import binned_statistic
    N = data.shape[-1]
    freq = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(freq, freq)
    k_mag = np.sqrt(kx**2 + ky**2).flatten()
    fhat = np.fft.fft2(data, norm='forward')
    power = np.abs(fhat)**2
    power_mean = power.mean(axis=0).flatten()
    k_bins = np.arange(0.5, N // 2 + 1, 1.0)
    Abins, edges, _ = binned_statistic(k_mag, power_mean, statistic='mean', bins=k_bins)
    kvals = 0.5 * (edges[:-1] + edges[1:])
    Abins_w = Abins * np.pi * (edges[1:]**2 - edges[:-1]**2)
    return kvals, Abins_w


# ─── Data-dependent noise ────────────────────────────────────────────────────

def make_data_dependent_noise(next_batch, batch_size):
    """
    z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j},  xi ~ N(0,1)
    next_batch: (N, H, W),  returns: (batch_size, 1, H, W)
    """
    N = next_batch.shape[0]
    centered = next_batch - next_batch.mean(dim=0, keepdim=True)
    xi = torch.randn(N, batch_size, device=next_batch.device)
    noise = torch.einsum('nhw,nb->bhw', centered, xi) / math.sqrt(N)
    return noise.unsqueeze(1)


# ─── Adaptive L2 loss ───────────────────────────────────────────────────────

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """L_adap = stopgrad(w) * ||error||^2,  w = 1/(||error||^2 + c)^(1-gamma)"""
    delta_sq = error.pow(2).mean(dim=(1, 2, 3))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq.detach() + c).pow(p)
    return (w * delta_sq).mean()


# ─── Network ─────────────────────────────────────────────────────────────────

class MeanFlowVelocity(nn.Module):
    """UNet conditioned on both s (current time) and r (target time)."""
    def __init__(self, config):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=1, out_channels=1,
            dim=config.unet_channels, dim_mults=config.unet_dim_mults,
            resnet_block_groups=config.unet_resnet_block_groups,
            learned_sinusoidal_cond=config.unet_learned_sinusoidal_cond,
            random_fourier_features=config.unet_random_fourier_features,
            learned_sinusoidal_dim=config.unet_learned_sinusoidal_dim,
            attn_dim_head=config.unet_attn_dim_head,
            attn_heads=config.unet_attn_heads,
            use_classes=False,
        )
        time_dim = config.unet_channels * 4
        sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(config.unet_learned_sinusoidal_dim, is_random=False)
        fourier_dim = config.unet_learned_sinusoidal_dim + 1
        self.r_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # Zero-init final r_mlp projection (DiT-style)
        nn.init.zeros_(self.r_mlp[-1].weight)
        nn.init.zeros_(self.r_mlp[-1].bias)
        self.time_scale = config.time_scale
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Network] MeanFlowVelocity: {n_params:,} parameters")

    def forward(self, zt, s, r):
        net = self.net
        s_scaled = s * self.time_scale
        r_scaled = r * self.time_scale
        s_emb = net.time_mlp(s_scaled)
        r_emb = self.r_mlp(r_scaled)
        t_emb = s_emb + r_emb

        x = net.init_conv(zt)
        r_skip = x.clone()
        h = []
        for block1, block2, attn, downsample in net.downs:
            x = block1(x, t_emb, None); h.append(x)
            x = block2(x, t_emb, None); x = attn(x); h.append(x)
            x = downsample(x)
        x = net.mid_block1(x, t_emb, None)
        x = net.mid_attn(x)
        x = net.mid_block2(x, t_emb, None)
        for block1, block2, attn, upsample in net.ups:
            x = torch.cat((x, h.pop()), dim=1); x = block1(x, t_emb, None)
            x = torch.cat((x, h.pop()), dim=1); x = block2(x, t_emb, None); x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r_skip), dim=1)
        x = net.final_res_block(x, t_emb, None)
        return net.final_conv(x)


# ─── JVP computation ────────────────────────────────────────────────────────

def compute_jvp(model, z, s, r, v):
    """JVP for total s-derivative along the flow."""
    def fn(z_, s_, r_):
        return model(z_, s_, r_)
    primals = (z, s, r)
    tangents = (v, torch.ones_like(s), torch.zeros_like(r))
    return torch.func.jvp(fn, primals, tangents)


# ─── Sampler ─────────────────────────────────────────────────────────────────

class Sampler:
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def mean_flow_sample(self, z0, model, steps):
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
        self.name = f"GFmeanflow_{config.noise}_s{config.s1}_grid{config.grid_size}_lr{config.base_lr}_{self.log_base}"

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
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
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

    def prepare_data(self):
        cfg = self.config
        sigma_sq = cfg.sigma_sq1 * ((2 * math.pi)**2 + cfg.ls1**2)**cfg.s1
        self.amplitude = precompute_matern_amplitude(cfg.grid_size, sigma_sq, cfg.ls1, cfg.s1)
        print(f"[Data] On-the-fly Matérn sampling, grid={cfg.grid_size}, s={cfg.s1}, ls={cfg.ls1}")
        self.test_data = sample_matern_batch(self.amplitude, 500, device='cpu')
        print(f"[Data] test set: {self.test_data.shape}, std={self.test_data.std():.4f}")

    def get_noise(self, batch_data):
        """Returns z0 of shape (B, 1, H, W). batch_data: (B, H, W)."""
        cfg = self.config
        B = batch_data.shape[0]
        if cfg.noise == 'gauss':
            return torch.randn(B, 1, cfg.grid_size, cfg.grid_size, device=self.device)
        elif cfg.noise == 'data_dep':
            if self.prev_batch is None:
                z0 = torch.randn(B, 1, cfg.grid_size, cfg.grid_size, device=self.device)
            else:
                z0 = make_data_dependent_noise(self.prev_batch, B)
            self.prev_batch = batch_data.clone()
            return z0
        else:
            raise ValueError(f"Unknown noise type: {cfg.noise}")

    def sample_s_r(self, batch_size):
        cfg = self.config
        if cfg.time_dist == 'lognorm':
            t1 = torch.sigmoid(torch.randn(batch_size, device=self.device) * cfg.time_sigma + cfg.time_mu)
            t2 = torch.sigmoid(torch.randn(batch_size, device=self.device) * cfg.time_sigma + cfg.time_mu)
        else:
            t1 = torch.rand(batch_size, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min
            t2 = torch.rand(batch_size, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min
        s = torch.minimum(t1, t2)
        r = torch.maximum(t1, t2)
        flow_mask = torch.rand(batch_size, device=self.device) < cfg.flow_ratio
        r = torch.where(flow_mask, s, r)
        return s, r

    def loss_function(self, z0, z1, s, r):
        sw = s[:, None, None, None]
        z_s = (1.0 - sw) * z0 + sw * z1
        v = z1 - z0
        w, dw_ds = compute_jvp(self.model, z_s, s, r, v)
        dr = (r - s)[:, None, None, None]
        w_tgt = v + dr * dw_ds
        error = w - w_tgt.detach()
        if self.config.loss_type == 'adaptive':
            return adaptive_l2_loss(error, gamma=self.config.adaptive_gamma)
        else:
            return error.pow(2).sum(dim=(1, 2, 3)).mean()

    def fit(self):
        cfg = self.config
        t_start = timer()
        print(f"[Training] max_steps={cfg.max_steps}, batch={cfg.batch_size}, lr={cfg.base_lr}, noise={cfg.noise}")
        print(f"[Training] flow_ratio={cfg.flow_ratio}, t_min={cfg.t_min}, t_max={cfg.t_max}")
        self.test_model()

        while self.global_step < cfg.max_steps:
            batch_data = sample_matern_batch(self.amplitude, cfg.batch_size, device=self.device)
            z0 = self.get_noise(batch_data)
            z1 = batch_data.unsqueeze(1)
            s, r = self.sample_s_r(cfg.batch_size)

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
        # Use EMA weights for evaluation
        eval_model = self.ema.ema_model
        eval_model.eval()

        num_eval = self.test_data.shape[0]
        truth_np = self.test_data.numpy()
        kvals, spec_truth = get_fourier_spectrum(truth_np)

        # Build z0 according to noise type
        if cfg.noise == 'gauss':
            z0 = torch.randn(num_eval, 1, cfg.grid_size, cfg.grid_size, device=self.device)
        else:
            noise_src = sample_matern_batch(self.amplitude, cfg.batch_size, device=self.device)
            z0 = make_data_dependent_noise(noise_src, num_eval)

        step_counts = cfg.eval_step_counts
        results = {}

        for nsteps in step_counts:
            gen = self.sampler.mean_flow_sample(z0.clone(), eval_model, nsteps)
            gen_np = gen.squeeze(1).cpu().numpy()
            _, spec_gen = get_fourier_spectrum(gen_np)

            std_ratio = gen_np.std() / (truth_np.std() + 1e-12)
            spec_rel = np.mean(np.abs(spec_truth - spec_gen) / (np.abs(spec_truth) + 1e-12))
            spec_l1 = np.mean(np.abs(spec_truth - spec_gen))

            tag = f"MF{nsteps}"
            results[tag] = dict(spec_gen=spec_gen, gen_np=gen_np,
                                std_ratio=std_ratio, spec_rel=spec_rel, spec_l1=spec_l1)
            print(f"    MF steps={nsteps:3d}:  spec_relErr={spec_rel:.4f}  spec_L1={spec_l1:.6f}  std_ratio={std_ratio:.4f}")
            if cfg.use_wandb:
                wandb.log({
                    f"spec_relErr/MF_steps{nsteps}": spec_rel,
                    f"spec_L1/MF_steps{nsteps}": spec_l1,
                    f"std_ratio/MF_steps{nsteps}": std_ratio,
                }, step=self.global_step)

        # ── Spectrum plot ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'MF{ns}']['spec_gen'], '--', color=colors[i], label=f'MF {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)'); ax.legend(fontsize=6); ax.set_title('Energy Spectrum')

        nmin, nmax = step_counts[0], step_counts[-1]
        vmax = max(abs(truth_np[0].min()), abs(truth_np[0].max()))
        combined = np.concatenate([truth_np[0], results[f'MF{nmin}']['gen_np'][0], results[f'MF{nmax}']['gen_np'][0]], axis=1)
        axes[1].imshow(combined, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Truth | MF {nmin}-step | MF {nmax}-step'); axes[1].axis('off')
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

        self.grid_size = 64
        self.sigma_sq1 = 1.0
        self.ls1 = 1.0
        self.s1 = 3.0
        self.batch_size = 200
        self.noise = 'gauss'   # 'gauss' or 'data_dep'

        self.base_lr = 2e-4
        self.max_steps = 50000      # mean flow needs MUCH longer training (toy: 100k for tiny model)
        self.t_min = 0.0
        self.t_max = 1.0
        self.print_loss_every = 100
        self.test_every = 5000
        self.eval_step_counts = [1, 2, 4, 8, 16]

        # mean-flow specific (validated by toy ablation)
        self.flow_ratio = 0.5       # reference default; toy ablation shows it works
        self.adaptive_gamma = 0.5
        self.loss_type = 'adaptive'
        self.time_dist = 'uniform'  # uniform; lognormal hurt in toy ablation
        self.time_mu = -0.4
        self.time_sigma = 1.0
        self.time_scale = 1.0       # NO scaling — UNet's learned sinusoidal handles t∈[0,1]
        self.grad_clip = 1.0        # tight grad clip for stability
        self.ema_decay = 0.9999     # standard EMA decay

        # architecture (medium)
        self.unet_channels = 32
        self.unet_dim_mults = (1, 2, 2, 2)
        self.unet_resnet_block_groups = 8
        self.unet_learned_sinusoidal_dim = 32
        self.unet_attn_dim_head = 32
        self.unet_attn_heads = 4
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False

        self.save_dir = 'results/gaussian_field_meanflow'


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--s1', type=float, default=3.0)
    p.add_argument('--ls1', type=float, default=1.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test_every', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default='results/gaussian_field_meanflow')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--noise', type=str, default='gauss', choices=['gauss', 'data_dep'])
    p.add_argument('--t_min', type=float, default=0.0)
    p.add_argument('--t_max', type=float, default=1.0)
    p.add_argument('--flow_ratio', type=float, default=0.75)
    p.add_argument('--adaptive_gamma', type=float, default=0.5)
    p.add_argument('--loss_type', type=str, default='adaptive', choices=['adaptive', 'mse'])
    p.add_argument('--time_dist', type=str, default='uniform', choices=['uniform', 'lognorm'])
    p.add_argument('--time_scale', type=float, default=1.0)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--unet_channels', type=int, default=32)
    p.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 2, 2])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    config.grid_size = args.grid_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.s1 = args.s1
    config.ls1 = args.ls1
    config.test_every = args.test_every
    config.save_dir = args.save_dir
    config.use_wandb = not args.no_wandb
    config.noise = args.noise
    config.t_min = args.t_min
    config.t_max = args.t_max
    config.flow_ratio = args.flow_ratio
    config.adaptive_gamma = args.adaptive_gamma
    config.loss_type = args.loss_type
    config.time_dist = args.time_dist
    config.time_scale = args.time_scale
    config.grad_clip = args.grad_clip
    config.ema_decay = args.ema_decay
    config.unet_channels = args.unet_channels
    config.unet_dim_mults = tuple(args.unet_dim_mults)

    os.makedirs(config.save_dir, exist_ok=True)

    logger = Logger(config)
    logger.setup_wandb(config)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
