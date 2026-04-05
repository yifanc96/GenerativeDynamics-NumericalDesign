"""
Wavelet-based multiscale flow matching for Gaussian fields.

Uses Haar wavelet decomposition instead of spatial stride masks:
  x = P_coarse(x) + P_detail(x)
where P_coarse = upsample_nearest ∘ avg_pool_2x2 (constant on 2x2 blocks)
      P_detail = I - P_coarse (zero-mean on 2x2 blocks)

These are orthogonal projections — no frequency aliasing between scales.

Multiscale generation (num_levels levels):
  t in [0,1]:     generate coarsest component
  t in [1,2]:     generate next detail, conditioned on coarsest
  ...
  t in [K-1,K]:   generate finest detail, conditioned on all coarser

The noise at each level has variance matching the conditional variance of
that wavelet band given coarser bands.
"""

import os, sys, math, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import wandb
from time import time as timer
import torch.fft as torch_fft

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet


# ─── Data generation (same as mask version) ──────────────────────────────────

def precompute_matern_amplitude(grid_size, sigma_sq, length_scale, s):
    freq = torch.from_numpy(np.fft.fftfreq(grid_size)).float() * 2 * math.pi * grid_size
    kx, ky = torch.meshgrid(freq, freq)
    k_sq = kx**2 + ky**2
    spectral_density = sigma_sq * (k_sq + length_scale**2) ** (-s)
    spectral_density[0, 0] = 0
    return torch.sqrt(spectral_density)


def sample_matern_batch(amplitude, batch_size, device='cpu'):
    G = amplitude.shape[0]
    amp = amplitude.to(device).unsqueeze(0)
    noise = torch.complex(torch.randn(batch_size, G, G, device=device),
                          torch.randn(batch_size, G, G, device=device))
    return torch_fft.ifftn(amp * noise, dim=(-2, -1), norm='forward').real


def get_fourier_spectrum(data):
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


# ─── Haar wavelet projections ────────────────────────────────────────────────

class HaarProjection:
    """
    Multi-level Haar wavelet projection operators on a G×G grid.

    Level 0 = finest detail (goes from G/2 to G)
    Level K-1 = coarsest (the low-pass at G/2^K)

    x = sum_{k=0}^{K-1} P_k(x)  (orthogonal decomposition)

    P_K-1 = coarsest low-pass (constant on 2^K × 2^K blocks)
    P_k   = detail at level k (constant on 2^(k+1) blocks, zero-mean on 2^(k+2) blocks)
    P_0   = finest detail (zero-mean on 2×2 blocks)
    """
    def __init__(self, grid_size, num_levels, device='cpu'):
        assert grid_size % (2**num_levels) == 0, \
            f"grid_size {grid_size} must be divisible by 2^{num_levels}={2**num_levels}"
        self.grid_size = grid_size
        self.num_levels = num_levels
        self.device = device

    def _lowpass(self, x, factor):
        """Low-pass: avg_pool then nearest upsample. Factor must be power of 2."""
        B = x.shape[0]
        G = self.grid_size
        Gc = G // factor
        # avg pool: reshape to (B, Gc, factor, Gc, factor), mean over block
        coarse = x.view(B, Gc, factor, Gc, factor).mean(dim=(2, 4))  # (B, Gc, Gc)
        # nearest upsample back to G×G: repeat each element factor times in each dim
        return coarse.unsqueeze(2).unsqueeze(4).expand(
            B, Gc, factor, Gc, factor
        ).reshape(B, G, G)

    def project(self, x, level):
        """Project x onto the subspace for the given level.
        level=num_levels-1 is coarsest (low-pass), level=0 is finest detail.
        K = num_levels.

        level 0 (finest):     x - lowpass(2)
        level 1:              lowpass(2) - lowpass(4)
        ...
        level K-2:            lowpass(2^(K-2)) - lowpass(2^(K-1))
        level K-1 (coarsest): lowpass(2^(K-1))

        Sum of all levels = x.
        Coarsest resolution = G / 2^(K-1).
        """
        K = self.num_levels
        if level == K - 1:
            # coarsest low-pass
            return self._lowpass(x, 2**(K - 1))
        elif level == 0:
            # finest detail
            return x - self._lowpass(x, 2)
        else:
            # intermediate detail
            return self._lowpass(x, 2**level) - self._lowpass(x, 2**(level + 1))

    def cumulative_lowpass(self, x, up_to_level):
        """Sum of projections from level up_to_level to num_levels-1 (all coarser).
        = low-pass at factor 2^(up_to_level+1)."""
        if up_to_level >= self.num_levels - 1:
            # everything
            return x
        return self._lowpass(x, 2**(up_to_level + 1))

    def make_noise(self, batch_size, level_variances):
        """Generate noise with independent variance per wavelet level.
        Returns: (B, G, G) noise where each level's component has the given variance."""
        G = self.grid_size
        device = self.device
        noise = torch.zeros(batch_size, G, G, device=device)
        for k in range(self.num_levels):
            # generate iid noise and project to level k's subspace
            z = torch.randn(batch_size, G, G, device=device)
            z_proj = self.project(z, k)
            # scale to desired variance
            # z_proj has some variance per pixel determined by the projection
            # for Haar: each level projects onto a subspace with specific dimension
            proj_var = z_proj.var(dim=0).mean().item()
            if proj_var > 1e-12:
                scale = math.sqrt(level_variances[k] / proj_var)
                noise = noise + scale * z_proj
        return noise


# ─── Conditional variance estimation ────────────────────────────────────────

def estimate_wavelet_variances(data, haar):
    """Estimate per-pixel variance of each wavelet level, and conditional variances.
    For Gaussian fields, projections are independent, so conditional var = marginal var."""
    variances = []
    for k in range(haar.num_levels):
        proj = haar.project(data, k)
        var_k = proj.var(dim=0).mean().item()
        variances.append(var_k)
    return variances


# ─── Network ────────────────────────────────────────────────────────────────

def make_unet(channels, dim_mults, in_ch=1, out_ch=1):
    return Unet(
        num_classes=1, in_channels=in_ch, out_channels=out_ch,
        dim=channels, dim_mults=dim_mults,
        resnet_block_groups=8,
        learned_sinusoidal_cond=True,
        random_fourier_features=False,
        learned_sinusoidal_dim=max(channels, 16),
        attn_dim_head=max(channels, 16),
        attn_heads=4,
        use_classes=False,
    )


class WaveletVelocity(nn.Module):
    """Single UNet for all levels. Time in [0, num_levels] encodes which level.
    Output is projected onto the active level's subspace."""
    def __init__(self, haar, channels=32, dim_mults=(1, 2, 2, 2)):
        super().__init__()
        self.haar = haar
        self.net = make_unet(channels, dim_mults, in_ch=1, out_ch=1)
        n = sum(p.numel() for p in self.parameters())
        print(f"[WaveletVelocity] {n:,} parameters")

    def forward(self, zt, t):
        """Training path: t can be mixed across levels."""
        raw = self.net(zt, t, classes=None)
        # Project output onto the active level for each sample
        B = zt.shape[0]
        out = torch.zeros_like(zt)
        K = self.haar.num_levels
        for k in range(K):
            # level k is active when t in [K-1-k, K-k] (coarsest first)
            # level_idx = K-1-k maps: k=0 → level K-1 (coarsest), k=K-1 → level 0 (finest)
            level_idx = K - 1 - k
            active = (t >= k) & (t <= k + 1)
            if not active.any():
                continue
            idx = active.nonzero(as_tuple=True)[0]
            proj = self.haar.project(raw[idx, 0], level_idx)
            out[idx, 0] = proj
        return out

    def forward_scalar_t(self, zt, t_scalar):
        """Fast path for sampling: all samples at same time."""
        B = zt.shape[0]
        K = self.haar.num_levels
        k = int(min(max(t_scalar, 0), K - 1))
        level_idx = K - 1 - k
        t = torch.full((B,), t_scalar, device=zt.device)
        raw = self.net(zt, t, classes=None)
        proj = self.haar.project(raw[:, 0], level_idx)
        out = torch.zeros_like(zt)
        out[:, 0] = proj
        return out


# ─── Interpolant & Sampler ──────────────────────────────────────────────────

class WaveletInterpolant:
    """
    Interpolant using wavelet decomposition.

    At time t in [k, k+1] (k=0 is coarsest phase):
      - Levels resolved before k: contain data components
      - Level being resolved (K-1-k): interpolating from noise to data
      - Levels not yet resolved: contain noise components

    I_t = sum_{j resolved} data_j + alpha(t)*noise_active + beta(t)*data_active + sum_{j unresolved} noise_j
    where alpha = 1-(t-k), beta = t-k for the active level.
    """
    def __init__(self, haar):
        self.haar = haar
        self.K = haar.num_levels

    def It(self, z0_levels, z1_levels, t):
        """Build interpolated state.
        z0_levels, z1_levels: list of (B, G, G) tensors, one per level.
        t: (B,) times in [0, K].
        Returns: (B, 1, G, G)
        """
        B = t.shape[0]
        K = self.K
        G = self.haar.grid_size
        result = torch.zeros(B, G, G, device=t.device)

        for k in range(K):
            level_idx = K - 1 - k  # coarsest first
            # before this phase: data
            resolved = (t >= k + 1).view(-1, 1, 1)
            # during this phase: interpolating
            active = ((t > k) & (t < k + 1))
            alpha_k = (1 - (t - k)).clamp(0, 1).view(-1, 1, 1)
            beta_k = (t - k).clamp(0, 1).view(-1, 1, 1)
            transitioning = active.view(-1, 1, 1)
            # after this phase: noise
            unresolved = (t <= k).view(-1, 1, 1)

            level_val = (resolved * z1_levels[level_idx]
                         + transitioning * (alpha_k * z0_levels[level_idx] + beta_k * z1_levels[level_idx])
                         + unresolved * z0_levels[level_idx])
            result = result + level_val

        return result.unsqueeze(1)

    def Rt(self, z0_levels, z1_levels, t):
        """Target velocity. Only the active level contributes: -noise_active + data_active."""
        B = t.shape[0]
        K = self.K
        G = self.haar.grid_size
        result = torch.zeros(B, G, G, device=t.device)

        for k in range(K):
            level_idx = K - 1 - k
            active = ((t >= k) & (t <= k + 1)).view(-1, 1, 1).float()
            result = result + active * (-z0_levels[level_idx] + z1_levels[level_idx])

        return result.unsqueeze(1)


class WaveletSampler:
    def __init__(self, num_levels):
        self.num_levels = num_levels

    def _f(self, model, zt, t_scalar):
        return model.forward_scalar_t(zt, t_scalar)

    def _make_tgrid(self, steps, t_min=1e-3, t_max=1 - 1e-3):
        K = self.num_levels
        if K == 1:
            return torch.linspace(t_min, t_max, steps)
        base = steps // K
        remainder = steps % K
        tgrid = []
        for k in range(K):
            n_k = base + (1 if k < remainder else 0)
            if n_k == 0:
                continue
            lo = k + t_min if k == 0 else float(k)
            hi = k + 1 - t_min if k == K - 1 else float(k + 1)
            pts = torch.linspace(lo, hi, n_k + 1)[:-1]
            tgrid.append(pts)
        return torch.cat(tgrid)

    @torch.no_grad()
    def EM(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3):
        tgrid = self._make_tgrid(steps, t_min, t_max).type_as(z0)
        zt = z0
        for i in range(len(tgrid)):
            t_val = tgrid[i]
            dt = tgrid[i + 1] - t_val if i + 1 < len(tgrid) else (self.num_levels * t_max - t_val)
            zt = zt + self._f(model, zt, t_val.item()) * dt
        return zt

    @torch.no_grad()
    def Heun(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3):
        tgrid = self._make_tgrid(steps, t_min, t_max).type_as(z0)
        zt = z0
        for i in range(len(tgrid)):
            t_val = tgrid[i]
            dt = tgrid[i + 1] - t_val if i + 1 < len(tgrid) else (self.num_levels * t_max - t_val)
            tv, dtv = t_val.item(), dt.item()
            k1 = self._f(model, zt, tv)
            k2 = self._f(model, zt + dt * k1, tv + dtv)
            zt = zt + 0.5 * dt * (k1 + k2)
        return zt

    @torch.no_grad()
    def RK4(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3):
        tgrid = self._make_tgrid(steps, t_min, t_max).type_as(z0)
        zt = z0
        for i in range(len(tgrid)):
            t_val = tgrid[i]
            dt = tgrid[i + 1] - t_val if i + 1 < len(tgrid) else (self.num_levels * t_max - t_val)
            tv, dtv = t_val.item(), dt.item()
            k1 = self._f(model, zt, tv)
            k2 = self._f(model, zt + 0.5 * dt * k1, tv + 0.5 * dtv)
            k3 = self._f(model, zt + 0.5 * dt * k2, tv + 0.5 * dtv)
            k4 = self._f(model, zt + dt * k3, tv + dtv)
            zt = zt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return zt


# ─── Trainer ────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = config
        sigma_sq = cfg.sigma_sq * ((2 * math.pi)**2 + cfg.ls**2)**cfg.s
        self.amplitude = precompute_matern_amplitude(cfg.grid_size, sigma_sq, cfg.ls, cfg.s)
        self.test_data = sample_matern_batch(self.amplitude, 500, device='cpu')
        print(f"[Data] Matern grid={cfg.grid_size}, s={cfg.s}, ls={cfg.ls}, test std={self.test_data.std():.4f}")

        # wavelet projection
        self.haar = HaarProjection(cfg.grid_size, cfg.num_levels, device=self.device)

        # estimate per-level variances
        var_data = sample_matern_batch(self.amplitude, 3000, device='cpu')
        self.level_variances = estimate_wavelet_variances(var_data, self.haar)
        print(f"[Wavelet variances] num_levels={cfg.num_levels}")
        for k in range(cfg.num_levels):
            print(f"  level {k} ({'coarsest' if k == cfg.num_levels-1 else 'detail'}): var={self.level_variances[k]:.6f}, std={self.level_variances[k]**0.5:.6f}")
        print(f"  sum={sum(self.level_variances):.4f}, data_var={var_data.var().item():.4f}")
        del var_data

        # interpolant & sampler
        self.interpolant = WaveletInterpolant(self.haar)
        self.sampler = WaveletSampler(cfg.num_levels)

        # model
        self.model = WaveletVelocity(
            self.haar, channels=cfg.unet_channels, dim_mults=cfg.unet_dim_mults
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.max_steps)
        self.time_dist = torch.distributions.Uniform(low=cfg.t_min_train, high=cfg.t_max_train)
        self.global_step = 0

    def decompose(self, x):
        """Decompose (B, G, G) into list of per-level projections."""
        return [self.haar.project(x, k) for k in range(self.haar.num_levels)]

    def make_noise_levels(self, batch_size):
        """Generate noise with correct per-level variance."""
        noise = self.haar.make_noise(batch_size, self.level_variances)
        return self.decompose(noise)

    def train_step(self):
        cfg = self.config
        batch = sample_matern_batch(self.amplitude, cfg.batch_size, device=self.device)

        # decompose data and noise into wavelet levels
        z1_levels = self.decompose(batch)
        z0_levels = self.make_noise_levels(cfg.batch_size)

        t = cfg.num_levels * self.time_dist.sample((cfg.batch_size,)).to(self.device)

        zt = self.interpolant.It(z0_levels, z1_levels, t)
        target = self.interpolant.Rt(z0_levels, z1_levels, t)
        pred = self.model(zt, t)
        loss = (pred - target).pow(2).sum(dim=(1, 2, 3)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e4)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), grad_norm.item()

    def fit(self):
        cfg = self.config
        t0 = timer()
        print(f"[Training] num_levels={cfg.num_levels}, max_steps={cfg.max_steps}, batch={cfg.batch_size}")
        self.test_model()

        while self.global_step < cfg.max_steps:
            loss, gn = self.train_step()
            if self.global_step % cfg.print_loss_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  step {self.global_step}/{cfg.max_steps}  loss={loss:.4f}  grad={gn:.2f}  lr={lr:.2e}  [{(timer()-t0)/60:.1f}m]")
                if cfg.use_wandb:
                    wandb.log({"loss": loss, "grad_norm": gn, "lr": lr}, step=self.global_step)
            if self.global_step > 0 and self.global_step % cfg.test_every == 0:
                self.test_model()
            self.global_step += 1

        print("[Training] Done.")
        self.test_model()
        os.makedirs(cfg.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(cfg.save_dir, 'model_final.pt'))

    @torch.no_grad()
    def test_model(self):
        cfg = self.config
        self.model.eval()

        num_eval = cfg.num_eval
        truth_np = self.test_data[:num_eval].numpy()
        kvals, spec_truth = get_fourier_spectrum(truth_np)

        # generate noise and assemble initial state
        z0_levels = self.make_noise_levels(num_eval)
        z0 = sum(z0_levels).unsqueeze(1)  # (B, 1, G, G)

        step_counts = cfg.eval_step_counts
        results = {}

        for nsteps in step_counts:
            for tag, fn in [('RK4', self.sampler.RK4), ('Heun', self.sampler.Heun), ('EM', self.sampler.EM)]:
                gen = fn(z0.clone(), self.model, steps=nsteps,
                         t_min=cfg.t_min_sample, t_max=cfg.t_max_sample)
                gen_np = gen.squeeze(1).cpu().numpy()
                _, spec_gen = get_fourier_spectrum(gen_np)
                std_ratio = gen_np.std() / (truth_np.std() + 1e-12)
                spec_rel = np.mean(np.abs(spec_truth - spec_gen) / (np.abs(spec_truth) + 1e-12))
                spec_l1 = np.mean(np.abs(spec_truth - spec_gen))
                key = f"{tag}{nsteps}"
                results[key] = dict(spec_gen=spec_gen, gen_np=gen_np,
                                    std_ratio=std_ratio, spec_rel=spec_rel, spec_l1=spec_l1)
                print(f"    {tag:4s} steps={nsteps:3d}:  spec_relErr={spec_rel:.4f}  spec_L1={spec_l1:.6f}  std_ratio={std_ratio:.4f}")
                if cfg.use_wandb:
                    wandb.log({
                        f"spec_relErr/{tag}_steps{nsteps}": spec_rel,
                        f"spec_L1/{tag}_steps{nsteps}": spec_l1,
                        f"std_ratio/{tag}_steps{nsteps}": std_ratio,
                    }, step=self.global_step)

        # spectrum plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'RK4{ns}']['spec_gen'], '--', color=colors[i], label=f'RK4 {ns}')
            ax.loglog(kvals, results[f'EM{ns}']['spec_gen'], ':', color=colors[i], alpha=0.5, label=f'EM {ns}')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)'); ax.legend(fontsize=6); ax.set_title('Energy Spectrum')

        nmin, nmax = step_counts[0], step_counts[-1]
        vmax = max(abs(truth_np[0].min()), abs(truth_np[0].max()))
        combined = np.concatenate([truth_np[0], results[f'RK4{nmin}']['gen_np'][0], results[f'RK4{nmax}']['gen_np'][0]], axis=1)
        axes[1].imshow(combined, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Truth | RK4 {nmin} | RK4 {nmax}'); axes[1].axis('off')
        plt.tight_layout()
        if cfg.use_wandb:
            wandb.log({"spectrum_comparison": wandb.Image(fig)}, step=self.global_step)
        plt.close()


# ─── Config & main ──────────────────────────────────────────────────────────

class Config:
    def __init__(self):
        self.use_wandb = True
        self.wandb_project = 'interpolants-design'
        self.wandb_entity = 'yifanc96'

        self.grid_size = 64
        self.sigma_sq = 1.0
        self.ls = 1.0
        self.s = 3.0
        self.batch_size = 200

        self.num_levels = 2  # number of wavelet levels (1 = standard FM, 2+ = multiscale)

        self.base_lr = 2e-4
        self.max_steps = 5000
        self.t_min_train = 1e-3
        self.t_max_train = 1.0 - 1e-3
        self.t_min_sample = 1e-3
        self.t_max_sample = 1.0 - 1e-3
        self.print_loss_every = 50
        self.test_every = 1000
        self.num_eval = 100
        self.eval_step_counts = [2, 5, 10, 20]

        self.unet_channels = 32
        self.unet_dim_mults = (1, 2, 2, 2)

        self.save_dir = 'results/wavelet_multiscale'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--ls', type=float, default=1.0)
    p.add_argument('--sigma_sq', type=float, default=1.0)
    p.add_argument('--num_levels', type=int, default=2)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test_every', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default='results/wavelet_multiscale')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    config.grid_size = args.grid_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.s = args.s
    config.ls = args.ls
    config.sigma_sq = args.sigma_sq
    config.num_levels = args.num_levels
    config.test_every = args.test_every
    config.save_dir = args.save_dir
    config.use_wandb = not args.no_wandb

    os.makedirs(config.save_dir, exist_ok=True)

    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    run_name = f"GF_wavelet_L{config.num_levels}_s{config.s}_grid{config.grid_size}_{log_base}"

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name)
        for k, v in vars(config).items():
            if isinstance(v, (int, float, bool, str, list, tuple)):
                setattr(wandb.config, k, v)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
