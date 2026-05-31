"""
Multiscale interpolation for Gaussian fields.

Idea: interpolate coarse pixels first, then fine pixels. Conditional on coarse
pixels, the fine-scale conditional variance is smaller, so we use it as the
noise variance at that level. This makes the learned drift well-conditioned.

Two network modes:
  - "single": one UNet handles all scales (time encodes which scale)
  - "multi":  one smaller UNet per scale, each receives 2 channels:
              channel 0 = zt (full state), channel 1 = resolved coarse pixels
"""

import os, sys, math, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import wandb
from time import time as timer
import torch.fft as torch_fft

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet


# ─── Data generation ─────────────────────────────────────────────────────────

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


# ─── Hierarchical masks ─────────────────────────────────────────────────────

class HierarchicalMasks:
    """
    Decompose a 2^n x 2^n grid into num_masks scales.
    masks[0] = finest (new points at stride 1 not in stride 2)
    masks[-1] = coarsest (points at stride 2^(n-1))

    Transition order: coarsest first.
      t in [0,1]: coarsest scale (masks[-1]) transitions alpha 1->0, beta 0->1
      t in [1,2]: next scale transitions
      ...
      t in [num_masks-1, num_masks]: finest scale transitions
    """
    def __init__(self, grid_size, num_masks, device='cpu'):
        n = int(math.log2(grid_size))
        assert 2**n == grid_size
        self.num_masks = min(num_masks, n)
        self.size = grid_size
        self.device = device

        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))

        all_masks = []
        for k in range(n):
            if k == n - 1:
                mask = (y % (2**k) == 0) & (x % (2**k) == 0)
            else:
                div_cur = (y % (2**k) == 0) & (x % (2**k) == 0)
                div_coarser = (y % (2**(k+1)) == 0) & (x % (2**(k+1)) == 0)
                mask = div_cur & ~div_coarser
            all_masks.append(mask.float().to(device))

        self.masks = []
        if self.num_masks < n:
            for k in range(self.num_masks - 1):
                self.masks.append(all_masks[k])
            agg = sum(all_masks[self.num_masks - 1:])
            self.masks.append(agg.clamp(max=1.0))
        else:
            self.masks = all_masks

        # precompute cumulative coarse masks for each scale k (pre-expanded to 4D):
        # coarse_masks_4d[k] = (1, 1, H, W) union of all scales resolved before k
        self.coarse_masks_4d = []
        for k in range(self.num_masks):
            if k == 0:
                cm = torch.zeros(1, 1, grid_size, grid_size, device=device)
            else:
                resolved_indices = [self.num_masks - 1 - j for j in range(k)]
                cm = sum(self.masks[idx] for idx in resolved_indices).clamp(max=1.0)
                cm = cm.unsqueeze(0).unsqueeze(0)
            self.coarse_masks_4d.append(cm)

        # precompute per-scale beta_dot masks (1, 1, H, W)
        self.beta_dot_masks = []
        for k in range(self.num_masks):
            scale_idx = self.num_masks - 1 - k
            self.beta_dot_masks.append(self.masks[scale_idx].unsqueeze(0).unsqueeze(0))

    def alpha(self, t):
        """Noise coefficient: starts at 1, transitions to 0 scale-by-scale (coarse first)."""
        B = t.shape[0]
        t = t.clamp(0, self.num_masks)
        result = torch.zeros(B, self.size, self.size, device=self.device)
        for k in range(self.num_masks):
            scale_idx = self.num_masks - 1 - k
            m = self.masks[scale_idx].unsqueeze(0)
            active = (t <= k).view(-1, 1, 1)
            trans = ((t > k) & (t < k + 1)).view(-1, 1, 1)
            val = (1 - (t - k)).view(-1, 1, 1)
            result = result + m * active + m * trans * val
        return result

    def beta(self, t):
        """Data coefficient: starts at 0, transitions to 1 scale-by-scale (coarse first)."""
        B = t.shape[0]
        t = t.clamp(0, self.num_masks)
        result = torch.zeros(B, self.size, self.size, device=self.device)
        for k in range(self.num_masks):
            scale_idx = self.num_masks - 1 - k
            m = self.masks[scale_idx].unsqueeze(0)
            active = (t >= k + 1).view(-1, 1, 1)
            trans = ((t > k) & (t < k + 1)).view(-1, 1, 1)
            val = (t - k).view(-1, 1, 1)
            result = result + m * active + m * trans * val
        return result

    def alpha_dot(self, t):
        B = t.shape[0]
        t = t.clamp(0, self.num_masks)
        result = torch.zeros(B, self.size, self.size, device=self.device)
        for k in range(self.num_masks):
            scale_idx = self.num_masks - 1 - k
            m = self.masks[scale_idx].unsqueeze(0)
            in_range = ((t >= k) & (t <= k + 1)).view(-1, 1, 1)
            result = result - m * in_range
        return result

    def beta_dot(self, t):
        B = t.shape[0]
        t = t.clamp(0, self.num_masks)
        result = torch.zeros(B, self.size, self.size, device=self.device)
        for k in range(self.num_masks):
            scale_idx = self.num_masks - 1 - k
            m = self.masks[scale_idx].unsqueeze(0)
            in_range = ((t >= k) & (t <= k + 1)).view(-1, 1, 1)
            result = result + m * in_range
        return result

    def get_active_scale(self, t_scalar):
        """For a scalar time value, return which scale index k is active (t in [k, k+1])."""
        return int(min(max(t_scalar, 0), self.num_masks - 1))

    def get_coarse_context(self, zt, t):
        """
        For each sample, extract the resolved coarse pixels from zt.
        At time t in [k, k+1], the coarse context = zt * (union of masks for scales resolved before k).
        Returns: (B, 1, H, W) tensor with resolved coarse values, zero elsewhere.
        """
        B = zt.shape[0]
        result = torch.zeros_like(zt)
        for k in range(self.num_masks):
            in_scale = ((t >= k) & (t <= k + 1))
            if not in_scale.any():
                continue
            idx = in_scale.nonzero(as_tuple=True)[0]
            result[idx] = zt[idx] * self.coarse_masks_4d[k]
        return result

    def get_coarse_context_scalar_t(self, zt, t_scalar):
        """Fast path: all samples have the same t value."""
        k = self.get_active_scale(t_scalar)
        return zt * self.coarse_masks_4d[k]

    def beta_dot_scalar_t(self, t_scalar, B):
        """Fast path: return (B,1,H,W) beta_dot for scalar t."""
        k = self.get_active_scale(t_scalar)
        return self.beta_dot_masks[k].expand(B, -1, -1, -1)


# ─── Conditional variance estimation ────────────────────────────────────────

def estimate_conditional_variances(data, masks):
    """
    For each scale k, estimate Var(data_k | coarser scales) using ridge regression.
    data: (N, H, W), masks: list of (H, W) tensors from finest to coarsest.
    Returns: list of scalar variances, one per scale.
    """
    n_scales = len(masks)
    N = data.shape[0]
    variances = []

    for k in range(n_scales):
        cur_idx = torch.nonzero(masks[k], as_tuple=True)
        Y = torch.stack([data[b][cur_idx] for b in range(N)])  # (N, n_cur)

        if k == n_scales - 1:
            variances.append(torch.var(Y, dim=0).mean().item())
            continue

        coarse_mask = sum(masks[j] for j in range(k + 1, n_scales)).clamp(max=1)
        coarse_idx = torch.nonzero(coarse_mask, as_tuple=True)
        X = torch.stack([data[b][coarse_idx] for b in range(N)])  # (N, n_coarse)

        X_aug = torch.cat([torch.ones(N, 1), X], dim=1)
        XtX = X_aug.T @ X_aug + 1e-6 * torch.eye(X_aug.shape[1])
        XtY = X_aug.T @ Y
        beta = torch.linalg.solve(XtX, XtY)
        residual = Y - X_aug @ beta
        variances.append(torch.var(residual, dim=0).mean().item())

    return variances


def build_noise_std(masks, cond_variances, device='cpu'):
    """Build per-pixel noise std from conditional variances."""
    H = masks[0].shape[0]
    noise_var = torch.zeros(H, H, device=device)
    for i, m in enumerate(masks):
        noise_var += cond_variances[i] * m.to(device)
    return noise_var.sqrt()


# ─── Networks ───────────────────────────────────────────────────────────────

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


class SingleVelocity(nn.Module):
    """One UNet for all scales. Time in [0, num_masks] encodes scale.
    Input is just zt (1 channel) — the UNet sees coarse pixels in zt directly."""
    def __init__(self, hier_masks, channels=32, dim_mults=(1, 2, 2, 2)):
        super().__init__()
        self.hier = hier_masks
        self.net = make_unet(channels, dim_mults, in_ch=1, out_ch=1)
        n = sum(p.numel() for p in self.parameters())
        print(f"[SingleVelocity] {n:,} parameters")

    def forward(self, zt, t):
        raw = self.net(zt, t, classes=None)
        w = self.hier.beta_dot(t)[:, None, :, :]
        return w * raw

    def forward_scalar_t(self, zt, t_scalar):
        """Fast path for sampling: all samples at same time."""
        B = zt.shape[0]
        t = torch.full((B,), t_scalar, device=zt.device)
        raw = self.net(zt, t, classes=None)
        w = self.hier.beta_dot_scalar_t(t_scalar, B)
        return w * raw


class MultiVelocity(nn.Module):
    """One smaller UNet per scale.
    Each sub-network receives 2 channels:
      ch0 = zt (full current state)
      ch1 = resolved coarse pixels (zero on unresolved pixels)
    This explicitly provides coarse context to each fine-scale network.
    """
    def __init__(self, hier_masks, channels_per_net=16, dim_mults=(1, 2, 2)):
        super().__init__()
        self.hier = hier_masks
        self.num_masks = hier_masks.num_masks
        self.nets = nn.ModuleList([
            make_unet(channels_per_net, dim_mults, in_ch=2, out_ch=1)
            for _ in range(self.num_masks)
        ])
        n = sum(p.numel() for p in self.parameters())
        print(f"[MultiVelocity] {self.num_masks} nets, {n:,} total parameters")

    def forward(self, zt, t):
        """Training path: batch has mixed t values across scales."""
        B, C, H, W = zt.shape
        out = torch.zeros_like(zt)
        w = self.hier.beta_dot(t)[:, None, :, :]
        coarse_ctx = self.hier.get_coarse_context(zt, t)

        for k in range(self.num_masks):
            active = (t >= k) & (t <= k + 1)
            if not active.any():
                continue
            idx = active.nonzero(as_tuple=True)[0]
            t_local = (t[idx] - k).clamp(0, 1)
            net_input = torch.cat([zt[idx], coarse_ctx[idx]], dim=1)
            out[idx] = self.nets[k](net_input, t_local, classes=None)

        return w * out

    def forward_scalar_t(self, zt, t_scalar):
        """Fast path for sampling: all samples at same time, only one net active."""
        B = zt.shape[0]
        k = self.hier.get_active_scale(t_scalar)
        t_local = torch.full((B,), t_scalar - k, device=zt.device).clamp(0, 1)
        coarse_ctx = self.hier.get_coarse_context_scalar_t(zt, t_scalar)
        net_input = torch.cat([zt, coarse_ctx], dim=1)
        raw = self.nets[k](net_input, t_local, classes=None)
        w = self.hier.beta_dot_scalar_t(t_scalar, B)
        return w * raw


# ─── Interpolant & Sampler ──────────────────────────────────────────────────

class MultiscaleInterpolant:
    def __init__(self, hier_masks):
        self.hier = hier_masks

    def It(self, z0, z1, t):
        a = self.hier.alpha(t)[:, None, :, :]
        b = self.hier.beta(t)[:, None, :, :]
        return a * z0 + b * z1

    def Rt(self, z0, z1, t):
        ad = self.hier.alpha_dot(t)[:, None, :, :]
        bd = self.hier.beta_dot(t)[:, None, :, :]
        return ad * z0 + bd * z1


class Sampler:
    """
    Multiscale sampler that integrates each phase INDEPENDENTLY.

    The mask velocity field is piecewise in t: across phase boundaries the
    active band changes, so the velocity has a jump. RK4/Heun/EM steps must
    therefore stay strictly inside one phase — otherwise k4 (or the second
    Heun stage) lands at t=k+1 and evaluates the *next* phase's velocity,
    silently mixing two unrelated fields and producing huge spurious high-k
    energy. (Earlier _make_tgrid concatenated phases and let one RK4 step
    span a boundary; that was a bug.)

    Each phase k is integrated over [k+t_min, k+1-t_min] with `n_k` points
    placed either uniformly or at Chebyshev (cosine-clustered) nodes. The
    final state of phase k is the initial state of phase k+1 — same total
    work as before, just no boundary-spanning steps.
    """
    def __init__(self, num_masks):
        self.num_masks = num_masks

    def _f(self, model, zt, t_scalar):
        return model.forward_scalar_t(zt, t_scalar.item() if isinstance(t_scalar, torch.Tensor) else t_scalar)

    def _phase_grids(self, steps, t_min=1e-3, grid_mode='uniform'):
        """Return list of per-phase node tensors. Each grid is strictly inside (k, k+1).
        grid_mode in {'uniform', 'cosine'}."""
        K = self.num_masks
        base = steps // K
        rem = steps % K
        grids = []
        for k in range(K):
            n_k = base + (1 if k < rem else 0)
            if n_k == 0:
                continue
            if grid_mode == 'cosine':
                i = torch.arange(n_k + 1, dtype=torch.float64)
                s = 0.5 - 0.5 * torch.cos(math.pi * i / n_k)
            else:
                s = torch.linspace(0, 1, n_k + 1, dtype=torch.float64)
            s = t_min + (1 - 2 * t_min) * s
            grids.append((k + s).float())
        return grids

    @torch.no_grad()
    def EM(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3, grid_mode='uniform'):
        grids = self._phase_grids(steps, t_min=t_min, grid_mode=grid_mode)
        zt = z0
        for nodes in grids:
            for i in range(len(nodes) - 1):
                tv = float(nodes[i]); dtv = float(nodes[i + 1] - nodes[i])
                zt = zt + self._f(model, zt, tv) * dtv
        return zt

    @torch.no_grad()
    def Heun(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3, grid_mode='uniform'):
        grids = self._phase_grids(steps, t_min=t_min, grid_mode=grid_mode)
        zt = z0
        for nodes in grids:
            for i in range(len(nodes) - 1):
                tv = float(nodes[i]); dtv = float(nodes[i + 1] - nodes[i])
                k1 = self._f(model, zt, tv)
                k2 = self._f(model, zt + dtv * k1, tv + dtv)
                zt = zt + 0.5 * dtv * (k1 + k2)
        return zt

    @torch.no_grad()
    def RK4(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3, grid_mode='uniform'):
        grids = self._phase_grids(steps, t_min=t_min, grid_mode=grid_mode)
        zt = z0
        for nodes in grids:
            for i in range(len(nodes) - 1):
                tv = float(nodes[i]); dtv = float(nodes[i + 1] - nodes[i])
                k1 = self._f(model, zt, tv)
                k2 = self._f(model, zt + 0.5 * dtv * k1, tv + 0.5 * dtv)
                k3 = self._f(model, zt + 0.5 * dtv * k2, tv + 0.5 * dtv)
                k4 = self._f(model, zt + dtv * k3, tv + dtv)
                zt = zt + (dtv / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return zt


# ─── Trainer ────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data
        cfg = config
        sigma_sq = cfg.sigma_sq * ((2 * math.pi)**2 + cfg.ls**2)**cfg.s
        self.amplitude = precompute_matern_amplitude(cfg.grid_size, sigma_sq, cfg.ls, cfg.s)
        self.test_data = sample_matern_batch(self.amplitude, 500, device='cpu')
        print(f"[Data] Matern grid={cfg.grid_size}, s={cfg.s}, ls={cfg.ls}, test std={self.test_data.std():.4f}")

        # hierarchical masks & conditional variance
        self.hier = HierarchicalMasks(cfg.grid_size, cfg.num_masks, device=self.device)
        var_data = sample_matern_batch(self.amplitude, 2000, device='cpu')
        cond_vars = estimate_conditional_variances(var_data, [m.cpu() for m in self.hier.masks])
        # optionally scale all conditional variances by noise_scale
        scaled_vars = [v * cfg.noise_scale for v in cond_vars]
        print(f"[Conditional variances] scale={cfg.noise_scale}")
        for i, (raw, sc) in enumerate(zip(cond_vars, scaled_vars)):
            n_pts = int(self.hier.masks[i].sum().item())
            print(f"  scale {i} ({n_pts:5d} pts): cond_var={raw:.8f}  noise_std={sc**0.5:.6f}")
        self.noise_std = build_noise_std(self.hier.masks, scaled_vars, device=self.device)
        print(f"[Noise] avg noise std = {self.noise_std.mean():.6f}")

        # build per-pixel loss weight so each scale contributes equally to the total loss.
        # unweighted loss at scale k ~ n_k * cond_var_k. We want w_k * n_k * cond_var_k = const.
        # so w_k = 1 / (n_k * cond_var_k), then normalize so mean weight = 1.
        loss_weight = torch.zeros(cfg.grid_size, cfg.grid_size, device=self.device)
        scale_contributions = []
        for i, m in enumerate(self.hier.masks):
            n_k = m.sum().item()
            w_k = 1.0 / (n_k * scaled_vars[i])
            loss_weight += w_k * m
            scale_contributions.append(w_k * n_k * scaled_vars[i])  # should all be 1.0
        loss_weight = loss_weight / loss_weight.mean()
        self.loss_weight = loss_weight[None, None, :, :]  # (1, 1, H, W)
        # print per-scale weight and verify equal contribution
        for i, m in enumerate(self.hier.masks):
            n_k = int(m.sum().item())
            w_i = (loss_weight[0, 0] * m).sum().item() / n_k if n_k > 0 else 0
            print(f"  scale {i} ({n_k:5d} pts): per-pixel loss weight={w_i:.4f}")
        del var_data

        # interpolant
        self.interpolant = MultiscaleInterpolant(self.hier)
        self.sampler = Sampler(cfg.num_masks)

        # model
        if cfg.net_mode == 'single':
            self.model = SingleVelocity(
                self.hier, channels=cfg.unet_channels, dim_mults=cfg.unet_dim_mults
            ).to(self.device)
        else:
            self.model = MultiVelocity(
                self.hier, channels_per_net=cfg.multi_channels, dim_mults=cfg.multi_dim_mults
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_steps, eta_min=cfg.base_lr * 0.01)
        self.time_dist = torch.distributions.Uniform(low=cfg.t_min_train, high=cfg.t_max_train)
        self.global_step = 0

    def make_noise_iid(self, batch_size):
        """IID noise with per-pixel variance matching conditional variance."""
        G = self.config.grid_size
        z = torch.randn(batch_size, 1, G, G, device=self.device)
        return z * self.noise_std[None, None, :, :]

    def make_noise_data_dep(self, batch_data):
        """Data-dependent noise: random linear combination of centered data,
        giving noise with the empirical covariance of the data.
        z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j}
        """
        N = batch_data.shape[0]
        # batch_data: (N, H, W)
        centered = batch_data - batch_data.mean(dim=0, keepdim=True)
        xi = torch.randn(N, N, device=batch_data.device)
        noise = torch.einsum('nhw,nb->bhw', centered, xi) / math.sqrt(N)
        return noise.unsqueeze(1)  # (N, 1, H, W)

    def train_step(self):
        cfg = self.config
        batch = sample_matern_batch(self.amplitude, cfg.batch_size, device=self.device)
        z0 = self.make_noise_iid(cfg.batch_size)
        z1 = batch.unsqueeze(1)
        t = cfg.num_masks * self.time_dist.sample((cfg.batch_size,)).to(self.device)

        zt = self.interpolant.It(z0, z1, t)
        target = self.interpolant.Rt(z0, z1, t)
        pred = self.model(zt, t)
        if self.config.use_weighted_loss:
            loss = (self.loss_weight * (pred - target).pow(2)).sum(dim=(1, 2, 3)).mean()
        else:
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
        print(f"[Training] mode={cfg.net_mode}, max_steps={cfg.max_steps}, batch={cfg.batch_size}")
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
        """Evaluate at multiple step counts with both EM and RK4."""
        cfg = self.config
        self.model.eval()

        num_eval = cfg.num_eval
        truth_np = self.test_data[:num_eval].numpy()
        kvals, spec_truth = get_fourier_spectrum(truth_np)

        z0 = self.make_noise_iid(num_eval)

        step_counts = cfg.eval_step_counts
        results = {}

        for nsteps in step_counts:
            # RK4 integration (boundary-aware)
            gen_rk = self.sampler.RK4(z0.clone(), self.model, steps=nsteps,
                                       t_min=cfg.t_min_sample, t_max=cfg.t_max_sample)
            gen_rk_np = gen_rk.squeeze(1).cpu().numpy()
            _, spec_rk = get_fourier_spectrum(gen_rk_np)

            # Heun (RK2, boundary-aware)
            gen_heun = self.sampler.Heun(z0.clone(), self.model, steps=nsteps,
                                          t_min=cfg.t_min_sample, t_max=cfg.t_max_sample)
            gen_heun_np = gen_heun.squeeze(1).cpu().numpy()
            _, spec_heun = get_fourier_spectrum(gen_heun_np)

            # EM integration (boundary-aware)
            gen_em = self.sampler.EM(z0.clone(), self.model, steps=nsteps,
                                      t_min=cfg.t_min_sample, t_max=cfg.t_max_sample)
            gen_em_np = gen_em.squeeze(1).cpu().numpy()
            _, spec_em = get_fourier_spectrum(gen_em_np)

            for tag, spec_gen, gen_np in [('RK4', spec_rk, gen_rk_np), ('Heun', spec_heun, gen_heun_np), ('EM', spec_em, gen_em_np)]:
                std_ratio = gen_np.std() / (truth_np.std() + 1e-12)
                spec_rel = np.mean(np.abs(spec_truth - spec_gen) / (np.abs(spec_truth) + 1e-12))
                spec_l1 = np.mean(np.abs(spec_truth - spec_gen))
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
        ax = axes[0]
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_counts)))
        for i, ns in enumerate(step_counts):
            ax.loglog(kvals, results[f'RK4{ns}']['spec_gen'], '--', color=colors[i], label=f'RK4 {ns}')
            ax.loglog(kvals, results[f'Heun{ns}']['spec_gen'], '-.', color=colors[i], alpha=0.7, label=f'Heun {ns}')
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

        # data
        self.grid_size = 64
        self.sigma_sq = 1.0
        self.ls = 1.0
        self.s = 3.0
        self.batch_size = 200

        # multiscale
        self.num_masks = 4
        self.net_mode = 'single'  # 'single' or 'multi'
        self.noise_scale = 1.0  # multiply all conditional variances by this constant
        self.use_weighted_loss = True

        # training
        self.base_lr = 2e-4
        self.max_steps = 5000
        self.t_min_train = 1e-3
        self.t_max_train = 1.0 - 1e-3
        self.t_min_sample = 1e-3
        self.t_max_sample = 1.0 - 1e-3
        self.print_loss_every = 50
        self.test_every = 1000
        self.num_eval = 100
        self.eval_step_counts = None  # set in main based on num_masks

        # single-net architecture
        self.unet_channels = 32
        self.unet_dim_mults = (1, 2, 2, 2)

        # multi-net architecture (per sub-network)
        self.multi_channels = 16
        self.multi_dim_mults = (1, 2, 2)

        self.save_dir = 'results/multiscale_interp'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--ls', type=float, default=1.0)
    p.add_argument('--sigma_sq', type=float, default=1.0)
    p.add_argument('--num_masks', type=int, default=4)
    p.add_argument('--net_mode', type=str, default='single', choices=['single', 'multi'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test_every', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default='results/multiscale_interp')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--multi_channels', type=int, default=16)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--no_weighted_loss', action='store_true')
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
    config.num_masks = args.num_masks
    config.net_mode = args.net_mode
    config.test_every = args.test_every
    config.save_dir = args.save_dir
    config.use_wandb = not args.no_wandb
    config.multi_channels = args.multi_channels
    config.noise_scale = args.noise_scale
    config.use_weighted_loss = not args.no_weighted_loss

    # Set eval step counts as multiples of num_masks
    K = config.num_masks
    config.eval_step_counts = [K, 2*K, 4*K, 8*K]
    config.num_eval = 200

    os.makedirs(config.save_dir, exist_ok=True)

    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    run_name = f"GF_ms_{config.net_mode}_masks{config.num_masks}_s{config.s}_grid{config.grid_size}_{log_base}"

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name)
        for k, v in vars(config).items():
            if isinstance(v, (int, float, bool, str, list, tuple)):
                setattr(wandb.config, k, v)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
