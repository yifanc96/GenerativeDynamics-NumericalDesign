"""
Wavelet-based multiscale flow matching v2: per-sub-band noise.

Key improvement over v1: noise is generated in Haar COEFFICIENT space with
per-sub-band variance (LH, HL, HH have different variances), then reconstructed
to pixel space. This gives Lip constant bounded independently of resolution.

The Haar wavelet transform decomposes x into:
  - LL_K: coarsest low-pass (G/2^K × G/2^K)
  - For each level j=K,...,1: LH_j, HL_j, HH_j (3 detail sub-bands)

The interpolant transitions sub-bands coarsest to finest:
  Phase 0:   LL_K
  Phase 1:   LH_K + HL_K + HH_K (simultaneously, with per-sub-band noise)
  Phase 2:   LH_{K-1} + HL_{K-1} + HH_{K-1}
  ...
  Phase K:   LH_1 + HL_1 + HH_1
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


# ─── Data generation (same as before) ────────────────────────────────────────

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


# ─── Haar wavelet transform ─────────────────────────────────────────────────

def haar_forward_one_level(x):
    """One level of 2D Haar forward transform.
    x: (B, H, W) -> LL, LH, HL, HH each (B, H/2, W/2)"""
    B, H, W = x.shape
    x = x.view(B, H // 2, 2, W // 2, 2)
    a, b, c, d = x[:, :, 0, :, 0], x[:, :, 0, :, 1], x[:, :, 1, :, 0], x[:, :, 1, :, 1]
    LL = (a + b + c + d) / 2.0
    LH = (a - b + c - d) / 2.0
    HL = (a + b - c - d) / 2.0
    HH = (a - b - c + d) / 2.0
    return LL, LH, HL, HH


def haar_inverse_one_level(LL, LH, HL, HH):
    """One level of 2D Haar inverse transform.
    LL, LH, HL, HH each (B, H/2, W/2) -> x (B, H, W)"""
    B, Hh, Wh = LL.shape
    a = (LL + LH + HL + HH) / 2.0
    b = (LL - LH + HL - HH) / 2.0
    c = (LL + LH - HL - HH) / 2.0
    d = (LL - LH - HL + HH) / 2.0
    x = torch.zeros(B, Hh * 2, Wh * 2, device=LL.device)
    x[:, 0::2, 0::2] = a
    x[:, 0::2, 1::2] = b
    x[:, 1::2, 0::2] = c
    x[:, 1::2, 1::2] = d
    return x


def haar_forward_multi(x, K):
    """K-level Haar forward transform.
    Returns: list of (LL, [(LH, HL, HH) for each level from coarsest to finest])
    The list order: [(LH_K, HL_K, HH_K), ..., (LH_1, HL_1, HH_1)]"""
    details = []
    current = x
    for j in range(K):
        LL, LH, HL, HH = haar_forward_one_level(current)
        details.append((LH, HL, HH))
        current = LL
    details.reverse()  # coarsest first
    return current, details  # current = LL_K


def haar_inverse_multi(LL, details):
    """K-level Haar inverse transform.
    details: [(LH_K, HL_K, HH_K), ..., (LH_1, HL_1, HH_1)] coarsest first"""
    current = LL
    for LH, HL, HH in details:
        current = haar_inverse_one_level(current, LH, HL, HH)
    return current


# ─── Wavelet coefficient management ─────────────────────────────────────────

class HaarWavelet:
    """Multi-level Haar wavelet with per-sub-band operations."""

    def __init__(self, grid_size, num_levels, device='cpu'):
        self.grid_size = grid_size
        self.num_levels = num_levels  # K
        self.num_phases = num_levels + 1  # LL + K detail levels
        self.device = device

    def decompose(self, x):
        """Decompose (B, G, G) into wavelet coefficients.
        Returns: (LL_K, details) where details = [(LH_K, HL_K, HH_K), ...]"""
        return haar_forward_multi(x, self.num_levels)

    def reconstruct(self, LL, details):
        """Reconstruct (B, G, G) from wavelet coefficients."""
        return haar_inverse_multi(LL, details)

    def reconstruct_phase(self, LL, details, phase):
        """Reconstruct ONLY the contribution of a specific phase to pixel space.
        Phase 0 = LL, Phase k (1..K) = detail level k (coarsest detail first)."""
        K = self.num_levels
        G = self.grid_size
        B = LL.shape[0]
        if phase == 0:
            # reconstruct LL only: set all details to zero
            zero_details = [(torch.zeros_like(d) for d in det) for det in details]
            zero_details = [tuple(z) for z in zero_details]
            return haar_inverse_multi(LL, zero_details)
        else:
            # reconstruct detail at level (phase-1), set everything else to zero
            Gh = G // (2**K)
            zero_LL = torch.zeros(B, Gh, Gh, device=self.device)
            zero_details = []
            for k in range(K):
                if k == phase - 1:
                    zero_details.append(details[k])
                else:
                    zero_details.append(tuple(torch.zeros_like(d) for d in details[k]))
            return haar_inverse_multi(zero_LL, zero_details)

    def estimate_sub_band_variances(self, data):
        """Estimate per-sub-band coefficient variance.
        Returns: (ll_var, [(lh_var, hl_var, hh_var) for each level])"""
        LL, details = self.decompose(data)
        ll_var = LL.var(dim=0).mean().item()
        detail_vars = []
        for LH, HL, HH in details:
            detail_vars.append((
                LH.var(dim=0).mean().item(),
                HL.var(dim=0).mean().item(),
                HH.var(dim=0).mean().item(),
            ))
        return ll_var, detail_vars

    def make_noise(self, batch_size, ll_var, detail_vars):
        """Generate noise with correct per-sub-band variance in coefficient space,
        then reconstruct to pixel space."""
        K = self.num_levels
        G = self.grid_size
        Gh = G // (2**K)

        # coarsest LL noise
        noise_LL = math.sqrt(ll_var) * torch.randn(batch_size, Gh, Gh, device=self.device)

        # detail noise per sub-band
        noise_details = []
        for k in range(K):
            lh_v, hl_v, hh_v = detail_vars[k]
            # resolution at this level: G / 2^(K-k) on each side
            # details[0] is coarsest detail (resolution Gh), details[-1] is finest (resolution G/2)
            res = Gh * (2**k)
            noise_LH = math.sqrt(lh_v) * torch.randn(batch_size, res, res, device=self.device)
            noise_HL = math.sqrt(hl_v) * torch.randn(batch_size, res, res, device=self.device)
            noise_HH = math.sqrt(hh_v) * torch.randn(batch_size, res, res, device=self.device)
            noise_details.append((noise_LH, noise_HL, noise_HH))

        # reconstruct to pixel space
        return self.reconstruct(noise_LL, noise_details), noise_LL, noise_details

    def make_noise_decomposed(self, batch_size, ll_var, detail_vars):
        """Generate noise and return both pixel-space and decomposed forms."""
        noise_pixel, noise_LL, noise_details = self.make_noise(batch_size, ll_var, detail_vars)
        return noise_LL, noise_details


# ─── Interpolant ─────────────────────────────────────────────────────────────

class WaveletInterpolant:
    """
    Interpolant in wavelet coefficient space.

    Phase ordering (coarsest first):
      Phase 0: LL_K
      Phase 1: (LH_K, HL_K, HH_K) — coarsest detail
      Phase 2: (LH_{K-1}, HL_{K-1}, HH_{K-1})
      ...
      Phase K: (LH_1, HL_1, HH_1) — finest detail

    At time t in [k, k+1], phase k's sub-bands interpolate: alpha*noise + beta*data
    alpha = 1-(t-k), beta = t-k
    """
    def __init__(self, haar):
        self.haar = haar
        self.K = haar.num_levels
        self.num_phases = haar.num_phases

    def build_state(self, noise_LL, noise_details, data_LL, data_details, t):
        """Build interpolated coefficients at time t.
        Returns: (interp_LL, interp_details) — the current wavelet coefficients."""
        K = self.K
        B = t.shape[0]

        # Phase 0: LL
        # LL is active at t in [0, 1]
        alpha_0 = (1 - t).clamp(0, 1).view(-1, 1, 1)
        beta_0 = t.clamp(0, 1).view(-1, 1, 1)
        resolved_0 = (t >= 1).view(-1, 1, 1).float()
        unresolved_0 = (t <= 0).view(-1, 1, 1).float()
        active_0 = ((t > 0) & (t < 1)).view(-1, 1, 1).float()

        interp_LL = (resolved_0 * data_LL
                     + active_0 * (alpha_0 * noise_LL + beta_0 * data_LL)
                     + unresolved_0 * noise_LL)

        # Phases 1..K: detail levels
        interp_details = []
        for j in range(K):
            phase = j + 1
            alpha_j = (1 - (t - phase)).clamp(0, 1).view(-1, 1, 1)
            beta_j = (t - phase).clamp(0, 1).view(-1, 1, 1)
            resolved_j = (t >= phase + 1).view(-1, 1, 1).float()
            unresolved_j = (t <= phase).view(-1, 1, 1).float()
            active_j = ((t > phase) & (t < phase + 1)).view(-1, 1, 1).float()

            level_interp = []
            for sb in range(3):  # LH, HL, HH
                n_sb = noise_details[j][sb]
                d_sb = data_details[j][sb]
                interp_sb = (resolved_j * d_sb
                             + active_j * (alpha_j * n_sb + beta_j * d_sb)
                             + unresolved_j * n_sb)
                level_interp.append(interp_sb)
            interp_details.append(tuple(level_interp))

        return interp_LL, interp_details

    def It(self, noise_LL, noise_details, data_LL, data_details, t):
        """Build pixel-space interpolated state."""
        interp_LL, interp_details = self.build_state(
            noise_LL, noise_details, data_LL, data_details, t)
        return self.haar.reconstruct(interp_LL, interp_details).unsqueeze(1)

    def Rt(self, noise_LL, noise_details, data_LL, data_details, t):
        """Target velocity: -noise + data on the active phase's sub-bands."""
        K = self.K
        B = t.shape[0]
        Gh = self.haar.grid_size // (2**K)

        # Phase 0: LL
        active_0 = ((t >= 0) & (t <= 1)).view(-1, 1, 1).float()
        target_LL = active_0 * (-noise_LL + data_LL)

        target_details = []
        for j in range(K):
            phase = j + 1
            active_j = ((t >= phase) & (t <= phase + 1)).view(-1, 1, 1).float()
            level_target = []
            for sb in range(3):
                level_target.append(active_j * (-noise_details[j][sb] + data_details[j][sb]))
            target_details.append(tuple(level_target))

        # reconstruct to pixel space
        return self.haar.reconstruct(target_LL, target_details).unsqueeze(1)


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
    """UNet in pixel space, output projected onto active sub-bands via wavelet."""
    def __init__(self, haar, channels=32, dim_mults=(1, 2, 2, 2)):
        super().__init__()
        self.haar = haar
        self.net = make_unet(channels, dim_mults)
        n = sum(p.numel() for p in self.parameters())
        print(f"[WaveletVelocity-single] {n:,} parameters")

    def _project_to_phase(self, raw_pixel, phase):
        """Project pixel-space output onto the active phase's sub-bands."""
        K = self.haar.num_levels
        LL, details = self.haar.decompose(raw_pixel)
        Gh = LL.shape[1]
        B = raw_pixel.shape[0]

        if phase == 0:
            zero_details = [tuple(torch.zeros_like(d) for d in det) for det in details]
            return self.haar.reconstruct(LL, zero_details)
        else:
            zero_LL = torch.zeros(B, Gh, Gh, device=raw_pixel.device)
            proj_details = []
            for k in range(K):
                if k == phase - 1:
                    proj_details.append(details[k])
                else:
                    proj_details.append(tuple(torch.zeros_like(d) for d in details[k]))
            return self.haar.reconstruct(zero_LL, proj_details)

    def forward(self, zt, t):
        """Training: mixed t values, project output per sample."""
        raw = self.net(zt, t, classes=None)[:, 0]  # (B, G, G)
        B = zt.shape[0]
        out = torch.zeros_like(raw)
        for phase in range(self.haar.num_phases):
            active = (t >= phase) & (t <= phase + 1)
            if not active.any():
                continue
            idx = active.nonzero(as_tuple=True)[0]
            out[idx] = self._project_to_phase(raw[idx], phase)
        return out.unsqueeze(1)

    def forward_scalar_t(self, zt, t_scalar):
        """Fast path: all samples at same time."""
        phase = int(min(max(t_scalar, 0), self.haar.num_phases - 1))
        B = zt.shape[0]
        t = torch.full((B,), t_scalar, device=zt.device)
        raw = self.net(zt, t, classes=None)[:, 0]
        proj = self._project_to_phase(raw, phase)
        return proj.unsqueeze(1)


class WaveletVelocityMulti(nn.Module):
    """One UNet per phase. Each gets the full pixel-space zt as input,
    outputs the velocity projected onto its phase's bands.
    Each sub-network sees t_local in [0, 1]."""
    def __init__(self, haar, channels_per_net=32, dim_mults=(1, 2, 2, 2)):
        super().__init__()
        self.haar = haar
        self.num_phases = haar.num_phases
        self.nets = nn.ModuleList([
            make_unet(channels_per_net, dim_mults)
            for _ in range(self.num_phases)
        ])
        n = sum(p.numel() for p in self.parameters())
        print(f"[WaveletVelocity-multi] {self.num_phases} nets, {n:,} total params")

    def _project_to_phase(self, raw_pixel, phase):
        K = self.haar.num_levels
        LL, details = self.haar.decompose(raw_pixel)
        Gh = LL.shape[1]
        B = raw_pixel.shape[0]
        if phase == 0:
            zero_details = [tuple(torch.zeros_like(d) for d in det) for det in details]
            return self.haar.reconstruct(LL, zero_details)
        else:
            zero_LL = torch.zeros(B, Gh, Gh, device=raw_pixel.device)
            proj_details = []
            for k in range(K):
                if k == phase - 1:
                    proj_details.append(details[k])
                else:
                    proj_details.append(tuple(torch.zeros_like(d) for d in details[k]))
            return self.haar.reconstruct(zero_LL, proj_details)

    def forward(self, zt, t):
        """Training: process each phase's samples with its own network."""
        B = zt.shape[0]
        out = torch.zeros(B, zt.shape[2], zt.shape[3], device=zt.device)
        for phase in range(self.num_phases):
            active = (t >= phase) & (t <= phase + 1)
            if not active.any():
                continue
            idx = active.nonzero(as_tuple=True)[0]
            t_local = (t[idx] - phase).clamp(0, 1)
            raw = self.nets[phase](zt[idx], t_local, classes=None)[:, 0]
            out[idx] = self._project_to_phase(raw, phase)
        return out.unsqueeze(1)

    def forward_scalar_t(self, zt, t_scalar):
        """Fast path: all samples at same time, only one network active."""
        phase = int(min(max(t_scalar, 0), self.num_phases - 1))
        B = zt.shape[0]
        t_local = max(0, min(1, t_scalar - phase))
        t = torch.full((B,), t_local, device=zt.device)
        raw = self.nets[phase](zt, t, classes=None)[:, 0]
        proj = self._project_to_phase(raw, phase)
        return proj.unsqueeze(1)


# ─── Sampler ────────────────────────────────────────────────────────────────

class WaveletSampler:
    """Per-phase integration. Each phase k is integrated independently over
    (k+t_min, k+1-t_min) so RK4 steps never span a phase boundary (which would
    mix the active wavelet sub-band of phase k with that of phase k+1)."""
    def __init__(self, num_phases):
        self.num_phases = num_phases

    def _f(self, model, zt, t_scalar):
        return model.forward_scalar_t(zt, t_scalar)

    def _phase_grids(self, steps, t_min=1e-3):
        K = self.num_phases
        base = steps // K
        rem = steps % K
        grids = []
        for k in range(K):
            n_k = base + (1 if k < rem else 0)
            if n_k == 0:
                continue
            s = torch.linspace(t_min, 1 - t_min, n_k + 1)
            grids.append(k + s)
        return grids

    @torch.no_grad()
    def EM(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3):
        grids = self._phase_grids(steps, t_min=t_min)
        zt = z0
        for nodes in grids:
            for i in range(len(nodes) - 1):
                tv = float(nodes[i]); dtv = float(nodes[i + 1] - nodes[i])
                zt = zt + self._f(model, zt, tv) * dtv
        return zt

    @torch.no_grad()
    def Heun(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3):
        grids = self._phase_grids(steps, t_min=t_min)
        zt = z0
        for nodes in grids:
            for i in range(len(nodes) - 1):
                tv = float(nodes[i]); dtv = float(nodes[i + 1] - nodes[i])
                k1 = self._f(model, zt, tv)
                k2 = self._f(model, zt + dtv * k1, tv + dtv)
                zt = zt + 0.5 * dtv * (k1 + k2)
        return zt

    @torch.no_grad()
    def RK4(self, z0, model, steps, t_min=1e-3, t_max=1 - 1e-3):
        grids = self._phase_grids(steps, t_min=t_min)
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

        cfg = config
        sigma_sq = cfg.sigma_sq * ((2 * math.pi)**2 + cfg.ls**2)**cfg.s
        self.amplitude = precompute_matern_amplitude(cfg.grid_size, sigma_sq, cfg.ls, cfg.s)
        self.test_data = sample_matern_batch(self.amplitude, 500, device='cpu')
        print(f"[Data] Matern grid={cfg.grid_size}, s={cfg.s}, ls={cfg.ls}, test std={self.test_data.std():.4f}")

        self.haar = HaarWavelet(cfg.grid_size, cfg.num_levels, device=self.device)

        # estimate per-sub-band variances
        var_data = sample_matern_batch(self.amplitude, 3000, device='cpu')
        self.ll_var, self.detail_vars = self.haar.estimate_sub_band_variances(var_data)
        print(f"[Wavelet] num_levels={cfg.num_levels}, num_phases={self.haar.num_phases}")
        print(f"  LL (coarsest): var={self.ll_var:.4f}")
        total_var = self.ll_var
        for k, (lh_v, hl_v, hh_v) in enumerate(self.detail_vars):
            print(f"  Level {k} detail: LH={lh_v:.6f}, HL={hl_v:.6f}, HH={hh_v:.6f}")
            total_var += lh_v + hl_v + hh_v
        print(f"  sum={total_var:.4f}, data_var={var_data.var().item():.4f}")
        del var_data

        self.interpolant = WaveletInterpolant(self.haar)
        self.sampler = WaveletSampler(self.haar.num_phases)

        if getattr(cfg, 'net_mode', 'single') == 'multi':
            self.model = WaveletVelocityMulti(
                self.haar, channels_per_net=cfg.unet_channels, dim_mults=cfg.unet_dim_mults
            ).to(self.device)
        else:
            self.model = WaveletVelocity(
                self.haar, channels=cfg.unet_channels, dim_mults=cfg.unet_dim_mults
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_steps, eta_min=cfg.base_lr * 0.01)
        self.time_dist = torch.distributions.Uniform(low=cfg.t_min_train, high=cfg.t_max_train)
        self.global_step = 0

    def train_step(self):
        cfg = self.config
        batch = sample_matern_batch(self.amplitude, cfg.batch_size, device=self.device)

        data_LL, data_details = self.haar.decompose(batch)
        noise_LL, noise_details = self.haar.make_noise_decomposed(
            cfg.batch_size, self.ll_var, self.detail_vars)

        t = self.haar.num_phases * self.time_dist.sample((cfg.batch_size,)).to(self.device)

        zt = self.interpolant.It(noise_LL, noise_details, data_LL, data_details, t)
        target = self.interpolant.Rt(noise_LL, noise_details, data_LL, data_details, t)
        pred = self.model(zt, t)

        if getattr(self.config, 'phase_balance_loss', False):
            # Normalize loss per phase by the target's expected magnitude
            # Each phase's target ~ (-noise + data) on its band, with norm ~ sqrt(2 * band_var)
            B = pred.shape[0]
            sq_err = (pred - target).pow(2).sum(dim=(1, 2, 3))  # (B,)
            phase_per_sample = t.long().clamp(0, self.haar.num_phases - 1)
            # Get expected target norm^2 for each sample's phase
            # Phase 0 (LL): variance = ll_var * num_LL_coefficients
            # Phase k+1 (detail level k): variance = sum over LH/HL/HH of var * dim
            phase_target_var = torch.zeros(self.haar.num_phases, device=self.device)
            Gh = self.config.grid_size // (2**self.haar.num_levels)
            phase_target_var[0] = 2 * self.ll_var * Gh * Gh  # 2x because target = -noise + data
            for j in range(self.haar.num_levels):
                res = Gh * (2**j)
                lh_v, hl_v, hh_v = self.detail_vars[j]
                phase_target_var[j+1] = 2 * (lh_v + hl_v + hh_v) * res * res
            # divide each sample's squared error by its phase's expected norm^2
            normalized = sq_err / phase_target_var[phase_per_sample]
            loss = normalized.mean()
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
        print(f"[Training] num_phases={self.haar.num_phases}, max_steps={cfg.max_steps}")
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

        # generate initial noise in pixel space
        noise_pixel, _, _ = self.haar.make_noise(num_eval, self.ll_var, self.detail_vars)
        z0 = noise_pixel.unsqueeze(1)

        for nsteps in cfg.eval_step_counts:
            for tag, fn in [('RK4', self.sampler.RK4), ('Heun', self.sampler.Heun), ('EM', self.sampler.EM)]:
                gen = fn(z0.clone(), self.model, steps=nsteps,
                         t_min=cfg.t_min_sample, t_max=cfg.t_max_sample)
                gen_np = gen.squeeze(1).cpu().numpy()
                _, spec_gen = get_fourier_spectrum(gen_np)
                std_ratio = gen_np.std() / (truth_np.std() + 1e-12)
                spec_rel = np.mean(np.abs(spec_truth - spec_gen) / (np.abs(spec_truth) + 1e-12))
                print(f"    {tag:4s} steps={nsteps:3d}:  spec_relErr={spec_rel:.4f}  std_ratio={std_ratio:.4f}")
                if cfg.use_wandb:
                    wandb.log({
                        f"spec_relErr/{tag}_steps{nsteps}": spec_rel,
                        f"std_ratio/{tag}_steps{nsteps}": std_ratio,
                    }, step=self.global_step)

        # spectrum plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        nmax = cfg.eval_step_counts[-1]
        gen_final = self.sampler.RK4(z0.clone(), self.model, steps=nmax,
                                      t_min=cfg.t_min_sample, t_max=cfg.t_max_sample)
        gen_np = gen_final.squeeze(1).cpu().numpy()
        _, spec_gen = get_fourier_spectrum(gen_np)
        ax.loglog(kvals, spec_gen, 'r--', lw=1.5, label=f'RK4 {nmax}')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)'); ax.legend(); ax.set_title('Energy Spectrum')
        vmax = max(abs(truth_np[0].min()), abs(truth_np[0].max()))
        combined = np.concatenate([truth_np[0], gen_np[0]], axis=1)
        axes[1].imshow(combined, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Truth | RK4 {nmax}'); axes[1].axis('off')
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

        self.num_levels = 2  # K wavelet levels → K+1 phases
        self.net_mode = 'single'  # 'single' or 'multi'

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

        self.save_dir = 'results/wavelet_v2'


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
    p.add_argument('--net_mode', type=str, default='single', choices=['single', 'multi'])
    p.add_argument('--unet_channels', type=int, default=32)
    p.add_argument('--phase_balance_loss', action='store_true',
                   help='Normalize loss per phase to balance contributions across scales')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test_every', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default='results/wavelet_v2')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    for k, v in vars(args).items():
        if k == 'no_wandb':
            config.use_wandb = not v
        elif k == 'lr':
            config.base_lr = v
        elif hasattr(config, k):
            setattr(config, k, v)

    os.makedirs(config.save_dir, exist_ok=True)

    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    run_name = f"GF_wav2_L{config.num_levels}_s{config.s}_grid{config.grid_size}_{log_base}"

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name)
        for k, v in vars(config).items():
            if isinstance(v, (int, float, bool, str, list, tuple)):
                setattr(wandb.config, k, v)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
