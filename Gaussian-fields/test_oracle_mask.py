"""
Oracle test for the mask-based multiscale Sampler in train_multiscale_interpolation.py.

Build a "model" that returns the EXACT analytic conditional velocity (Gaussian
multiscale interpolant), plug it into the same Sampler.RK4 used in training,
and check that the generated samples have the right covariance / spectrum.

If this oracle reproduces the truth distribution closely, the Sampler + boundary
convention + per-band noise structure are correct, and any gap from a trained
network is purely an optimization issue.
"""
import math, sys, os
import numpy as np
import torch
import torch.nn as nn

from train_multiscale_interpolation import (
    HierarchicalMasks, Sampler,
    precompute_matern_amplitude, sample_matern_batch,
    estimate_conditional_variances, build_noise_std,
    get_fourier_spectrum,
)

torch.manual_seed(0)
np.random.seed(0)

# ─── Config ─────────────────────────────────────────────────────────────────
G          = 64
s          = 3.0
ls         = 1.0
sigma_sq_o = 1.0
num_masks  = int(os.environ.get('K', '6'))
n_emp      = 10000
n_test     = 500
device     = torch.device('cpu')

# ─── Data and ANALYTIC Sigma in pixel basis ────────────────────────────────
# Matern is exactly diagonal in the Fourier basis with eigenvalues = spec(k).
# Σ_pix = F^* diag(spec) F  →  built explicitly so we have the EXACT covariance,
# free of finite-sample rank-deficiency at large bands.
sigma_sq = sigma_sq_o * ((2 * math.pi)**2 + ls**2) ** s
amp = precompute_matern_amplitude(G, sigma_sq, ls, s)
spec = amp ** 2                                                  # (G, G)
spec_flat = spec.reshape(-1)                                     # (G*G,)
data = sample_matern_batch(amp, n_emp, device=device)            # for test only
d = G * G
print(f'[Data] G={G} s={s} d={d} std={data.std():.4f}')

# Build Sigma in pixel basis: it is *circulant* (translation-invariant) so
# Σ[i, j] = c[(i-j) mod (G,G)] where c = iFFT2(spec) (real-valued).
# Form the full d x d Sigma by indexing this 2D autocorrelation function.
print('[Sigma] building analytical covariance from circulant autocorrelation...')
spec_np = spec.numpy()
c2 = np.fft.ifft2(spec_np, norm='forward').real                 # autocorrelation; c2[0,0] = pix_var
# Sigma indices: row (i_y, i_x), col (j_y, j_x), Sigma = c2[(i-j) mod G]
ii = np.arange(G * G)
iy, ix = np.divmod(ii, G)
dy = (iy[:, None] - iy[None, :]) % G
dx = (ix[:, None] - ix[None, :]) % G
Sigma = torch.from_numpy(c2[dy, dx].astype(np.float64))
del dy, dx, ii, iy, ix
print(f'[Sigma] analytical Σ shape={tuple(Sigma.shape)}, '
      f'trace/d={Sigma.diag().mean().item():.4f}, '
      f'truth pixel var={(data**2).mean().item():.4f}')

# ─── Hierarchical masks (matches the trainer's convention) ──────────────────
hier = HierarchicalMasks(G, num_masks, device=device)
cond_vars = estimate_conditional_variances(data, [m.cpu() for m in hier.masks])
noise_std = build_noise_std(hier.masks, cond_vars, device=device)
print('[Noise] per-band cond variances:')
for i, (m, cv) in enumerate(zip(hier.masks, cond_vars)):
    n_pts = int(m.sum().item())
    print(f'  band {i} ({n_pts:4d} pts): cond_var={cv:.6e} std={cv**0.5:.4f}')

# ─── Per-phase oracle precomputation ────────────────────────────────────────
# Phase k (t in [k, k+1]) activates scale_idx = num_masks - 1 - k.
# F = pixels in active band, C = union of all already-resolved bands (phases 0..k-1).
phases = []                # list of dicts with all matrices needed at runtime
for k in range(num_masks):
    scale_idx = num_masks - 1 - k
    band_2d = hier.masks[scale_idx].cpu()
    F = torch.nonzero(band_2d.flatten(), as_tuple=False).flatten()
    # Resolved coarse pixels = bands at scale_idx > num_masks - 1 - k (coarser scales)
    # i.e. scales (num_masks - 1), (num_masks - 2), ..., (num_masks - k)
    C_list = []
    for j in range(k):
        prev_scale_idx = num_masks - 1 - j
        C_list.append(torch.nonzero(hier.masks[prev_scale_idx].cpu().flatten(), as_tuple=False).flatten())
    C = torch.cat(C_list) if len(C_list) > 0 else torch.empty(0, dtype=torch.long)

    Sigma_FF = Sigma[F][:, F]
    if len(C) > 0:
        Sigma_FC = Sigma[F][:, C]
        Sigma_CC = Sigma[C][:, C]
        ridge = 1e-12 * Sigma_CC.diag().mean().item()
        M_op = torch.solve(Sigma_FC.T, Sigma_CC + ridge * torch.eye(len(C), dtype=Sigma_CC.dtype))[0].T
        Sigma_FF_C = Sigma_FF - M_op @ Sigma_FC.T
    else:
        M_op = None
        Sigma_FF_C = Sigma_FF.clone()
    Sigma_FF_C = 0.5 * (Sigma_FF_C + Sigma_FF_C.T)               # symmetrize

    # PRACTICAL choice: σ²_k = mean conditional variance per pixel on band F.
    #   = (1/|F|) trace(Sigma_FF|C) = (1/|F|) Σ λ_i
    # This is exactly what train_multiscale_interpolation.estimate_conditional_variances
    # already computes (residual variance after ridge regression on coarser pixels).
    # No eigendecomposition needed; just average the diagonal.
    sigma2 = float(Sigma_FF_C.diag().mean().item())

    eigs = torch.symeig(Sigma_FF_C, eigenvectors=False)[0].numpy()
    eigs = eigs[eigs > eigs.max() * 1e-10]
    cn = float(eigs.max() / eigs.min()) if len(eigs) > 1 else 1.0
    print(f'  phase {k}: |F|={len(F):4d} |C|={len(C):4d}  cond={cn:.2e}  '
          f'σ²_mean={sigma2:.3e}  Lip~√cond≈{cn**0.5:.2f}')

    phases.append(dict(
        F=F, C=C, M_op=M_op, Sigma_FF_C=Sigma_FF_C, sigma2=sigma2,
    ))


# ─── Oracle: iid scalar noise σ²_k I per band, σ²_k = √(λ_min · λ_max) ─────
# (geometric mean of conditional eigenvalues — minimizes worst per-mode ratio)
class OracleVelocity(nn.Module):
    def __init__(self, hier, phases):
        super().__init__()
        self.hier = hier
        self.phases = phases
        self.num_masks = hier.num_masks

    def forward_scalar_t(self, zt, t_scalar):
        B, _, H, W = zt.shape
        k = self.hier.get_active_scale(t_scalar)
        s = float(t_scalar) - k
        a, b = 1.0 - s, s

        ph = self.phases[k]
        F, C = ph['F'], ph['C']
        Sigma_FF_C = ph['Sigma_FF_C']
        sigma2 = ph['sigma2']

        zt_flat = zt.view(B, -1)
        z_F = zt_flat[:, F]
        if len(C) > 0:
            z_C = zt_flat[:, C]
            mu = z_C @ ph['M_op'].T
        else:
            mu = torch.zeros_like(z_F)

        nF = len(F)
        Total = (a * a * sigma2) * torch.eye(nF, dtype=Sigma_FF_C.dtype) + (b * b) * Sigma_FF_C
        diff = (z_F - b * mu).T
        Xt = torch.solve(diff, Total)[0]
        z1_hat = mu + b * (Sigma_FF_C @ Xt).T
        z0_hat = (a * sigma2) * Xt.T
        v_F = z1_hat - z0_hat            # linear schedule, a' = -1, b' = 1

        out = torch.zeros(B, H * W, dtype=zt.dtype)
        out[:, F] = v_F
        return out.view(B, 1, H, W)


oracle = OracleVelocity(hier, phases)
sampler = Sampler(num_masks)


# ─── Custom Sampler with designed (paper) schedule per phase ────────────────
# Within each phase k (s in [0,1]), use α_k(s)² = (M_k - M_k^s)/(M_k - 1),
# β_k(s)² = (M_k^s - 1)/(M_k - 1).  M_k chosen as conditional cov cond number.
# This is the per-phase analogue of the paper's optimal scalar schedule.

class DesignedOracleVelocity(nn.Module):
    """Same as OracleVelocity but uses designed (α, β, α', β') per phase."""
    def __init__(self, hier, phases, M_k_list):
        super().__init__()
        self.hier = hier
        self.phases = phases
        self.M_k = M_k_list
        self.num_masks = hier.num_masks

    def _ab(self, k, s):
        M = self.M_k[k]
        if abs(M - 1.0) < 1e-9:
            a, b = 1.0 - s, s
            ad, bd = -1.0, 1.0
            return a, b, ad, bd
        denom = M - 1.0
        a2 = (M - M**s) / denom
        b2 = (M**s - 1.0) / denom
        a = math.sqrt(max(a2, 0.0))
        b = math.sqrt(max(b2, 0.0))
        # derivatives
        lnM = math.log(M)
        da2 = -M**s * lnM / denom
        db2 =  M**s * lnM / denom
        ad = 0.5 * da2 / (a + 1e-30)
        bd = 0.5 * db2 / (b + 1e-30)
        return a, b, ad, bd

    def forward_scalar_t(self, zt, t_scalar):
        B, _, H, W = zt.shape
        k = self.hier.get_active_scale(t_scalar)
        s = float(t_scalar) - k
        a, b, ad, bd = self._ab(k, s)

        ph = self.phases[k]
        F, C = ph['F'], ph['C']
        Sigma_FF_C = ph['Sigma_FF_C']
        sigma2 = ph['sigma2']

        zt_flat = zt.view(B, -1)
        z_F = zt_flat[:, F]
        if len(C) > 0:
            z_C = zt_flat[:, C]
            mu = z_C @ ph['M_op'].T
        else:
            mu = torch.zeros_like(z_F)

        nF = len(F)
        Total = (a * a * sigma2) * torch.eye(nF, dtype=Sigma_FF_C.dtype) + (b * b) * Sigma_FF_C
        diff = (z_F - b * mu).T
        Xt = torch.solve(diff, Total)[0]
        z1_hat = mu + b * (Sigma_FF_C @ Xt).T
        z0_hat = (a * sigma2) * Xt.T
        # designed velocity:  v = a' z0 + b' z1   (from It = a z0 + b z1, Rt = a' z0 + b' z1)
        v_F = ad * z0_hat + bd * z1_hat

        out = torch.zeros(B, H * W, dtype=zt.dtype)
        out[:, F] = v_F
        return out.view(B, 1, H, W)


# ─── Initial noise: iid scalar per band, std = √sigma2 ─────────────────────
def make_z0(B):
    out = torch.zeros(B, G * G, dtype=torch.float64)
    for k, ph in enumerate(phases):
        F = ph['F']
        out[:, F] = math.sqrt(ph['sigma2']) * torch.randn(B, len(F), dtype=torch.float64)
    return out.view(B, 1, G, G)


# ─── Run integration with multiple step counts ──────────────────────────────
test_data = sample_matern_batch(amp, n_test, device=device)      # truth samples
truth_var = test_data.var().item()

print()
print('Oracle integration test (RK4):')
print(f'{"steps":>6} | {"out_var":>12} | {"truth_var":>12} | {"|var_ratio-1|":>14} | {"band L2 err":>12}')

# precompute truth band power
def band_l2_err(out, truth):
    """Mean L2 relative error per Fourier band."""
    kv, Po = get_fourier_spectrum(out.squeeze(1).numpy() if out.dim() == 4 else out.numpy())
    kv, Pt = get_fourier_spectrum(truth.numpy())
    err = np.abs(Po - Pt) / (np.abs(Pt) + 1e-30)
    return float(np.mean(err))


for steps in [num_masks, 2 * num_masks, 4 * num_masks, 8 * num_masks, 16 * num_masks, 40 * num_masks]:
    z0 = make_z0(n_test)
    out = sampler.RK4(z0, oracle, steps=steps)
    out_var = out.var().item()
    err = band_l2_err(out.squeeze(1), test_data)
    print(f'{steps:>6} | {out_var:>12.4f} | {truth_var:>12.4f} | {abs(out_var/truth_var - 1):>14.4e} | {err:>12.4e}')


# ─── Linear schedule with non-uniform time grid (clustered near boundaries) ─
print()
print('Linear schedule, COSINE-clustered grid per phase:')

@torch.no_grad()
def rk4_clustered(model, z0, total_steps, eps=1e-3):
    """Per phase: nodes at s = 0.5 - 0.5 cos(pi * i/N), so they cluster near 0 and 1."""
    K = model.num_masks
    base = total_steps // K
    rem  = total_steps % K
    grid = []
    for k in range(K):
        n_k = base + (1 if k < rem else 0)
        if n_k == 0:
            continue
        i = torch.arange(n_k + 1, dtype=torch.float64)
        s = 0.5 - 0.5 * torch.cos(math.pi * i / n_k)
        s = eps + (1 - 2 * eps) * s
        nodes = (k + s).float()
        grid.append(nodes)
    zt = z0
    for nodes in grid:
        for i in range(len(nodes) - 1):
            tv = float(nodes[i])
            dtv = float(nodes[i + 1] - nodes[i])
            k1 = model.forward_scalar_t(zt, tv)
            k2 = model.forward_scalar_t(zt + 0.5 * dtv * k1, tv + 0.5 * dtv)
            k3 = model.forward_scalar_t(zt + 0.5 * dtv * k2, tv + 0.5 * dtv)
            k4 = model.forward_scalar_t(zt + dtv * k3, tv + dtv)
            zt = zt + (dtv / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return zt

print(f'{"steps":>6} | {"out_var":>12} | {"truth_var":>12} | {"|var_ratio-1|":>14} | {"band L2 err":>12}')
for steps in [num_masks, 2 * num_masks, 4 * num_masks, 8 * num_masks]:
    z0 = make_z0(n_test)
    out = rk4_clustered(oracle, z0, steps)
    out_var = out.var().item()
    err = band_l2_err(out.squeeze(1), test_data)
    print(f'{steps:>6} | {out_var:>12.4f} | {truth_var:>12.4f} | {abs(out_var/truth_var - 1):>14.4e} | {err:>12.4e}')

# ─── Designed schedule oracle ──────────────────────────────────────────────
# Paper's choice: σ²_k = λ_max(Sigma_FF|C) (noise ≥ every data mode), so the
# schedule monotonically decays noise to data with M_k = σ²/λ_min = cond.
print()
print('Designed-per-phase schedule (σ²_k = λ_max, M_k = cond number):')

phases_d = []
M_k_list = []
for ph in phases:
    Sigma_FF_C = ph['Sigma_FF_C']
    eigs = torch.symeig(Sigma_FF_C, eigenvectors=False)[0].numpy()
    eigs = eigs[eigs > 1e-12]
    if len(eigs) > 1:
        sigma2_d = float(eigs.max())
        # Paper convention: M = (target var)/(noise var) at the WORST mode, < 1
        Mk = float(eigs.min() / eigs.max())
    else:
        sigma2_d = float(eigs[0]) if len(eigs) > 0 else 1.0
        Mk = 1.0
    phases_d.append(dict(ph, sigma2=sigma2_d))
    M_k_list.append(Mk)
print('  M_k        =', [f'{m:.2e}' for m in M_k_list])
print('  σ²_k (new) =', [f'{p["sigma2"]:.2e}' for p in phases_d])

# Build noise std for the designed initial condition: per-band σ_k on band F.
noise_std_designed = torch.zeros(G, G)
for k, ph in enumerate(phases_d):
    scale_idx = num_masks - 1 - k
    band_2d = hier.masks[scale_idx].cpu()
    noise_std_designed += math.sqrt(ph['sigma2']) * band_2d

def make_z0_designed(B):
    eps = torch.randn(B, 1, G, G)
    return eps * noise_std_designed.unsqueeze(0).unsqueeze(0)

import sys as _sys; _sys.exit(0)  # designed-schedule path is float32-only; skip
oracle_d = DesignedOracleVelocity(hier, phases_d, M_k_list)

# Custom RK4 that clips s away from {0, 1} in EVERY phase (designed schedule
# is singular at the endpoints).
@torch.no_grad()
def rk4_designed(model, z0, total_steps, eps=0.02):
    K = model.num_masks
    base = total_steps // K
    rem  = total_steps % K
    grid = []
    for k in range(K):
        n_k = base + (1 if k < rem else 0)
        if n_k == 0:
            continue
        nodes = torch.linspace(k + eps, k + 1 - eps, n_k + 1)
        grid.append(nodes)
    zt = z0
    for nodes in grid:
        for i in range(len(nodes) - 1):
            tv = float(nodes[i])
            dtv = float(nodes[i + 1] - nodes[i])
            k1 = model.forward_scalar_t(zt, tv)
            k2 = model.forward_scalar_t(zt + 0.5 * dtv * k1, tv + 0.5 * dtv)
            k3 = model.forward_scalar_t(zt + 0.5 * dtv * k2, tv + 0.5 * dtv)
            k4 = model.forward_scalar_t(zt + dtv * k3, tv + dtv)
            zt = zt + (dtv / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return zt

print(f'{"steps":>6} | {"out_var":>12} | {"truth_var":>12} | {"|var_ratio-1|":>14} | {"band L2 err":>12}')
for steps in [num_masks, 2 * num_masks, 4 * num_masks, 8 * num_masks]:
    z0 = make_z0_designed(n_test)
    out = rk4_designed(oracle_d, z0, steps)
    out_var = out.var().item()
    err = band_l2_err(out.squeeze(1), test_data)
    print(f'{steps:>6} | {out_var:>12.4f} | {truth_var:>12.4f} | {abs(out_var/truth_var - 1):>14.4e} | {err:>12.4e}')


# Sanity: single very-fine step count
steps = 16 * num_masks
z0 = make_z0(n_test)
out = sampler.RK4(z0, oracle, steps=steps)
print()
print(f'High-resolution check (RK4 steps={steps}):')
print(f'  out var = {out.var().item():.4f}, truth var = {truth_var:.4f}')
print(f'  out std per band:')
for i, m in enumerate(hier.masks):
    n_k = int(m.sum().item())
    pix = (out.squeeze(1) * m.cpu()).reshape(n_test, -1)
    truth_pix = (test_data * m.cpu()).reshape(n_test, -1)
    print(f'    band {i} ({n_k:4d} pts): out_std={pix[pix != 0].std():.4f}  '
          f'truth_std={truth_pix[truth_pix != 0].std():.4f}')
