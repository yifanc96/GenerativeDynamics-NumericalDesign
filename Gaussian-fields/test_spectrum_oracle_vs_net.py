"""
Side-by-side energy spectrum E(k) for:
  (a) truth Matérn samples
  (b) the analytical mask oracle (K=6, mean conditional variance noise)
  (c) the trained multi-net mask checkpoint at results/mask_K6_multi_proper/

Same n_test, same RK4 step counts. Reports E(k) for each k bin and the
relative error per bin so we can see where errors live in frequency.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_multiscale_interpolation import (
    HierarchicalMasks, Sampler, MultiVelocity,
    precompute_matern_amplitude, sample_matern_batch,
    estimate_conditional_variances, build_noise_std,
    get_fourier_spectrum,
)

torch.manual_seed(0)
np.random.seed(0)

# ── Config (matches the trained model) ───────────────────────────────────────
G          = 64
s          = 3.0
ls         = 1.0
sigma_sq_o = 1.0
num_masks  = 6
n_test     = 1000
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path  = 'results/mask_K6_multi_proper/model_final.pt'

sigma_sq = sigma_sq_o * ((2 * math.pi) ** 2 + ls ** 2) ** s
amp = precompute_matern_amplitude(G, sigma_sq, ls, s)
truth = sample_matern_batch(amp, n_test, device='cpu')
print(f'[Data] G={G} s={s} device={device}  truth std={truth.std():.4f}')

# ── Build hier_masks and noise (same as trainer) ─────────────────────────────
hier = HierarchicalMasks(G, num_masks, device=device)
var_data = sample_matern_batch(amp, 2000, device='cpu')
cond_vars = estimate_conditional_variances(var_data, [m.cpu() for m in hier.masks])
noise_std = build_noise_std(hier.masks, cond_vars, device=device)
del var_data
print(f'[Cond vars] {[f"{v:.3e}" for v in cond_vars]}')

def make_z0(B):
    """Trainer-style z0: empirical cond_vars from ridge regression on 2000 samples."""
    z = torch.randn(B, 1, G, G, device=device)
    return z * noise_std[None, None, :, :]

def make_z0_analytic(B, phases):
    """Oracle-consistent z0: σ²_k from analytical Σ_FF|C — same value as the velocity formula uses."""
    z_flat = torch.zeros(B, G * G, device=device)
    for k, ph in enumerate(phases):
        F = ph['F']
        std_k = math.sqrt(ph['sigma2'])
        z_flat[:, F] = std_k * torch.randn(B, len(F), device=device)
    return z_flat.view(B, 1, G, G)

# ── Build pixel-basis Σ analytically and per-phase conditional structure ────
spec_np = (amp ** 2).numpy()
c2 = np.fft.ifft2(spec_np, norm='forward').real
ii = np.arange(G * G)
iy, ix = np.divmod(ii, G)
dy = (iy[:, None] - iy[None, :]) % G
dx = (ix[:, None] - ix[None, :]) % G
Sigma = torch.from_numpy(c2[dy, dx].astype(np.float64))
del dy, dx, ii, iy, ix
print(f'[Sigma] analytical Σ trace/d={Sigma.diag().mean().item():.4f}')

phases = []
for k in range(num_masks):
    scale_idx = num_masks - 1 - k
    band = hier.masks[scale_idx].cpu()
    F = torch.nonzero(band.flatten(), as_tuple=False).flatten()
    C_list = []
    for j in range(k):
        prev_si = num_masks - 1 - j
        C_list.append(torch.nonzero(hier.masks[prev_si].cpu().flatten(), as_tuple=False).flatten())
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
    Sigma_FF_C = 0.5 * (Sigma_FF_C + Sigma_FF_C.T)
    sigma2 = float(Sigma_FF_C.diag().mean().item())
    # Eigendecomp once: Σ_FF|C = U Λ U^T  →  Total = U (a²σ² + b²Λ) U^T
    e_all, U = torch.symeig(Sigma_FF_C, eigenvectors=True)
    e_all = torch.clamp(e_all, min=0.0)
    if M_op is not None:
        M_op = M_op.to(device)
    phases.append(dict(F=F.to(device), C=C.to(device) if len(C) > 0 else C,
                       M_op=M_op, U=U.to(device), eig=e_all.to(device),
                       sigma2=sigma2))
    print(f'  phase {k}: |F|={len(F):4d} |C|={len(C):4d} σ²_mean={sigma2:.3e}')

# ── Oracle velocity (same formula as test_oracle_mask.py) ───────────────────
import torch.nn as nn
class OracleVelocity(nn.Module):
    def __init__(self, hier, phases):
        super().__init__()
        self.hier = hier
        self.phases = phases
        self.num_masks = hier.num_masks
    def forward_scalar_t(self, zt, t_scalar):
        B, _, H, W = zt.shape
        k = self.hier.get_active_scale(t_scalar)
        s_loc = float(t_scalar) - k
        a, b = 1.0 - s_loc, s_loc
        ph = self.phases[k]
        F, C = ph['F'], ph['C']
        U = ph['U']; eig = ph['eig']; sigma2 = ph['sigma2']

        zt_flat = zt.view(B, -1).double()
        z_F = zt_flat[:, F]
        if len(C) > 0:
            z_C = zt_flat[:, C]
            mu = z_C @ ph['M_op'].T
        else:
            mu = torch.zeros_like(z_F)

        # Solve Total Xt = diff using eigendecomp of Σ_FF|C
        diff = z_F - b * mu                                  # (B, |F|)
        d_eig = (diff @ U)                                   # (B, |F|) — into eigenbasis
        denom = a * a * sigma2 + b * b * eig                 # (|F|,)
        Xt_eig = d_eig / denom                                # (B, |F|)
        # z1_hat = mu + b * Σ Xt = mu + b * U Λ Xt_eig (back to original)
        z1_hat = mu + b * (Xt_eig * eig) @ U.T
        z0_hat = (a * sigma2) * (Xt_eig @ U.T)
        v_F = z1_hat - z0_hat

        out = torch.zeros(B, H * W, dtype=torch.float64, device=zt.device)
        out[:, F] = v_F
        return out.view(B, 1, H, W).float()

oracle = OracleVelocity(hier, phases)
sampler = Sampler(num_masks)

# ── Load trained multi-net checkpoint ────────────────────────────────────────
print(f'[Net] loading {ckpt_path}')
net = MultiVelocity(hier, channels_per_net=24, dim_mults=(1, 2, 2)).to(device)
sd = torch.load(ckpt_path, map_location=device)
net.load_state_dict(sd)
net.eval()

# ── Run with FIXED sampler (per-phase integration) in both grid modes ──────
# IMPORTANT: oracle uses z0_analytic (consistent σ²) — net uses z0 (trainer-style)
z0          = make_z0(n_test)
z0_analytic = make_z0_analytic(n_test, phases)
NSTEPS = 24

def chunked_rk4(model, z0, steps, grid_mode, chunk=200):
    outs = []
    for i in range(0, z0.shape[0], chunk):
        out = sampler.RK4(z0[i:i+chunk].clone(), model, steps=steps, grid_mode=grid_mode)
        outs.append(out.cpu())
    return torch.cat(outs, dim=0)

with torch.no_grad():
    out_oracle      = chunked_rk4(oracle, z0_analytic, NSTEPS, 'cosine')
    out_oracle_uni  = chunked_rk4(oracle, z0_analytic, NSTEPS, 'uniform')
    out_net         = chunked_rk4(net,    z0,          NSTEPS, 'cosine')
    out_net_uni     = chunked_rk4(net,    z0,          NSTEPS, 'uniform')

# ── Energy spectrum E(k) ─────────────────────────────────────────────────────
truth_np     = truth.numpy()
oracle_np    = out_oracle.squeeze(1).cpu().numpy()
oracle_uninp = out_oracle_uni.squeeze(1).cpu().numpy()
net_np       = out_net.squeeze(1).cpu().numpy()
net_uninp    = out_net_uni.squeeze(1).cpu().numpy()

# Empirical truth E(k) from samples (for net comparison)
kvals, E_truth     = get_fourier_spectrum(truth_np)

# Analytic truth E(k) from the Matérn spec — same radial binning, no MC noise
def analytic_E_from_spec(spec_2d, G):
    from scipy.stats import binned_statistic
    freq = np.fft.fftfreq(G, d=1.0 / G)
    kx, ky = np.meshgrid(freq, freq)
    k_mag = np.sqrt(kx**2 + ky**2).flatten()
    p_flat = spec_2d.flatten()
    k_bins = np.arange(0.5, G // 2 + 1, 1.0)
    Abins, edges, _ = binned_statistic(k_mag, p_flat, statistic='mean', bins=k_bins)
    Abins_w = Abins * np.pi * (edges[1:]**2 - edges[:-1]**2)
    return Abins_w

E_truth_analytic = analytic_E_from_spec(spec_np, G)
_,     E_oracle    = get_fourier_spectrum(oracle_np)
_,     E_oracle_un = get_fourier_spectrum(oracle_uninp)
_,     E_net       = get_fourier_spectrum(net_np)
_,     E_net_un    = get_fourier_spectrum(net_uninp)

print()
print(f'Radial energy spectrum E(k)  ({NSTEPS} RK4 steps, FIXED per-phase sampler):')
print(f'{"k":>5}  {"truth":>11}  {"orcl_cos":>11}  {"orcl_uni":>11}  {"net_cos":>11}  {"net_uni":>11}  '
      f'{"net_cos/tr":>10}  {"net_uni/tr":>10}')
for i, k in enumerate(kvals):
    et = E_truth[i]; eo = E_oracle[i]; eou = E_oracle_un[i]
    en = E_net[i]; enu = E_net_un[i]
    rnc  = en  / (et + 1e-30)
    rnu  = enu / (et + 1e-30)
    print(f'{k:5.1f}  {et:11.3e}  {eo:11.3e}  {eou:11.3e}  {en:11.3e}  {enu:11.3e}  '
          f'{rnc:10.3f}  {rnu:10.3f}')

def relerr(eg, et, lo=None, hi=None):
    eg, et = eg[lo:hi], et[lo:hi]
    return np.mean(np.abs(eg - et) / (np.abs(et) + 1e-30))

print()
print('mean |E_gen / E_truth_EMPIRICAL - 1|:')
print(f'  oracle_cos        all_k = {relerr(E_oracle,    E_truth):.3e}   k≤30 = {relerr(E_oracle,    E_truth, hi=30):.3e}')
print(f'  oracle_uni        all_k = {relerr(E_oracle_un, E_truth):.3e}   k≤30 = {relerr(E_oracle_un, E_truth, hi=30):.3e}')
print(f'  net_cos           all_k = {relerr(E_net,       E_truth):.3e}   k≤30 = {relerr(E_net,       E_truth, hi=30):.3e}')
print(f'  net_uni           all_k = {relerr(E_net_un,    E_truth):.3e}   k≤30 = {relerr(E_net_un,    E_truth, hi=30):.3e}')

print()
print('mean |E_gen / E_truth_ANALYTIC - 1|  (no MC noise in truth):')
print(f'  truth_emp         all_k = {relerr(E_truth,     E_truth_analytic):.3e}   k≤30 = {relerr(E_truth, E_truth_analytic, hi=30):.3e}')
print(f'  oracle_cos        all_k = {relerr(E_oracle,    E_truth_analytic):.3e}   k≤30 = {relerr(E_oracle, E_truth_analytic, hi=30):.3e}')
print(f'  oracle_uni        all_k = {relerr(E_oracle_un, E_truth_analytic):.3e}   k≤30 = {relerr(E_oracle_un, E_truth_analytic, hi=30):.3e}')
print(f'  net_cos           all_k = {relerr(E_net,       E_truth_analytic):.3e}   k≤30 = {relerr(E_net, E_truth_analytic, hi=30):.3e}')
print(f'  net_uni           all_k = {relerr(E_net_un,    E_truth_analytic):.3e}   k≤30 = {relerr(E_net_un, E_truth_analytic, hi=30):.3e}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].loglog(kvals, E_truth,     'k-',   label='truth', lw=2)
ax[0].loglog(kvals, E_oracle,    'C0--', label='oracle (cos)', lw=2)
ax[0].loglog(kvals, E_oracle_un, 'C0:',  label='oracle (uni)', lw=1.5)
ax[0].loglog(kvals, E_net,       'C3-',  label='net (cos)', lw=1.5)
ax[0].loglog(kvals, E_net_un,    'C3--', label='net (uni)', lw=1.5)
ax[0].set_xlabel('k'); ax[0].set_ylabel('E(k)'); ax[0].legend(fontsize=9)
ax[0].set_title(f'energy spectrum, {NSTEPS} RK4 steps')

ax[1].loglog(kvals, np.abs(E_oracle    - E_truth) / (E_truth + 1e-30), 'C0--', label='oracle (cos)')
ax[1].loglog(kvals, np.abs(E_oracle_un - E_truth) / (E_truth + 1e-30), 'C0:',  label='oracle (uni)')
ax[1].loglog(kvals, np.abs(E_net       - E_truth) / (E_truth + 1e-30), 'C3-',  label='net (cos)')
ax[1].loglog(kvals, np.abs(E_net_un    - E_truth) / (E_truth + 1e-30), 'C3--', label='net (uni)')
ax[1].set_xlabel('k'); ax[1].set_ylabel('|E_gen / E_truth - 1|'); ax[1].legend(fontsize=9)
ax[1].set_title('per-bin relative error')

plt.tight_layout()
out_path = 'results/spectrum_oracle_vs_net.png'
plt.savefig(out_path, dpi=140)
print(f'[plot] saved {out_path}')

# ── Oracle step-count convergence study ─────────────────────────────────────
print()
print('=== Oracle convergence with RK4 step count (vs analytic truth) ===')
print(f'{"steps":>6}  {"all k":>10}  {"k≤30":>10}  {"k≤24":>10}  {"k=31,32 ratio":>14}')
for st in [12, 24, 48, 96, 240]:
    out_n = chunked_rk4(oracle, z0_analytic, st, 'cosine')
    _, E = get_fourier_spectrum(out_n.squeeze(1).cpu().numpy())
    err_all = relerr(E, E_truth_analytic)
    err_30  = relerr(E, E_truth_analytic, hi=30)
    err_24  = relerr(E, E_truth_analytic, hi=24)
    nyq_ratio = float(np.mean(E[30:32] / (E_truth_analytic[30:32] + 1e-30)))
    print(f'{st:6d}  {err_all:10.3e}  {err_30:10.3e}  {err_24:10.3e}  {nyq_ratio:14.3f}')
