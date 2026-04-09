"""
Oracle test for the wavelet-based multiscale interpolant (analogous to
test_oracle_mask.py but using a wavelet decomposition instead of pixel masks).

We use a 2-D orthonormal wavelet transform W with K levels. Sub-bands form a
hierarchy: phase 0 = LL (coarsest), phase k = LH/HL/HH at level (K-k+1).
Per-band conditional noise variance comes from the diagonal of the empirical
covariance in coefficient space (since W is orthonormal, joint conditioning
across bands is automatic — coefficients are well-decorrelated by the wavelet).

Result: linear schedule + cosine-clustered time grid per phase, ~20 RK4 total
should match the paper's quality.
"""
import math, sys, os
import numpy as np
import torch
import torch.nn as nn
import pywt

from train_multiscale_interpolation import (
    precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)

torch.manual_seed(0)
np.random.seed(0)

# ─── Config ────────────────────────────────────────────────────────────────
G          = 64
s          = 3.0
ls         = 1.0
sigma_sq_o = 1.0
K_levels   = 5
wavelet    = 'db4'
n_emp      = 10000
n_test     = 500
device     = torch.device('cpu')

num_phases = K_levels + 1   # phase 0 = LL, phases 1..K = (LH,HL,HH) at each level
print(f'[Config] G={G} s={s} wavelet={wavelet} K_levels={K_levels} num_phases={num_phases}')

# ─── Data + ANALYTICAL pixel covariance (circulant from Matérn spec) ───────
sigma_sq = sigma_sq_o * ((2 * math.pi) ** 2 + ls ** 2) ** s
amp = precompute_matern_amplitude(G, sigma_sq, ls, s)
spec = (amp ** 2).numpy()
test_data = sample_matern_batch(amp, n_test, device=device)
print(f'[Data] truth pix var={(test_data**2).mean().item():.4f}')

# Σ_pix is circulant; build from autocorrelation
c2 = np.fft.ifft2(spec, norm='forward').real
ii = np.arange(G * G); iy, ix = np.divmod(ii, G)
dy = (iy[:, None] - iy[None, :]) % G
dx = (ix[:, None] - ix[None, :]) % G
Sigma_pix = c2[dy, dx].astype(np.float64)
del dy, dx, ii, iy, ix
print(f'[Sigma] analytical Σ_pix shape={Sigma_pix.shape} '
      f'trace/d={np.diag(Sigma_pix).mean():.4f}')

# ─── Wavelet transform via pywt ────────────────────────────────────────────
# We'll keep coefficients in the form returned by pywt.wavedec2:
# coeffs = [LL, (LH_K, HL_K, HH_K), ..., (LH_1, HL_1, HH_1)]
# We work with a flattened "coeff vector" of length G*G with a fixed slot map.

# Build slot map by performing wavedec2 on a dummy.
dummy = np.zeros((G, G))
coeffs_template = pywt.wavedec2(dummy, wavelet=wavelet, level=K_levels, mode='periodization')
slots = []          # list of (name, slice, shape) entries
offset = 0
def add_slot(name, arr):
    global offset
    n = arr.size
    slots.append((name, slice(offset, offset + n), arr.shape))
    offset += n

add_slot('LL', coeffs_template[0])
for j, det in enumerate(coeffs_template[1:]):
    level_idx = K_levels - j        # detail level (K_levels = coarsest, 1 = finest)
    LH, HL, HH = det
    add_slot(f'LH_{level_idx}', LH)
    add_slot(f'HL_{level_idx}', HL)
    add_slot(f'HH_{level_idx}', HH)

D = offset
print(f'[Wavelet] coefficient vector length = {D} (should be {G*G})')
assert D == G * G

def to_coeffs(x_2d_np):
    """Forward wavelet to flat coefficient vector."""
    cs = pywt.wavedec2(x_2d_np, wavelet=wavelet, level=K_levels, mode='periodization')
    out = np.zeros(D)
    out[slots[0][1]] = cs[0].ravel()
    idx = 1
    for det in cs[1:]:
        for sub in det:
            out[slots[idx][1]] = sub.ravel()
            idx += 1
    return out

def from_coeffs(c_flat_np):
    """Inverse wavelet from flat coefficient vector."""
    LL = c_flat_np[slots[0][1]].reshape(slots[0][2])
    cs = [LL]
    idx = 1
    for j in range(K_levels):
        LH = c_flat_np[slots[idx][1]].reshape(slots[idx][2]); idx += 1
        HL = c_flat_np[slots[idx][1]].reshape(slots[idx][2]); idx += 1
        HH = c_flat_np[slots[idx][1]].reshape(slots[idx][2]); idx += 1
        cs.append((LH, HL, HH))
    return pywt.waverec2(cs, wavelet=wavelet, mode='periodization')

# Sanity: round-trip
samp = test_data[0].numpy()
rt = from_coeffs(to_coeffs(samp))
print(f'[Wavelet] round-trip max err = {np.max(np.abs(samp - rt)):.2e}')

# ─── Build phase index lists in coefficient space ───────────────────────────
# phase 0 = LL slot
# phase k for k=1..K_levels = the (LH_l, HL_l, HH_l) slots with l = K_levels - k + 1
#   so phase 1 → coarsest details (level K_levels), phase K → finest details (level 1)
phase_idx = [None] * num_phases
phase_idx[0] = list(range(slots[0][1].start, slots[0][1].stop))
for k in range(1, num_phases):
    level = K_levels - k + 1
    idx_list = []
    for sub in ['LH', 'HL', 'HH']:
        name = f'{sub}_{level}'
        for slot_name, slot_slice, _ in slots:
            if slot_name == name:
                idx_list.extend(range(slot_slice.start, slot_slice.stop))
    phase_idx[k] = idx_list

for k, idxs in enumerate(phase_idx):
    print(f'  phase {k}: {len(idxs):4d} coeffs')

# ─── Build Sigma in WAVELET basis = W @ Sigma_pix @ W^T (analytical) ───────
# W is the orthonormal DWT matrix; we materialize it column-by-column by
# transforming each canonical pixel basis vector.
print('[Sigma] building W (DWT matrix) column by column...')
W = np.zeros((D, D))
e = np.zeros((G, G))
for i in range(D):
    iy, ix = divmod(i, G)
    e[iy, ix] = 1.0
    W[:, i] = to_coeffs(e)
    e[iy, ix] = 0.0
print('[Sigma] applying W Σ_pix W^T ...')
Sigma_w = W @ Sigma_pix @ W.T
Sigma_t = torch.from_numpy(Sigma_w).double()
print(f'[Sigma] wavelet-basis Σ shape={Sigma_w.shape} '
      f'trace/d={np.diag(Sigma_w).mean():.4f}')

# ─── Build per-phase conditional structure ─────────────────────────────────
phases = []
for k in range(num_phases):
    F = torch.tensor(phase_idx[k], dtype=torch.long)
    C_idx = []
    for j in range(k):
        C_idx.extend(phase_idx[j])
    C_idx = torch.tensor(C_idx, dtype=torch.long) if len(C_idx) > 0 else torch.empty(0, dtype=torch.long)

    Sigma_FF = Sigma_t[F][:, F]
    if len(C_idx) > 0:
        Sigma_FC = Sigma_t[F][:, C_idx]
        Sigma_CC = Sigma_t[C_idx][:, C_idx]
        ridge = 1e-12 * Sigma_CC.diag().mean().item()
        M_op = torch.solve(Sigma_FC.T, Sigma_CC + ridge * torch.eye(len(C_idx), dtype=Sigma_CC.dtype))[0].T
        Sigma_FF_C = Sigma_FF - M_op @ Sigma_FC.T
    else:
        M_op = None
        Sigma_FF_C = Sigma_FF.clone()
    Sigma_FF_C = 0.5 * (Sigma_FF_C + Sigma_FF_C.T)

    e_all = torch.symeig(Sigma_FF_C, eigenvectors=False)[0].numpy()
    e_pos = e_all[e_all > 0]
    floor = max(e_pos.max() * 1e-10, 0.0) if len(e_pos) else 0.0
    eigs = e_pos[e_pos > floor]
    cn = float(eigs.max() / eigs.min()) if len(eigs) > 1 else 1.0
    # geometric-mean noise variance
    sigma2 = float(math.sqrt(eigs.min() * eigs.max())) if len(eigs) > 1 else float(eigs[0])
    print(f'  phase {k}: |F|={len(F):4d} |C|={len(C_idx):4d}  '
          f'cond={cn:.2e}  σ²_geom={sigma2:.3e}  Lip~cond^¼≈{cn**0.25:.2f}')

    phases.append(dict(F=F, C=C_idx, M_op=M_op, Sigma_FF_C=Sigma_FF_C, sigma2=sigma2))


# ─── Oracle velocity: iid scalar noise σ²_k I (geometric mean), linear ─────
class WaveletOracleVelocity(nn.Module):
    def __init__(self, phases, num_phases):
        super().__init__()
        self.phases = phases
        self.num_phases = num_phases

    def get_active_scale(self, t_scalar):
        return int(min(max(t_scalar, 0), self.num_phases - 1))

    def forward_scalar_t(self, zt_c, t_scalar):
        B = zt_c.shape[0]
        k = self.get_active_scale(t_scalar)
        s = float(t_scalar) - k
        a, b = 1.0 - s, s

        ph = self.phases[k]
        F, C = ph['F'], ph['C']
        Sigma_FF_C = ph['Sigma_FF_C']
        sigma2 = ph['sigma2']

        z_F = zt_c[:, F]
        if len(C) > 0:
            z_C = zt_c[:, C]
            mu = z_C @ ph['M_op'].T
        else:
            mu = torch.zeros_like(z_F)

        nF = len(F)
        Total = (a * a * sigma2) * torch.eye(nF, dtype=Sigma_FF_C.dtype) + (b * b) * Sigma_FF_C
        diff = (z_F - b * mu).T
        Xt = torch.solve(diff, Total)[0]
        z1_hat = mu + b * (Sigma_FF_C @ Xt).T
        z0_hat = (a * sigma2) * Xt.T
        v_F = z1_hat - z0_hat

        out = torch.zeros(B, zt_c.shape[1], dtype=zt_c.dtype)
        out[:, F] = v_F
        return out


oracle = WaveletOracleVelocity(phases, num_phases)


# ─── Initial noise: iid scalar per band, std = √sigma2 ─────────────────────
def make_z0_coeffs(B):
    out = torch.zeros(B, D, dtype=torch.float64)
    for k, ph in enumerate(phases):
        F = ph['F']
        out[:, F] = math.sqrt(ph['sigma2']) * torch.randn(B, len(F), dtype=torch.float64)
    return out


# ─── Integrators ────────────────────────────────────────────────────────────
@torch.no_grad()
def rk4_uniform(model, z0, total_steps, eps=1e-3):
    K = model.num_phases
    base = total_steps // K
    rem = total_steps % K
    grid = []
    for k in range(K):
        n_k = base + (1 if k < rem else 0)
        if n_k == 0:
            continue
        lo = k + (eps if k == 0 else 0.0)
        hi = k + 1 - (eps if k == K - 1 else 0.0)
        nodes = torch.linspace(lo, hi, n_k + 1)
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

@torch.no_grad()
def rk4_clustered(model, z0, total_steps, eps=1e-3):
    K = model.num_phases
    base = total_steps // K
    rem = total_steps % K
    grid = []
    for k in range(K):
        n_k = base + (1 if k < rem else 0)
        if n_k == 0:
            continue
        i = torch.arange(n_k + 1, dtype=torch.float64)
        s_nodes = 0.5 - 0.5 * torch.cos(math.pi * i / n_k)
        s_nodes = eps + (1 - 2 * eps) * s_nodes
        nodes = (k + s_nodes).float()
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


# ─── Helpers ───────────────────────────────────────────────────────────────
def coeffs_to_pixels(zt_c):
    """Inverse wavelet for a batch of coefficient vectors."""
    out = np.zeros((zt_c.shape[0], G, G))
    arr = zt_c.numpy()
    for i in range(zt_c.shape[0]):
        out[i] = from_coeffs(arr[i])
    return torch.from_numpy(out).float()

def band_l2_err(pix, truth):
    kv, Po = get_fourier_spectrum(pix.numpy())
    kv, Pt = get_fourier_spectrum(truth.numpy())
    err = np.abs(Po - Pt) / (np.abs(Pt) + 1e-30)
    return float(np.mean(err))


truth_var = test_data.var().item()


print()
print('Oracle integration test, UNIFORM grid (linear schedule):')
print(f'{"steps":>6} | {"out_var":>12} | {"truth_var":>12} | {"|var_ratio-1|":>14} | {"band L2 err":>12}')
for steps in [num_phases, 2 * num_phases, 4 * num_phases, 8 * num_phases]:
    z0 = make_z0_coeffs(n_test)
    out_c = rk4_uniform(oracle, z0, steps)
    out_pix = coeffs_to_pixels(out_c)
    out_var = out_pix.var().item()
    err = band_l2_err(out_pix, test_data)
    print(f'{steps:>6} | {out_var:>12.4f} | {truth_var:>12.4f} | {abs(out_var/truth_var - 1):>14.4e} | {err:>12.4e}')

print()
print('Oracle integration test, COSINE-CLUSTERED grid (linear schedule):')
print(f'{"steps":>6} | {"out_var":>12} | {"truth_var":>12} | {"|var_ratio-1|":>14} | {"band L2 err":>12}')
for steps in [num_phases, 2 * num_phases, 4 * num_phases, 8 * num_phases]:
    z0 = make_z0_coeffs(n_test)
    out_c = rk4_clustered(oracle, z0, steps)
    out_pix = coeffs_to_pixels(out_c)
    out_var = out_pix.var().item()
    err = band_l2_err(out_pix, test_data)
    print(f'{steps:>6} | {out_var:>12.4f} | {truth_var:>12.4f} | {abs(out_var/truth_var - 1):>14.4e} | {err:>12.4e}')
