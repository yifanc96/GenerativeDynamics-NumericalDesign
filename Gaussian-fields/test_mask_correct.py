"""
Verify the mask-based multiscale gives bounded Lip per band when using
the joint conditional velocity (not just per-band marginal).

The key: at each phase k, the active band's velocity is computed CONDITIONAL
on the resolved coarser bands (which contain data values). The conditional
covariance has bounded condition number per the screening effect.
"""
import sys, os, math, numpy as np
import torch
import torch.fft as torch_fft

torch.manual_seed(42)

def gen_2d_matern(G, s, sigma_factor=1.0, n=2000):
    sigma_sq = sigma_factor * ((2*math.pi)**2 + 1.0)**s
    freq = torch.from_numpy(np.fft.fftfreq(G)).float() * 2 * math.pi * G
    fx, fy = torch.meshgrid(freq, freq)
    spec = sigma_sq * (fx**2 + fy**2 + 1.0)**(-s)
    spec[0, 0] = 0
    amp = spec.sqrt()
    noise = torch.complex(torch.randn(n, G, G), torch.randn(n, G, G))
    return torch_fft.ifftn(amp.unsqueeze(0) * noise, dim=(-2,-1), norm='forward').real

G = 32; s = 3.0
data = gen_2d_matern(G, s=s, sigma_factor=1.0, n=5000)
flat = data.reshape(5000, -1)
cent = flat - flat.mean(0, keepdim=True)
Sigma = cent.T @ cent / 5000
d = G * G

# Build hierarchical pixel masks: phase k generates pixels at level k
# Coarsest = level K-1 (single pixel at corner)
# At level k, generate pixels at stride 2^(K-1-k) but not 2^(K-k)
# (k=0 = coarsest, k=K-1 = finest)
K = 5  # 5 levels
band_indices = []  # band_indices[k] = list of pixel indices at level k
for k in range(K):
    level = K - 1 - k  # level 0 = finest (k=K-1), level K-1 = coarsest (k=0)
    stride = 2 ** level
    next_stride = 2 ** (level + 1) if level < K - 1 else 2 ** K
    # Wait, this isn't quite right. Let me think again.
    # Coarsest band = pixels at stride 2^(K-1): just (0,0), (0, 2^(K-1)), etc.
    # Next band = at stride 2^(K-2) but not at stride 2^(K-1)
    # ...
    # Finest = at stride 1 (any) but not at stride 2: half the pixels

# Let me redo: phase k (k=0 coarsest) corresponds to band at level K-1-k
# At "wavelet level" l (l=0 finest, l=K-1 coarsest):
#   - Pixels at stride 2^l but NOT at stride 2^(l+1)  (for l < K-1)
#   - Pixels at stride 2^(K-1) (just the coarsest)  (for l = K-1)

phase_pixel_idx = {}  # phase k → list of pixel indices
for k in range(K):
    level = K - 1 - k  # phase k = level (K-1-k)
    pixels = []
    if level == K - 1:
        # coarsest: stride 2^(K-1)
        stride = 2 ** level
        for i in range(0, G, stride):
            for j in range(0, G, stride):
                pixels.append(i * G + j)
    else:
        stride = 2 ** level
        next_stride = 2 ** (level + 1)
        for i in range(0, G, stride):
            for j in range(0, G, stride):
                if i % next_stride != 0 or j % next_stride != 0:
                    pixels.append(i * G + j)
    phase_pixel_idx[k] = pixels
    print('Phase %d (level %d): %d pixels' % (k, level, len(pixels)))

total = sum(len(v) for v in phase_pixel_idx.values())
print('Total: %d (should be %d)' % (total, d))

print()
print("Per-band conditional condition number (the right multiscale structure):")
print()

# For each phase k, compute the conditional covariance:
# Cov(x_band_k | x_resolved_bands)
# where resolved bands are k' < k (coarser)

for k in range(K):
    band_idx = torch.tensor(phase_pixel_idx[k])
    if len(band_idx) < 2: continue

    # Resolved (coarser) bands: phases 0..k-1
    resolved_idx = []
    for k_prev in range(k):
        resolved_idx.extend(phase_pixel_idx[k_prev])

    if len(resolved_idx) == 0:
        # Phase 0: marginal (no coarser bands)
        block = Sigma[band_idx][:, band_idx]
        cond = torch.linalg.eigvalsh(block)
        pos = cond[cond > 1e-12].numpy()
        cond_num = pos.max() / pos.min() if len(pos) > 1 else 1.0
        print('  phase %d: %d pixels, NO conditioning (coarsest), cond = %.2e' % (k, len(band_idx), cond_num))
    else:
        resolved_t = torch.tensor(resolved_idx)
        Sigma_FF = Sigma[band_idx][:, band_idx]
        Sigma_FC = Sigma[band_idx][:, resolved_t]
        Sigma_CC = Sigma[resolved_t][:, resolved_t]
        Sigma_CF = Sigma[resolved_t][:, band_idx]

        # Cov(F | C) = Sigma_FF - Sigma_FC @ Sigma_CC^{-1} @ Sigma_CF
        try:
            X = torch.linalg.solve(Sigma_CC + 1e-8 * torch.eye(len(resolved_idx)), Sigma_CF)
            cond_cov = Sigma_FF - Sigma_FC @ X
            eigs = torch.linalg.eigvalsh(cond_cov)
            pos = eigs[eigs > 1e-12].numpy()
            cond_num = pos.max() / pos.min() if len(pos) > 1 else 1.0

            # Also compute Lip using mean variance as scalar noise
            nv_mean = pos.mean()
            tg = np.linspace(0.001, 0.999, 200)
            max_lip = 0
            for t in tg:
                a, b = 1-t, t
                gains = ((-a)*nv_mean + b*pos) / (a*a*nv_mean + b*b*pos + 1e-30)
                max_lip = max(max_lip, np.max(np.abs(gains)))

            print('  phase %d: %d pixels | %d coarse → cond = %.2e, Lip(mean nv) = %.2f, log(cond)/2 = %.2f' % (
                k, len(band_idx), len(resolved_idx), cond_num, max_lip, math.log(cond_num)/2))
        except Exception as e:
            print('  phase %d: FAILED %s' % (k, str(e)[:60]))
