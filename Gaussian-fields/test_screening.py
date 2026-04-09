"""
Test the screening effect from scratch:
For a Matern field on G x G, take a coarse subset C (every other pixel)
and the fine subset F = complement. Compute the conditional covariance
of x_F | x_C and check its condition number.

Per Schaefer-Owhadi-Chen (Gamblets / sparse Cholesky), this should be
well-conditioned even for high smoothness s.
"""
import sys, os, math, numpy as np
import torch
import torch.fft as torch_fft

torch.manual_seed(42)

# ─── Step 1: Generate 1D Matern field, compute Sigma in pixel basis ─────────
def gen_1d_matern(L, s, sigma_factor=1.0, n=5000):
    sigma_sq = sigma_factor * ((2*math.pi)**2 + 1.0)**s
    freq = torch.from_numpy(np.fft.fftfreq(L)).float() * 2 * math.pi * L
    spec = sigma_sq * (freq**2 + 1.0)**(-s)
    spec[0] = 0
    amp = spec.sqrt()
    noise = torch.complex(torch.randn(n, L), torch.randn(n, L))
    field = torch_fft.ifftn(amp.unsqueeze(0) * noise, dim=-1, norm='forward').real
    return field, spec

def get_pixel_cov(data):
    """Empirical pixel-space covariance."""
    flat = data.reshape(data.shape[0], -1)
    cent = flat - flat.mean(0, keepdim=True)
    return cent.T @ cent / data.shape[0]

# ─── Step 2: Compute conditional covariance ──────────────────────────────────
def conditional_cov(Sigma, F_idx, C_idx):
    """
    Compute Cov(x_F | x_C) = Sigma_FF - Sigma_FC @ Sigma_CC^{-1} @ Sigma_CF
    """
    Sigma_FF = Sigma[F_idx][:, F_idx]
    Sigma_FC = Sigma[F_idx][:, C_idx]
    Sigma_CC = Sigma[C_idx][:, C_idx]
    # solve Sigma_CC @ X = Sigma_CF (where Sigma_CF = Sigma_FC.T)
    Sigma_CF = Sigma[C_idx][:, F_idx]
    X, _ = torch.solve(Sigma_CF, Sigma_CC + 1e-8 * torch.eye(len(C_idx)))
    return Sigma_FF - Sigma_FC @ X

def cond_number(M):
    eigs = torch.symeig(M, eigenvectors=False)[0]
    pos = eigs[eigs > 1e-12].numpy()
    if len(pos) < 2: return 1.0
    return pos.max() / pos.min()

# ─── Step 3: Screening test for 1D Matern ───────────────────────────────────
print("=" * 80)
print("1D Matern: condition number of conditional covariances at multiple scales")
print("=" * 80)

for s in [1.5, 3.0]:
    print()
    print("s = %.1f:" % s)

    L = 64
    data, spec = gen_1d_matern(L, s, sigma_factor=1.0, n=10000)
    Sigma = get_pixel_cov(data)
    print("  Full Sigma cond: %.2e" % cond_number(Sigma))

    # Hierarchical decomposition: at level k, coarse = every 2^k pixel
    # Fine at level k = pixels at stride 2^(k-1) but not 2^k
    print("  Within-band (fine | coarser) condition numbers:")
    for level in range(1, int(math.log2(L))):
        stride = 2 ** level
        # Coarse pixels at this level: indices divisible by stride
        coarse_idx = torch.tensor([i for i in range(L) if i % stride == 0])
        # Fine pixels at this level: divisible by stride/2 but NOT by stride
        prev_stride = 2 ** (level - 1)
        fine_idx = torch.tensor([i for i in range(L) if (i % prev_stride == 0) and (i % stride != 0)])

        if len(fine_idx) == 0 or len(coarse_idx) == 0: continue

        # Conditional cov of fine | coarse
        C_cond = conditional_cov(Sigma, fine_idx, coarse_idx)
        cond = cond_number(C_cond)
        print("    level %d: %d fine | %d coarse → cond = %.2e" % (
            level, len(fine_idx), len(coarse_idx), cond))

    # Also: what's the condition of the cov of just the fine pixels (no conditioning)?
    print("  Within-band (fine, NOT conditioned) condition numbers:")
    for level in range(1, int(math.log2(L))):
        stride = 2 ** level
        prev_stride = 2 ** (level - 1)
        fine_idx = torch.tensor([i for i in range(L) if (i % prev_stride == 0) and (i % stride != 0)])
        if len(fine_idx) == 0: continue
        block = Sigma[fine_idx][:, fine_idx]
        print("    level %d: %d fine pixels, marginal cond = %.2e" % (
            level, len(fine_idx), cond_number(block)))
