"""Oracle (analytic Gaussian drift) for bands 0-2, eval per-bin at R=32.
Uses the same proper_schur logic as test_oracle_exact.py.
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.fft as torch_fft

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)


def proper_schur(F, C, Sigma, ridge=1e-6):
    """Schur complement with explicit ridge=1e-6 on Sigma_CC (matches trainer)."""
    Sigma_FF = Sigma[F][:, F]
    if len(C) == 0:
        S = 0.5 * (Sigma_FF + Sigma_FF.T)
        return S, None
    Sigma_FC = Sigma[F][:, C]
    Sigma_CC = Sigma[C][:, C]
    Sigma_CC_reg = Sigma_CC + ridge * torch.eye(len(C), device=Sigma.device, dtype=Sigma.dtype)
    M_op = torch.linalg.solve(Sigma_CC_reg, Sigma_FC.T).T  # |F|×|C|
    S = Sigma_FF - M_op @ Sigma_FC.T
    S = 0.5 * (S + S.T)
    return S, M_op


def oracle_velocity(zt_F, z_C, M_op, S, sigma2, s):
    a, b = 1.0 - s, s
    nF = zt_F.shape[1]
    mu = (z_C @ M_op.T) if M_op is not None else torch.zeros_like(zt_F)
    Total = (a**2 * sigma2) * torch.eye(nF, device=zt_F.device) + (b**2) * S
    Xt = torch.linalg.solve(Total, (zt_F - b * mu).T)
    z1_hat = mu + b * (S @ Xt).T
    z0_hat = (a * sigma2) * Xt.T
    return z1_hat - z0_hat


def rk4_band(z_C, M_op, S, sigma2, nF, n_rk4, n_eval, device, eps=1e-3):
    zt = math.sqrt(sigma2) * torch.randn(n_eval, nF, device=device, dtype=torch.float64)
    nodes = torch.linspace(eps, 1.0 - eps, n_rk4 + 1)
    for i in range(len(nodes) - 1):
        s = float(nodes[i]); ds = float(nodes[i + 1] - nodes[i])
        zC = z_C if z_C is not None else torch.zeros(n_eval, 0, device=device)
        k1 = oracle_velocity(zt, zC, M_op, S, sigma2, s)
        k2 = oracle_velocity(zt + .5 * ds * k1, zC, M_op, S, sigma2, s + .5 * ds)
        k3 = oracle_velocity(zt + .5 * ds * k2, zC, M_op, S, sigma2, s + .5 * ds)
        k4 = oracle_velocity(zt + ds * k3, zC, M_op, S, sigma2, s + ds)
        zt = zt + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return zt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--n_eval', type=int, default=2000)
    p.add_argument('--n_rk4', type=int, default=4)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G, K = args.G, args.K

    sigma_sq = 1.0 * ((2 * math.pi)**2 + 1.0**2)**args.s
    amp = precompute_matern_amplitude(G, sigma_sq, 1.0, args.s).float()

    # Truth samples
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()

    # Analytic Sigma from amp^2 spectrum (exact, not empirical)
    spec = amp**2
    c2 = np.fft.ifft2(spec.numpy(), norm='forward').real
    ii = np.arange(G * G); iy, ix = np.divmod(ii, G)
    dy = (iy[:, None] - iy[None, :]) % G
    dx = (ix[:, None] - ix[None, :]) % G
    Sigma = torch.from_numpy(c2[dy, dx]).double().to(device)
    del dy, dx, ii, iy, ix

    # Build mask phases (same convention as trainer)
    hier = HierarchicalMasks(G, K + 1, device='cpu')
    bands = []
    for k in range(K + 1):
        si = K - k
        F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten().to(device)
        Cl = [torch.nonzero(hier.masks[K - j].cpu().flatten()).flatten() for j in range(k)]
        C = torch.cat(Cl).to(device) if Cl else torch.empty(0, dtype=torch.long, device=device)
        S, M_op = proper_schur(F, C, Sigma)
        sigma2 = float(S.diag().mean())
        bands.append(dict(F=F, C=C, S=S, M_op=M_op, sigma2=sigma2, R=G // (2**(K - k))))
        print(f"  band {k}: |F|={len(F)}  |C|={len(C)}  sigma²={sigma2:.3e}")

    # Generate up to band 2
    gen = torch.zeros(args.n_eval, G * G, device=device, dtype=torch.float64)
    for bi in [0, 1, 2]:
        b = bands[bi]
        F = b['F']; C = b['C']
        nF = len(F); S = b['S']; M_op = b['M_op']; sigma2 = b['sigma2']
        z_C = gen[:, C] if len(C) > 0 else None
        gen_F = rk4_band(z_C, M_op, S, sigma2, nF, args.n_rk4, args.n_eval, device)
        gen[:, F] = gen_F

    stride = G // 32
    gen_R = gen.view(args.n_eval, G, G)[:, ::stride, ::stride].cpu().numpy()
    tru_R = test_data[:, ::stride, ::stride].numpy()
    _, sg = get_fourier_spectrum(gen_R)
    _, st = get_fourier_spectrum(tru_R)
    rel = np.abs(sg - st) / (np.abs(st) + 1e-30)

    print(f"\n=== ORACLE bands 0-2, RK4={args.n_rk4}/band, NFE={args.n_rk4*4*3} ===")
    print(f"Per-bin at R=32:")
    print(f"  k=01..16: " + " ".join(f"{r:.3f}" for r in rel[:16]))
    print(f"  mean(k=1..15)={rel[:15].mean():.4f}  max={rel[:15].max():.4f}  argmax k={1+int(np.argmax(rel[:15]))}")
    print(f"  k=16 (Nyquist): {rel[15]:.3f}")


if __name__ == '__main__':
    main()
