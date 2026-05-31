"""
Sample from the 2D lattice φ⁴ model with Ising-like coupling on GPU.

Energy:
  E(φ) = -J Σ_{nn} φ_u φ_v + Σ_u (a φ_u² + b φ_u⁴)

With J=1, a=-0.5, b=0.25: the on-site potential V(φ) = -0.5φ² + 0.25φ⁴
has minima at φ=±1. With strong coupling J=1, the system orders into
one of the two wells, producing bimodal marginals and very ill-conditioned
spectrum (κ ~ 10⁴).

This parameterization gives:
  - Bimodal (non-Gaussian) distribution with kurtosis ≈ -1.9
  - Spectral condition number κ ≈ 9000+ at L=64
  - std ≈ 2.2
"""

import os, argparse
import numpy as np
import torch
from time import time


def compute_energy(x, J, a, b):
    """
    E = -J Σ_nn φ_u φ_v + Σ_u (a φ_u² + b φ_u⁴)
    x: (B, L, L), returns: (B,)
    """
    nn = x * torch.roll(x, -1, 1) + x * torch.roll(x, -1, 2)
    E_nn = -J * nn.sum(dim=(1, 2))
    E_site = (a * x**2 + b * x**4).sum(dim=(1, 2))
    return E_nn + E_site


def compute_force(x, J, a, b):
    """
    -dE/dφ = J Σ_nn φ_v - 2a φ - 4b φ³
    x: (B, L, L), returns: (B, L, L)
    """
    nn_sum = (torch.roll(x, 1, 1) + torch.roll(x, -1, 1) +
              torch.roll(x, 1, 2) + torch.roll(x, -1, 2))
    return J * nn_sum - 2 * a * x - 4 * b * x**3


def hmc_step(x, J, a, b, dt, n_leapfrog):
    """One HMC step for a batch of configurations."""
    p = torch.randn_like(x)
    x_old = x.clone()
    H_old = compute_energy(x, J, a, b) + 0.5 * p.pow(2).sum(dim=(1, 2))

    x_new = x.clone()
    force = compute_force(x_new, J, a, b)
    p = p + 0.5 * dt * force
    for _ in range(n_leapfrog - 1):
        x_new = x_new + dt * p
        force = compute_force(x_new, J, a, b)
        p = p + dt * force
    x_new = x_new + dt * p
    force = compute_force(x_new, J, a, b)
    p = p + 0.5 * dt * force

    H_new = compute_energy(x_new, J, a, b) + 0.5 * p.pow(2).sum(dim=(1, 2))
    dH = H_new - H_old

    accept = (dH < 0) | (torch.rand(x.shape[0], device=x.device) < torch.exp(-dH.clamp(max=50)))
    x_out = torch.where(accept[:, None, None], x_new, x_old)
    return x_out, accept


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--J', type=float, default=1.0)
    p.add_argument('--a', type=float, default=-0.5)
    p.add_argument('--b', type=float, default=0.25)
    p.add_argument('--n_samples', type=int, default=20000)
    p.add_argument('--n_chains', type=int, default=500)
    p.add_argument('--n_thermalize', type=int, default=3000)
    p.add_argument('--n_between', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--n_leapfrog', type=int, default=30)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--save', type=str, default='phi4_ising_L64.pt')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[φ⁴-Ising] L={args.L}, J={args.J}, a={args.a}, b={args.b}")
    print(f"[φ⁴-Ising] n_chains={args.n_chains}, dt={args.dt}, n_leapfrog={args.n_leapfrog}")

    x = 0.5 * torch.randn(args.n_chains, args.L, args.L, device=device)
    t0 = time()

    # Thermalize
    for step in range(args.n_thermalize):
        x, acc = hmc_step(x, args.J, args.a, args.b, args.dt, args.n_leapfrog)
        if (step + 1) % 500 == 0:
            mag = x.mean(dim=(1, 2))
            print(f"  therm {step+1}/{args.n_thermalize}: accept={acc.float().mean():.3f}, "
                  f"std={x.std():.3f}, |m|={mag.abs().mean():.3f}, [{time()-t0:.1f}s]")

    print(f"[φ⁴-Ising] Thermalization done. {time()-t0:.1f}s")

    # Collect
    samples = []
    rounds_needed = (args.n_samples + args.n_chains - 1) // args.n_chains
    for r in range(rounds_needed):
        for _ in range(args.n_between):
            x, _ = hmc_step(x, args.J, args.a, args.b, args.dt, args.n_leapfrog)
        samples.append(x.cpu().clone())
        if (r + 1) % 10 == 0:
            print(f"  collected {min((r+1)*args.n_chains, args.n_samples)}/{args.n_samples}, [{time()-t0:.1f}s]")

    samples = torch.cat(samples, dim=0)[:args.n_samples]
    print(f"[φ⁴-Ising] Collected {samples.shape[0]} samples in {time()-t0:.1f}s")
    print(f"[φ⁴-Ising] std={samples.std():.4f}, mean={samples.mean():.4f}")
    mag = samples.mean(dim=(1, 2))
    print(f"[φ⁴-Ising] |magnetization|={mag.abs().mean():.4f}")

    torch.save(samples, args.save)
    print(f"[Saved] {args.save}, shape={samples.shape}")

    # Diagnostics
    import scipy.stats as stats
    fhat = torch.fft.fftn(samples, dim=(1, 2), norm='forward')
    fourier_amp = (torch.abs(fhat)**2).mean(dim=0)
    kfreq = np.fft.fftfreq(args.L) * args.L
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    fourier_flat = fourier_amp.numpy().flatten()
    kbins = np.arange(0.5, args.L // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    Abins_raw = Abins.copy()
    Abins *= area_weight
    cond = Abins_raw[0] / Abins_raw[-1] if Abins_raw[-1] > 0 else float('inf')

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].imshow(samples[0].numpy(), cmap='RdBu')
    axes[0].set_title(f'φ⁴-Ising (L={args.L})'); axes[0].axis('off')
    axes[1].hist(samples.numpy().ravel(), bins=100, density=True)
    axes[1].set_title('Marginal (bimodal)')
    axes[2].loglog(kvals, Abins, 'b-o', ms=3)
    axes[2].set_xlabel('k'); axes[2].set_ylabel('Power (area-weighted)')
    axes[2].set_title('Power spectrum')
    axes[3].loglog(kvals, Abins_raw, 'r-o', ms=3)
    axes[3].set_xlabel('k'); axes[3].set_ylabel('C(k)')
    axes[3].set_title(f'Propagator (κ≈{cond:.0f})')
    plt.tight_layout()
    diag_name = args.save.replace('.pt', '_diagnostics.png')
    plt.savefig(diag_name, dpi=150)
    print(f"[Saved] {diag_name}")
    print(f"[Condition number] κ ≈ {cond:.1f}")
    print(f"[Kurtosis] {stats.kurtosis(samples.numpy().ravel()):.3f}")
