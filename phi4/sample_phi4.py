"""
Sample from the 2D φ⁴ model at criticality using parallel HMC on GPU.

Energy from Guth et al. (arXiv:2208.05003):
  E(x) = (β/2) Σ_{|u-v|=1} (x(u) - x(v))² + Σ_u (x(u)² - 1)²

β_c ≈ 0.68 is the critical inverse temperature.
At criticality: long-range correlations, power-law spectrum, bimodal marginals ≈ ±1.
"""

import os, argparse
import numpy as np
import torch
from time import time


def compute_energy(x, beta):
    """
    E(x) = (β/2) Σ_{nn} (x_u - x_v)² + Σ_u (x_u² - 1)²
    x: (B, L, L), returns: (B,)
    """
    # Nearest-neighbor differences (periodic BC)
    dx1 = x - torch.roll(x, 1, 1)
    dx2 = x - torch.roll(x, 1, 2)
    E_nn = (beta / 2) * (dx1**2 + dx2**2).sum(dim=(1, 2))
    # Double-well potential
    E_dw = ((x**2 - 1)**2).sum(dim=(1, 2))
    return E_nn + E_dw


def compute_force(x, beta):
    """
    -dE/dx for a batch.
    x: (B, L, L), returns: (B, L, L)
    """
    # Laplacian: Σ_{nn} (x_v - x_u) = (sum of neighbors) - 4*x_u
    nn_sum = (torch.roll(x, 1, 1) + torch.roll(x, -1, 1) +
              torch.roll(x, 1, 2) + torch.roll(x, -1, 2))
    laplacian = nn_sum - 4 * x
    # -dE/dx = β * laplacian - 4*x*(x²-1)
    return beta * laplacian - 4 * x * (x**2 - 1)


def hmc_step(x, beta, dt, n_leapfrog):
    """
    One HMC step for a batch of configurations.
    x: (B, L, L), returns: (x_new, accepted) where accepted: (B,) bool
    """
    p = torch.randn_like(x)
    x_old = x.clone()
    H_old = compute_energy(x, beta) + 0.5 * p.pow(2).sum(dim=(1, 2))

    # Leapfrog
    x_new = x.clone()
    force = compute_force(x_new, beta)
    p = p + 0.5 * dt * force
    for _ in range(n_leapfrog - 1):
        x_new = x_new + dt * p
        force = compute_force(x_new, beta)
        p = p + dt * force
    x_new = x_new + dt * p
    force = compute_force(x_new, beta)
    p = p + 0.5 * dt * force

    H_new = compute_energy(x_new, beta) + 0.5 * p.pow(2).sum(dim=(1, 2))
    dH = H_new - H_old

    # Metropolis accept/reject
    accept = (dH < 0) | (torch.rand(x.shape[0], device=x.device) < torch.exp(-dH.clamp(max=50)))
    x_out = torch.where(accept[:, None, None], x_new, x_old)
    return x_out, accept


def sample_phi4(L=32, beta=0.68, n_samples=10000,
                n_chains=200, n_thermalize=1000, n_between=10,
                dt=0.1, n_leapfrog=20, device='cuda', gpu=0):
    """
    Parallel HMC sampling of φ⁴ model at criticality.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    print(f"[φ⁴] L={L}, β={beta} (β_c≈0.68)")
    print(f"[φ⁴] n_chains={n_chains}, n_thermalize={n_thermalize}, n_between={n_between}")
    print(f"[φ⁴] dt={dt}, n_leapfrog={n_leapfrog}, device={device}")

    # Initialize near ±1 (ordered state)
    x = torch.sign(torch.randn(n_chains, L, L, device=device))
    x = x + 0.1 * torch.randn_like(x)

    t0 = time()

    # Thermalize with dt tuning
    total_accept = 0
    for step in range(n_thermalize):
        x, acc = hmc_step(x, beta, dt, n_leapfrog)
        total_accept += acc.sum().item()
        if (step + 1) % 200 == 0:
            rate = total_accept / ((step + 1) * n_chains)
            E = compute_energy(x, beta).mean().item() / L**2
            mag = x.mean(dim=(1, 2)).abs().mean().item()
            print(f"  therm {step+1}/{n_thermalize}: accept={rate:.3f}, E/site={E:.4f}, "
                  f"|m|={mag:.3f}, std={x.std().item():.3f}, [{time()-t0:.1f}s]")

    final_rate = total_accept / (n_thermalize * n_chains)
    print(f"[φ⁴] Thermalization done. accept={final_rate:.3f}, {time()-t0:.1f}s")

    # Collect samples
    samples = []
    total_accept = 0
    rounds_needed = (n_samples + n_chains - 1) // n_chains
    for r in range(rounds_needed):
        for _ in range(n_between):
            x, acc = hmc_step(x, beta, dt, n_leapfrog)
            total_accept += acc.sum().item()
        samples.append(x.cpu().clone())
        if (r + 1) % 10 == 0:
            n_collected = min((r + 1) * n_chains, n_samples)
            print(f"  collected {n_collected}/{n_samples}, [{time()-t0:.1f}s]")

    samples = torch.cat(samples, dim=0)[:n_samples]
    total_hmc = rounds_needed * n_between * n_chains
    print(f"[φ⁴] HMC accept rate: {total_accept / total_hmc:.3f}")
    print(f"[φ⁴] Collected {samples.shape[0]} samples in {time()-t0:.1f}s")
    print(f"[φ⁴] mean={samples.mean():.4f}, std={samples.std():.4f}")

    return samples


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=32)
    p.add_argument('--n_samples', type=int, default=20000)
    p.add_argument('--n_chains', type=int, default=500)
    p.add_argument('--n_thermalize', type=int, default=2000)
    p.add_argument('--n_between', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--n_leapfrog', type=int, default=30)
    p.add_argument('--beta', type=float, default=0.68)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--save', type=str, default='phi4_data.pt')
    args = p.parse_args()

    samples = sample_phi4(L=args.L, beta=args.beta,
                          n_samples=args.n_samples, n_chains=args.n_chains,
                          n_thermalize=args.n_thermalize, n_between=args.n_between,
                          dt=args.dt, n_leapfrog=args.n_leapfrog, gpu=args.gpu)

    torch.save(samples, args.save)
    print(f"[Saved] {args.save}, shape={samples.shape}")

    # Diagnostics
    import scipy.stats as stats
    fhat = torch.fft.fftn(samples, dim=(1, 2), norm='forward')
    fourier_amp = (torch.abs(fhat)**2).mean(dim=0)
    npix = args.L
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    fourier_flat = fourier_amp.numpy().flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    Abins, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    Abins *= area_weight

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(samples[0].numpy(), cmap='RdBu', vmin=-2, vmax=2)
    axes[0].set_title(f'φ⁴ sample (L={args.L}, β={args.beta})'); axes[0].axis('off')
    axes[1].hist(samples.numpy().ravel(), bins=100, density=True)
    axes[1].set_title('Marginal distribution')
    axes[2].loglog(kvals, Abins)
    axes[2].set_xlabel('k'); axes[2].set_ylabel('Power')
    axes[2].set_title('Power spectrum')
    plt.tight_layout()
    diag_name = args.save.replace('.pt', '_diagnostics.png')
    plt.savefig(diag_name, dpi=150)
    print(f"[Saved] {diag_name}")
