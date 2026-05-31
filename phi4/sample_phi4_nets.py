"""
Sample from the 2D Euclidean φ⁴ model using parallel HMC on GPU.

NETS parameterization (Albergo & Vanden-Eijnden, arXiv:2410.02711, Eq. 128):
  S(φ) = Σ_x [ -2 Σ_μ φ_x φ_{x+μ} + (2D + m²) φ_x² + λ φ_x⁴ ]

D=2 spacetime dimensions. Critical parameters from Appendix 6.6.2:
  L=20: m²=-1.0, λ=0.9 (at phase transition)
  L=16: m²=-1.0, λ=0.8 (ordered phase)

Free propagator: C(k) = 1 / [m² + 4 - 2(cos k₁ + cos k₂)]
Condition number diverges as effective mass → 0 at criticality.
"""

import os, argparse
import numpy as np
import torch
from time import time


def compute_energy(x, m2, lam):
    """
    S(φ) = Σ_x [ -2 Σ_μ φ_x φ_{x+μ} + (2D + m²) φ_x² + λ φ_x⁴ ]
    x: (B, L, L), returns: (B,)
    D=2, so 2D+m² = 4+m²
    """
    # Nearest-neighbor coupling: -2 Σ_μ φ_x φ_{x+μ}
    nn_coupling = (x * torch.roll(x, -1, 1) + x * torch.roll(x, -1, 2))
    E_nn = -2 * nn_coupling.sum(dim=(1, 2))
    # Mass + quartic
    E_mass = (4 + m2) * (x**2).sum(dim=(1, 2))
    E_quartic = lam * (x**4).sum(dim=(1, 2))
    return E_nn + E_mass + E_quartic


def compute_force(x, m2, lam):
    """
    -dS/dφ_x = 2 Σ_μ (φ_{x+μ} + φ_{x-μ}) - 2(2D+m²) φ_x - 4λ φ_x³
    x: (B, L, L), returns: (B, L, L)
    """
    nn_sum = (torch.roll(x, 1, 1) + torch.roll(x, -1, 1) +
              torch.roll(x, 1, 2) + torch.roll(x, -1, 2))
    return 2 * nn_sum - 2 * (4 + m2) * x - 4 * lam * x**3


def hmc_step(x, m2, lam, dt, n_leapfrog):
    """
    One HMC step for a batch of configurations.
    x: (B, L, L), returns: (x_new, accepted) where accepted: (B,) bool
    """
    p = torch.randn_like(x)
    x_old = x.clone()
    H_old = compute_energy(x, m2, lam) + 0.5 * p.pow(2).sum(dim=(1, 2))

    # Leapfrog
    x_new = x.clone()
    force = compute_force(x_new, m2, lam)
    p = p + 0.5 * dt * force
    for _ in range(n_leapfrog - 1):
        x_new = x_new + dt * p
        force = compute_force(x_new, m2, lam)
        p = p + dt * force
    x_new = x_new + dt * p
    force = compute_force(x_new, m2, lam)
    p = p + 0.5 * dt * force

    H_new = compute_energy(x_new, m2, lam) + 0.5 * p.pow(2).sum(dim=(1, 2))
    dH = H_new - H_old

    # Metropolis accept/reject
    accept = (dH < 0) | (torch.rand(x.shape[0], device=x.device) < torch.exp(-dH.clamp(max=50)))
    x_out = torch.where(accept[:, None, None], x_new, x_old)
    return x_out, accept


def sample_phi4_nets(L=20, m2=-1.0, lam=0.9, n_samples=20000,
                     n_chains=500, n_thermalize=2000, n_between=20,
                     dt=0.1, n_leapfrog=20, device='cuda', gpu=0):
    """
    Parallel HMC sampling of NETS-style φ⁴ model.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    print(f"[φ⁴-NETS] L={L}, m²={m2}, λ={lam}")
    print(f"[φ⁴-NETS] n_chains={n_chains}, n_thermalize={n_thermalize}, n_between={n_between}")
    print(f"[φ⁴-NETS] dt={dt}, n_leapfrog={n_leapfrog}, device={device}")

    # Free propagator condition number
    # M_k = m² + 4 - 2(cos k1 + cos k2)
    # M_min at k=0: m² + 4 - 4 = m²
    # M_max at k=(π,π): m² + 4 + 4 = m² + 8
    # For m²>0: κ = (m²+8)/m²
    # For m²<0: free theory unstable, need λ>0
    if m2 > 0:
        kappa_free = (m2 + 8) / m2
        print(f"[φ⁴-NETS] Free theory condition number: {kappa_free:.1f}")
    else:
        print(f"[φ⁴-NETS] m²<0: free theory unstable, λ={lam} stabilizes")

    # Initialize from free theory if m²>0, else random near 0
    if m2 > 0:
        x = torch.randn(n_chains, L, L, device=device) * (1.0 / (m2 + 4)**0.5)
    else:
        x = 0.5 * torch.randn(n_chains, L, L, device=device)

    t0 = time()

    # Thermalize with acceptance monitoring
    total_accept = 0
    for step in range(n_thermalize):
        x, acc = hmc_step(x, m2, lam, dt, n_leapfrog)
        total_accept += acc.sum().item()
        if (step + 1) % 200 == 0:
            rate = total_accept / ((step + 1) * n_chains)
            E = compute_energy(x, m2, lam).mean().item() / L**2
            mag = x.mean(dim=(1, 2)).abs().mean().item()
            print(f"  therm {step+1}/{n_thermalize}: accept={rate:.3f}, E/site={E:.4f}, "
                  f"|m|={mag:.3f}, std={x.std().item():.3f}, [{time()-t0:.1f}s]")

    final_rate = total_accept / (n_thermalize * n_chains)
    print(f"[φ⁴-NETS] Thermalization done. accept={final_rate:.3f}, {time()-t0:.1f}s")

    # Collect samples
    samples = []
    total_accept = 0
    rounds_needed = (n_samples + n_chains - 1) // n_chains
    for r in range(rounds_needed):
        for _ in range(n_between):
            x, acc = hmc_step(x, m2, lam, dt, n_leapfrog)
            total_accept += acc.sum().item()
        samples.append(x.cpu().clone())
        if (r + 1) % 10 == 0:
            n_collected = min((r + 1) * n_chains, n_samples)
            print(f"  collected {n_collected}/{n_samples}, [{time()-t0:.1f}s]")

    samples = torch.cat(samples, dim=0)[:n_samples]
    total_hmc = rounds_needed * n_between * n_chains
    print(f"[φ⁴-NETS] HMC accept rate: {total_accept / total_hmc:.3f}")
    print(f"[φ⁴-NETS] Collected {samples.shape[0]} samples in {time()-t0:.1f}s")
    print(f"[φ⁴-NETS] mean={samples.mean():.4f}, std={samples.std():.4f}")
    mag = samples.mean(dim=(1, 2))
    print(f"[φ⁴-NETS] |magnetization|={mag.abs().mean():.4f} ± {mag.abs().std():.4f}")

    return samples


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=20)
    p.add_argument('--m2', type=float, default=-1.0)
    p.add_argument('--lam', type=float, default=0.9)
    p.add_argument('--n_samples', type=int, default=20000)
    p.add_argument('--n_chains', type=int, default=500)
    p.add_argument('--n_thermalize', type=int, default=3000)
    p.add_argument('--n_between', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--n_leapfrog', type=int, default=30)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--save', type=str, default=None)
    args = p.parse_args()

    if args.save is None:
        args.save = f'phi4_nets_L{args.L}_m2{args.m2}_lam{args.lam}.pt'

    samples = sample_phi4_nets(L=args.L, m2=args.m2, lam=args.lam,
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
    # Raw propagator (no area weight) for condition number visualization
    Abins_raw = Abins.copy()
    Abins *= area_weight

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].imshow(samples[0].numpy(), cmap='RdBu')
    axes[0].set_title(f'φ⁴-NETS (L={args.L}, m²={args.m2}, λ={args.lam})')
    axes[0].axis('off')

    axes[1].hist(samples.numpy().ravel(), bins=100, density=True)
    axes[1].set_title('Marginal distribution')

    axes[2].loglog(kvals, Abins, 'b-o', ms=3)
    axes[2].set_xlabel('k'); axes[2].set_ylabel('Power (area-weighted)')
    axes[2].set_title('Power spectrum')

    axes[3].loglog(kvals, Abins_raw, 'r-o', ms=3)
    axes[3].set_xlabel('k'); axes[3].set_ylabel('C(k)')
    cond = Abins_raw[0] / Abins_raw[-1] if Abins_raw[-1] > 0 else float('inf')
    axes[3].set_title(f'Propagator (κ≈{cond:.0f})')

    plt.tight_layout()
    diag_name = args.save.replace('.pt', '_diagnostics.png')
    plt.savefig(diag_name, dpi=150)
    print(f"[Saved] {diag_name}")
    print(f"[Condition number] C(k_min)/C(k_max) ≈ {cond:.1f}")
