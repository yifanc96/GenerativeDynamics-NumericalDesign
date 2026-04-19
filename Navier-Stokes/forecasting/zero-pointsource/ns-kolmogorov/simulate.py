"""Generate 2D Kolmogorov-flow vorticity trajectories via torch-cfd.

Deterministic dynamics (periodic BC, sinusoidal Kolmogorov forcing), so
pointwise comparison of forecast vs truth is meaningful. Stochasticity in
the forecasting problem is then introduced *later* via partial observation
(AvgPool coarsening) of the conditioning input.

Usage:
    python simulate.py --n_traj 700 --resolution 128 --out NSdata/kolmogorov_128/data.pt
"""
import argparse
import math
import os
import time

import torch
import torch.fft as fft

from torch_cfd.grids import Grid
from torch_cfd.forcings import KolmogorovForcing
from torch_cfd.spectral import NavierStokes2DSpectral, RK4CrankNicolsonStepper


DEFAULTS = dict(
    viscosity=1e-3,
    forcing_k=4,           # Kolmogorov forcing: sin(4 y)
    forcing_scale=1.0,
    drag=0.1,              # small damping prevents pile-up at large scales
    domain=2.0 * math.pi,
    warmup_time=5.0,       # drop this initial transient
    snap_dt=0.05,          # Δt between stored snapshots (lag × snap_dt = physical lag)
    n_snaps=200,           # snapshots per trajectory (after warmup)
    dt_solver=0.005,       # internal CN-RK4 step; 128² stable at ν=1e-3
    ic_peak_wavenumber=4,  # GRF peak scale
    ic_tau=7.0, ic_gamma=2.5,  # GRF covariance parameters
)


def sample_initial_vorticity(batch, N, ic_tau, ic_gamma, ic_peak,
                             device, dtype=torch.float32):
    """Draw a Gaussian random field vorticity IC on an N×N grid.

    Matches the FNO paper family (Li et al. 2020) approximately, but we
    re-scale so that the vorticity has O(10) peak amplitude regardless of N.
    """
    # FFT-space Gaussian with covariance ∝ ((-Δ + τ² I)^-γ)
    kx = torch.fft.fftfreq(N, d=1.0 / N).to(device)
    ky = torch.fft.fftfreq(N, d=1.0 / N).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2
    # Spectral density shape
    S = (4.0 * math.pi ** 2 * K2 + ic_tau ** 2) ** (-ic_gamma)
    S[0, 0] = 0.0  # zero mean
    # Sample complex Gaussian and inverse FFT
    noise = torch.randn(batch, N, N, device=device, dtype=dtype) \
        + 1j * torch.randn(batch, N, N, device=device, dtype=dtype)
    spec = noise * (S.unsqueeze(0)).sqrt()
    omega = torch.fft.ifft2(spec).real
    # Normalize so that the typical peak |omega| ≈ 10
    omega = omega / omega.std(dim=(1, 2), keepdim=True).clamp_min(1e-8) * 3.0
    return omega.to(dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_traj', type=int, default=700)
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--viscosity', type=float, default=DEFAULTS['viscosity'])
    p.add_argument('--forcing_k', type=int, default=DEFAULTS['forcing_k'])
    p.add_argument('--drag', type=float, default=DEFAULTS['drag'])
    p.add_argument('--snap_dt', type=float, default=DEFAULTS['snap_dt'])
    p.add_argument('--n_snaps', type=int, default=DEFAULTS['n_snaps'])
    p.add_argument('--dt_solver', type=float, default=DEFAULTS['dt_solver'])
    p.add_argument('--warmup_time', type=float, default=DEFAULTS['warmup_time'])
    p.add_argument('--batch', type=int, default=50, help='trajectories simulated in parallel')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]

    torch.manual_seed(args.seed)
    N = args.resolution
    domain = DEFAULTS['domain']

    grid = Grid(shape=(N, N), domain=((0, domain), (0, domain)), device=device)
    forcing = KolmogorovForcing(
        grid=grid, diam=domain, offsets=((0, 0), (0, 0)),
        vorticity=True, wave_number=args.forcing_k,
        scale=DEFAULTS['forcing_scale'],
    )
    stepper = RK4CrankNicolsonStepper(requires_grad=False)
    eq = NavierStokes2DSpectral(
        viscosity=args.viscosity, grid=grid, drag=args.drag,
        smooth=True, forcing_fn=forcing, step_fn=stepper,
    ).to(device)

    warmup_steps = max(1, int(round(args.warmup_time / args.dt_solver)))
    inner_steps = max(1, int(round(args.snap_dt / args.dt_solver)))
    print(f"[sim] N={N}, viscosity={args.viscosity}, forcing_k={args.forcing_k}")
    print(f"[sim] dt_solver={args.dt_solver}, warmup {args.warmup_time} = {warmup_steps} solver steps")
    print(f"[sim] snap_dt={args.snap_dt} = {inner_steps} solver steps; n_snaps={args.n_snaps}")

    out = torch.zeros(args.n_traj, args.n_snaps, N, N, dtype=dtype)

    t_start = time.time()
    b = args.batch
    done = 0
    while done < args.n_traj:
        cur_b = min(b, args.n_traj - done)
        omega0 = sample_initial_vorticity(cur_b, N, DEFAULTS['ic_tau'],
                                          DEFAULTS['ic_gamma'],
                                          DEFAULTS['ic_peak_wavenumber'], device, dtype)
        vort_hat = fft.rfft2(omega0)
        # warmup
        vort_hat, _ = eq(vort_hat, args.dt_solver, steps=warmup_steps)
        # snapshots
        for s in range(args.n_snaps):
            omega = fft.irfft2(vort_hat, s=(N, N))
            out[done:done + cur_b, s] = omega.detach().cpu().to(dtype)
            vort_hat, _ = eq(vort_hat, args.dt_solver, steps=inner_steps)
        done += cur_b
        elapsed = time.time() - t_start
        print(f"  [sim] {done}/{args.n_traj} trajectories done  ({elapsed:.1f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'omega': out, 'args': vars(args)}, args.out)
    mb = out.element_size() * out.numel() / 1024 ** 2
    print(f"[saved] {args.out}  shape={tuple(out.shape)}  {mb:.1f} MiB")
    # Quick sanity
    enstrophy = 0.5 * (out ** 2).mean(dim=(-1, -2))
    print(f"[sanity] mean enstrophy (per-traj average over snaps): "
          f"{enstrophy.mean():.3f} ± {enstrophy.std():.3f}")
    print(f"[sanity] vorticity abs max: {out.abs().max():.2f}, std: {out.std():.3f}")


if __name__ == '__main__':
    main()
