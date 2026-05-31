"""
Resolution scaling of the drift Lipschitz constant -- direct numerical
verification of Proposition 3.3 in the rougher-than-data regime, and of the
divergence predicted in the smoother-than-data regime.

For Gaussian targets with closed-form drift b_t(x) = B(t) x, we compute
||B(t)||_2 = max_m |B(t,m)| in Fourier space and report it as a function of
resolution N for several t values. The finding is:
  - Rougher than data: ||B(t)||_2 stable in N (Prop 3.3 satisfied uniformly).
  - Matched: ||B(t)||_2 stable in N (Prop 3.3 satisfied).
  - Smoother than data: ||B(t)||_2 grows polynomially in N near t=0
    (Prop 3.1 prediction; Prop 3.3 cannot apply).

Companion to Appendix F.A (V-norm scaling): same story, two complementary
checks of the Cameron-Martin / Lipschitz analysis.
"""
import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))


def matern_density(grid, sigma_sq, ls, s):
    f = torch.fft.fftfreq(grid, device=DEVICE) * 2 * math.pi * grid
    fx, fy = torch.meshgrid(f, f, indexing='ij')
    sd = sigma_sq * (fx ** 2 + fy ** 2 + ls ** 2) ** (-s)
    sd[0, 0] = 0.0
    return sd


def lin_sched(t):
    return 1.0 - t, -1.0, t, 1.0


def designed_sched(t, lambda_star):
    r = lambda_star; log_r = math.log(r)
    a2 = max((r - r ** t) / (r - 1.0), 1e-30)
    b2 = max((r ** t - 1.0) / (r - 1.0), 1e-30)
    a = math.sqrt(a2); b = math.sqrt(b2)
    da = -0.5 * (r ** t) * log_r / ((r - 1.0) * a)
    db = 0.5 * (r ** t) * log_r / ((r - 1.0) * b)
    return a, da, b, db


def b_op_norm(sd0, sd1, sched_fn, t):
    a, da, b, db = sched_fn(t)
    num = a * da * sd0 + b * db * sd1
    den = a ** 2 * sd0 + b ** 2 * sd1 + 1e-40
    return (num / den).abs().max().item()


def main():
    grid_list = [32, 64, 128, 256, 512]
    ls = 1.0
    s1 = 3
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left panel: Lipschitz at t=0.05 (initial-time stiff regime) vs N
    cases = [
        ('White noise (rougher than data)', 0, 1.0, 'C0'),
        ('Matched (Mat\'ern $s_0{=}3$)', 3, sig1, 'C2'),
        ('Smoother (Mat\'ern $s_0{=}4$)', 4, ((2 * math.pi) ** 2 + ls ** 2) ** 4, 'C3'),
    ]
    ax = axes[0]
    for label, s0, sig0, color in cases:
        Ls = []
        for grid in grid_list:
            sd0 = matern_density(grid, sig0, ls, s0)
            sd1 = matern_density(grid, sig1, ls, s1)
            Ls.append(b_op_norm(sd0, sd1, lin_sched, 0.05))
        ax.plot(grid_list, Ls, '-o', label=label, color=color, lw=1.5, ms=5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'Grid resolution $N$')
    ax.set_ylabel(r'$\|B(t)\|_2$ at $t = 0.05$')
    ax.set_title(r'Drift Lipschitz vs.\ resolution at $t=0.05$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xticks(grid_list)
    ax.set_xticklabels([str(g) for g in grid_list])

    # Right panel: peak Lipschitz vs N, designed schedule on white noise
    ax = axes[1]
    peaks_des = []
    peaks_lin_w = []
    log_pred = []
    for grid in grid_list:
        sd0 = matern_density(grid, 1.0, ls, 0)
        sd1 = matern_density(grid, sig1, ls, s1)
        nyq = grid // 2
        lambda_star = float((sd1[nyq, nyq] / sd0[nyq, nyq]).item())
        log_pred.append(0.5 * abs(math.log(lambda_star)))
        tgrid = np.linspace(1e-3, 1 - 1e-3, 200)
        Lt_des = [b_op_norm(sd0, sd1, lambda tt: designed_sched(tt, lambda_star), t) for t in tgrid]
        Lt_lin_w = [b_op_norm(sd0, sd1, lin_sched, t) for t in tgrid]
        peaks_des.append(max(Lt_des))
        peaks_lin_w.append(max(Lt_lin_w))
    ax.plot(grid_list, peaks_lin_w, '-s', color='C0', lw=1.5, ms=5,
            label=r'Linear sched., white noise (peak in $t$)')
    ax.plot(grid_list, peaks_des, '-o', color='C4', lw=1.5, ms=5,
            label=r'Designed schedule, white noise')
    ax.plot(grid_list, log_pred, ':', color='gray', lw=1.0,
            label=r'$\frac{1}{2}|\log\lambda^\star(N)|$ (Prop 5.1)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'Grid resolution $N$')
    ax.set_ylabel(r'$\max_{t}\|B(t)\|_2$')
    ax.set_title(r'White noise: linear vs.\ designed peak Lipschitz')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xticks(grid_list)
    ax.set_xticklabels([str(g) for g in grid_list])
    plt.tight_layout()
    out = os.path.join(HOME, 'lip_resolution_scaling.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"saved: {out}")

    print("\n=== Linear schedule: peak Lipschitz vs N ===")
    print(f"{'N':>5}  {'white':>10}  {'matched':>10}  {'smoother':>10}")
    for grid in grid_list:
        sd1 = matern_density(grid, sig1, ls, s1)
        ps = []
        for label, s0, sig0, _ in cases:
            sd0 = matern_density(grid, sig0, ls, s0)
            tgrid = np.linspace(1e-3, 1 - 1e-3, 200)
            Lt = [b_op_norm(sd0, sd1, lin_sched, t) for t in tgrid]
            ps.append(max(Lt))
        print(f"{grid:>5d}  {ps[0]:>10.3e}  {ps[1]:>10.3e}  {ps[2]:>10.3e}")


if __name__ == '__main__':
    main()
