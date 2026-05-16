"""
Empirical validation of the drift Lipschitz constant -- directly addresses
R2.B4 ("an experiment to validate the assumptions in Proposition 3.3").

For the Gaussian closed-form drift b_t(x) = B(t) x, the Lipschitz constant in
the 2-norm equals the operator norm ||B(t)||, which is computable analytically
mode-by-mode in the Fourier basis. We plot ||B(t)|| as t ranges over (0,1)
under three setups:
  (a) Linear schedule, white-noise prior, Matern-3 data: stiff near t=1.
  (b) Linear schedule, matched-spectrum noise, Matern-3 data: bounded.
  (c) Designed schedule, white-noise prior, Matern-3 data: bounded
      (~ 0.5 |log lambda*|).

This figure makes Proposition 3.3's Lipschitz claim numerical.
"""
import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))


def matern_density(grid, sigma_sq, ls, s):
    f = torch.fft.fftfreq(grid, device=DEVICE) * 2 * math.pi * grid
    fx, fy = torch.meshgrid(f, f, indexing='ij')
    sd = sigma_sq * (fx ** 2 + fy ** 2 + ls ** 2) ** (-s)
    sd[0, 0] = 0.0  # exclude DC mode (zero-mean fields)
    return sd


def linear_schedule(t):
    return 1.0 - t, -1.0, t, 1.0


def designed_schedule(t, lambda_star):
    r = lambda_star
    log_r = math.log(r)
    a2 = max((r - r ** t) / (r - 1.0), 1e-30)
    b2 = max((r ** t - 1.0) / (r - 1.0), 1e-30)
    a = math.sqrt(a2); b = math.sqrt(b2)
    da = -0.5 * (r ** t) * log_r / ((r - 1.0) * a)
    db = 0.5 * (r ** t) * log_r / ((r - 1.0) * b)
    return a, da, b, db


def b_op_norm(sd0, sd1, sched_fn, t):
    """Compute operator norm of B(t) = (alpha alpha' c0 + beta beta' c1) / (alpha^2 c0 + beta^2 c1).

    In Fourier basis B(t) is diagonal; the 2-norm equals the max |entry|.
    """
    a, da, b, db = sched_fn(t)
    num = a * da * sd0 + b * db * sd1
    den = a ** 2 * sd0 + b ** 2 * sd1 + 1e-40
    entries = num / den
    return entries.abs().max().item()


def main():
    grid = 128
    ls = 1.0
    s1 = 3
    sig1 = ((2 * math.pi) ** 2 + ls ** 2) ** s1
    sd_data = matern_density(grid, sig1, ls, s1)

    # noise priors
    sd_white = matern_density(grid, 1.0, ls, 0)
    sd_match = sd_data.clone()  # matched spectrum
    sd_smoother = matern_density(grid, ((2 * math.pi) ** 2 + ls ** 2) ** 4, ls, 4)
    # smoother-than-data prior (s_0 = 4 > s_1 = 3): violates the rougher-than-data
    # condition, so the drift Lipschitz should diverge as t -> 0.

    # lambda* for white-noise + designed schedule: ratio at Nyquist mode.
    nyq = grid // 2
    lambda_star = float((sd_data[nyq, nyq] / sd_white[nyq, nyq]).item())
    print(f"lambda* (white noise vs Matern-3 data, Nyquist) = {lambda_star:.3e}")
    log_pred = 0.5 * abs(math.log(lambda_star))
    print(f"theoretical bound for designed schedule: 0.5*|log lambda*| = {log_pred:.3f}")

    # Use log-spaced t to capture both initial-time (small t) and terminal-time (close to 1) stiffness
    tg_low = np.logspace(-4, -0.05, 80)  # from 1e-4 up to ~0.89
    tg_high = 1 - np.logspace(-4, -0.05, 80)[::-1]  # from ~0.11 to ~1-1e-4
    tgrid = np.unique(np.concatenate([tg_low, [0.5], tg_high]))

    sd_smoother_5 = matern_density(grid, ((2 * math.pi) ** 2 + ls ** 2) ** 5, ls, 5)
    cases = [
        ('Linear, white noise (rougher)',          linear_schedule, sd_white, sd_data, 'C3', '-'),
        ('Linear, matched-spectrum',               linear_schedule, sd_match, sd_data, 'C2', '-'),
        ('Designed, white noise',                  lambda t: designed_schedule(t, lambda_star), sd_white, sd_data, 'C0', '-'),
        ('Linear, smoother-than-data ($s_0{=}5$)', linear_schedule, sd_smoother_5, sd_data, 'C5', '-'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    # Reference line for the analytical designed-schedule bound
    ax.axhline(log_pred, color='black', linestyle='--', alpha=0.8, lw=2.0, zorder=1,
               label=fr'$\frac{{1}}{{2}}|\log\lambda^\star| \approx {log_pred:.2f}$ (theory)')
    for label, sched, sd0, sd1, color, ls_style in cases:
        norms = [b_op_norm(sd0, sd1, sched, t) for t in tgrid]
        if 'Designed' in label:
            ax.plot(tgrid, norms, label=label, color=color, linestyle='none',
                    marker='o', markersize=7, markevery=8, markerfacecolor='none',
                    markeredgewidth=2.0, zorder=3)
        else:
            ax.plot(tgrid, norms, label=label, color=color, linestyle=ls_style, lw=2.2, zorder=2)
    ax.set_yscale('log')
    ax.set_xscale('linear')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'Interpolation time $t$')
    ax.set_ylabel(r'$\|B(t)\|_2$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, framealpha=0.9)
    plt.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.42)
    out = os.path.join(HOME, 'lipschitz_validation_gaussian.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"saved: {out}")

    # Print summary numbers at a few times
    print(f"\n{'t':>5} {'lin/white':>12} {'lin/match':>12} {'des/white':>12} {'lin/smoother':>14}")
    for t in [0.01, 0.1, 0.5, 0.9, 0.99]:
        L_lw = b_op_norm(sd_white, sd_data, linear_schedule, t)
        L_lm = b_op_norm(sd_match, sd_data, linear_schedule, t)
        L_dw = b_op_norm(sd_white, sd_data, lambda tt: designed_schedule(tt, lambda_star), t)
        L_ls = b_op_norm(sd_smoother, sd_data, linear_schedule, t)
        print(f"{t:>5.2f} {L_lw:>12.3e} {L_lm:>12.3e} {L_dw:>12.3e} {L_ls:>14.3e}")


if __name__ == '__main__':
    main()
