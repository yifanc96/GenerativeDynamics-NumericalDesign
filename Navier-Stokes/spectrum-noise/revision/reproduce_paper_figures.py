"""
Reproduce paper figures from the existing saved spectra in
spectrum-noise/inference/result/.

This is a verification step: we re-plot Figure 5 (NS spectrum noise comparison)
and Figure 6 (NS white noise + designed schedule) using the saved per-mode
amplitude .pt files and confirm they match the existing PDFs in the same
folder.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES_DIR = '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/Navier-Stokes/spectrum-noise/inference/result/'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_spec(name):
    return torch.load(os.path.join(RES_DIR, name), weights_only=False)


def fig_paper_fig5():
    """Figure 5 of the paper: enstrophy spectra at 128x128, 10 RK4 steps,
    comparing the three noise types (matched-spectrum, mul-k rougher, white)
    all with the linear schedule."""
    panels = [
        ('NS-res128-spectrum-noise-truth-RK10.pt',     'NS-res128-spectrum-noise-noise-RK10.pt',     'NS-res128-spectrum-noise-generated-RK10.pt',     'Matched-spectrum noise + linear'),
        ('NS-res128-spectrum-noisemulk-truth-RK10.pt', 'NS-res128-spectrum-noisemulk-noise-RK10.pt', 'NS-res128-spectrum-noisemulk-generated-RK10.pt', r'Rougher (mul-$k$) spectrum noise + linear'),
        ('NS-res128-white-noise-truth-RK10.pt',        'NS-res128-white-noise-noise-RK10.pt',        'NS-res128-white-noise-generated-RK10.pt',        'White noise + linear'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (truth_f, noise_f, gen_f, title) in zip(axes, panels):
        S_truth = load_spec(truth_f)
        S_noise = load_spec(noise_f)
        S_gen = load_spec(gen_f)
        kvals = np.arange(1, len(S_truth) + 1)
        ax.plot(kvals, S_truth, 'k-', lw=1.6, label='Truth')
        ax.plot(kvals, S_noise, 'g:', lw=1.2, label='Noise prior')
        ax.plot(kvals, S_gen, 'r-', lw=1.2, marker='o', ms=2, label='Generated (10 RK4 steps)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel(r'Wavenumber $k$')
        ax.set_ylabel(r'Enstrophy $S(k)$')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'reproduced_paper_fig5_NS_spectrum_comparison.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"saved: {out}")


def fig_paper_fig6():
    """Figure 6 of the paper: NS white-noise prior, comparing linear schedule
    vs designed schedule, both at 10 RK4 steps."""
    S_truth = load_spec('NS-res128-white-noise-truth-RK10.pt')
    S_noise = load_spec('NS-res128-white-noise-noise-RK10.pt')
    S_lin = load_spec('NS-res128-white-noise-generated-RK10.pt')
    S_des = load_spec('NS-res128-white-noise-generated-designed-schedule-RK10.pt')
    kvals = np.arange(1, len(S_truth) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.5))
    ax.plot(kvals, S_truth, 'k-', lw=1.6, label='Truth')
    ax.plot(kvals, S_noise, 'g:', lw=1.0, label='Noise prior (white)')
    ax.plot(kvals, S_lin, 'r--', lw=1.2, marker='s', ms=2, label='Linear schedule (10 RK4 steps)')
    ax.plot(kvals, S_des, 'b-', lw=1.4, marker='o', ms=2, label='Designed schedule (10 RK4 steps)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'Enstrophy $S(k)$')
    ax.set_title(r'NS $128{\times}128$, white-noise prior: linear vs.\ designed schedule')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'reproduced_paper_fig6_NS_white_designed.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"saved: {out}")

    # also report band errors
    def banded(kvals, S_gen, S_truth, lo, hi):
        m = (kvals >= lo) & (kvals < hi)
        rel = np.abs(S_gen[m] - S_truth[m]) / np.abs(S_truth[m])
        return float(rel.mean())

    print("\n[Fig 6 verification] Linear vs Designed at 10 RK4 steps, 128x128:")
    for name, S in [('Linear', S_lin), ('Designed', S_des)]:
        low = banded(kvals, S, S_truth, 1, 8)
        mid = banded(kvals, S, S_truth, 8, 24)
        high = banded(kvals, S, S_truth, 24, 65)
        print(f"  {name:10s} low_band={low:.3e}  mid_band={mid:.3e}  high_band={high:.3e}")


if __name__ == '__main__':
    fig_paper_fig5()
    fig_paper_fig6()
