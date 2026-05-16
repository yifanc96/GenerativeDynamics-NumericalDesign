"""Combined summary figure (2 rows x N targets) for the §3.4.1 study.
Run compare_zps.py for each target first to produce zps_<target>_results.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='./figs')
    p.add_argument('--targets', type=str, nargs='+',
                   default=['gaussian1d', 'bimodal1d', 'ou_forecast', 'gmm2d_forecast'])
    p.add_argument('--schedules_order', type=str, nargs='+',
                   default=['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero'])
    args = p.parse_args()

    data = {}
    for t in args.targets:
        with open(os.path.join(args.out_dir, f'zps_{t}_results.json')) as f:
            data[t] = json.load(f)['agg']

    pretty_target = {
        'gaussian1d':      r'Target A: $\mathcal{N}(\mu, \sigma^2)$',
        'bimodal1d':       r'Target B: bimodal GMM',
        'ou_forecast':     r'Target C: OU forecasting',
        'gmm2d_forecast':  r'Target D: 2D GMM jump-forecast',
    }
    pretty_sched = {
        'follmer':   r'Föllmer $\sqrt{1-t^2}$',
        'baseline':  r'baseline $1-t$',
        'triangle':  r'$\sqrt{t(1-t)}$',
        'const':     r'const $g{=}1$',
        'sqrt_t':    r'$\sqrt{t}$',
        'zero':      r'ODE ($g{=}0$)',
    }

    n_t = len(args.targets)
    fig, axes = plt.subplots(2, n_t, figsize=(4.0 * n_t, 5.6), squeeze=False)

    for j, t in enumerate(args.targets):
        agg = data[t]
        # Row 0: path KL
        ax = axes[0, j]
        schedules = [s for s in args.schedules_order if s != 'zero']
        means = [agg[s]['kl_mean'] for s in schedules]
        stds = [agg[s]['kl_std'] for s in schedules]
        idx_min = int(np.argmin(means))
        colors = ['tab:blue'] * len(schedules)
        colors[idx_min] = 'tab:orange'
        # specifically highlight follmer if present in orange
        if 'follmer' in schedules:
            colors[schedules.index('follmer')] = 'tab:orange' if schedules[idx_min] != 'follmer' else 'tab:orange'
        x = np.arange(len(schedules))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_sched[s] for s in schedules], rotation=30, ha='right')
        if j == 0:
            ax.set_ylabel('path KL (Girsanov)')
        ax.set_title(pretty_target[t])
        ax.axhline(min(means), color='tab:orange', linestyle=':', linewidth=1, alpha=0.5)

        # Row 1: marginal W2
        ax = axes[1, j]
        schedules = args.schedules_order
        means = [agg[s]['w2_mean'] for s in schedules]
        stds = [agg[s]['w2_std'] for s in schedules]
        x = np.arange(len(schedules))
        ax.bar(x, means, yerr=stds, capsize=4, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_sched[s] for s in schedules], rotation=30, ha='right')
        if j == 0:
            ax.set_ylabel('marginal $W_2$')

    plt.tight_layout()
    out_pdf = os.path.join(args.out_dir, 'zps_summary.pdf')
    out_png = os.path.join(args.out_dir, 'zps_summary.png')
    plt.savefig(out_pdf, dpi=150)
    plt.savefig(out_png, dpi=150)
    print(f'[saved] {out_pdf}\n[saved] {out_png}')


if __name__ == '__main__':
    main()
