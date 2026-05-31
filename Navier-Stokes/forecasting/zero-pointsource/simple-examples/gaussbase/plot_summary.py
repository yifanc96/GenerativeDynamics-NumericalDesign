"""Combine per-target results into a single headline figure.
Run after compare_schedules.py has produced <target>_results.json in out_dir.
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
                   default=['gaussian1d', 'bimodal1d', 'ou_forecast'])
    p.add_argument('--schedules_order', type=str, nargs='+',
                   default=['const', 'follmer', 'triangle', 'sqrt_t', 'lin_decay', 'ode'])
    args = p.parse_args()

    # Load results
    data = {}
    for t in args.targets:
        with open(os.path.join(args.out_dir, f'{t}_results.json')) as f:
            data[t] = json.load(f)['agg']

    pretty_target = {
        'gaussian1d':  r'Target A: $\mathcal{N}(\mu, \sigma^2)$',
        'bimodal1d':   r'Target B: bimodal GMM',
        'ou_forecast': r'Target C: OU forecasting',
    }
    pretty_sched = {
        'const':     r'const ($g{=}$const)',
        'follmer':   r'$\sqrt{1-t}$',
        'triangle':  r'$\sqrt{t(1-t)}$',
        'sqrt_t':    r'$\sqrt{t}$',
        'lin_decay': r'$1-t$',
        'ode':       r'ODE ($g{=}0$)',
    }

    n_t = len(args.targets)
    fig, axes = plt.subplots(2, n_t, figsize=(4.0 * n_t, 5.6), squeeze=False)

    for j, t in enumerate(args.targets):
        agg = data[t]
        # path KL row
        ax = axes[0, j]
        schedules = [s for s in args.schedules_order if s != 'ode']
        means = [agg[s]['kl_mean'] for s in schedules]
        stds = [agg[s]['kl_std'] for s in schedules]
        idx_min = int(np.argmin(means))
        colors = ['tab:blue'] * len(schedules)
        colors[idx_min] = 'tab:orange'
        x = np.arange(len(schedules))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_sched[s] for s in schedules], rotation=30, ha='right')
        ax.set_ylabel('path KL (Girsanov)' if j == 0 else '')
        ax.set_title(pretty_target[t])
        ax.axhline(min(means), color='tab:orange', linestyle=':', linewidth=1, alpha=0.5)

        # marginal W2 row
        ax = axes[1, j]
        schedules = args.schedules_order
        means = [agg[s]['w2_mean'] for s in schedules]
        stds = [agg[s]['w2_std'] for s in schedules]
        x = np.arange(len(schedules))
        ax.bar(x, means, yerr=stds, capsize=4, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_sched[s] for s in schedules], rotation=30, ha='right')
        ax.set_ylabel('marginal $W_2$ (x1 vs samples)' if j == 0 else '')

    plt.tight_layout()
    out = os.path.join(args.out_dir, 'summary_all_targets.pdf')
    plt.savefig(out, dpi=150)
    out2 = os.path.join(args.out_dir, 'summary_all_targets.png')
    plt.savefig(out2, dpi=150)
    print(f'[saved] {out}\n[saved] {out2}')


if __name__ == '__main__':
    main()
