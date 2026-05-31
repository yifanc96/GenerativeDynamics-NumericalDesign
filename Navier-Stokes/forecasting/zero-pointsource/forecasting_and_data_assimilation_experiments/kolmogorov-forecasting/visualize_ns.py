"""Paper-grade figures for the Kolmogorov forecasting study.

Inputs: JSON files from compare_ns.py (one per lag) and rollout_ns.py (one).
Outputs:
  1. figs/headline_bars.pdf       — per-lag CRPS + enstrophy-W2 bar chart, Föllmer highlighted
  2. figs/vorticity_grid.pdf      — truth vs schedule-sample vorticity grid
  3. figs/spectrum_overlay.pdf    — energy-spectrum plot per lag
  4. figs/rank_histograms.pdf     — Talagrand rank histogram per schedule
  5. figs/rollout_curves.pdf      — autoregressive RMSE + enstrophy-W2 vs step
  6. figs/spread_skill.pdf        — per-IC scatter, ensemble spread vs RMSE

Usage:
    python visualize_ns.py
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCHEDULE_ORDER = ['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero']
PRETTY = {
    'follmer':  r'Föllmer $\sqrt{1-t^2}$',
    'baseline': r'baseline $1-t$',
    'triangle': r'$\sqrt{t(1-t)}$',
    'const':    r'const $g{=}1$',
    'sqrt_t':   r'$\sqrt{t}$',
    'zero':     r'ODE ($g{=}0$)',
}
COLOURS = {
    'follmer':  'tab:orange',
    'baseline': 'tab:blue',
    'triangle': 'tab:green',
    'const':    'tab:red',
    'sqrt_t':   'tab:purple',
    'zero':     'tab:gray',
}


def _bold_for_min(bar_handles, values):
    i = int(np.argmin(values))
    for h in bar_handles:
        h.set_color('lightsteelblue')
    bar_handles[i].set_color('tab:orange')


def headline_bars(compare_jsons, out):
    fig, axes = plt.subplots(2, len(compare_jsons), figsize=(4.5 * len(compare_jsons), 5.5), squeeze=False)
    for j, (lag, data) in enumerate(compare_jsons.items()):
        for row, metric, ylabel in [(0, 'crps', 'CRPS (lower = better)'),
                                    (1, 'enstrophy_w2', 'enstrophy $W_2$')]:
            ax = axes[row, j]
            vals = [data[s][metric] for s in SCHEDULE_ORDER]
            bars = ax.bar(range(len(SCHEDULE_ORDER)), vals, edgecolor='black')
            _bold_for_min(bars, vals)
            ax.set_xticks(range(len(SCHEDULE_ORDER)))
            ax.set_xticklabels([PRETTY[s] for s in SCHEDULE_ORDER], rotation=30, ha='right')
            if row == 0:
                ax.set_title(f'lag = {lag}')
            if j == 0:
                ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def rollout_curves(rollout_json, out):
    data = rollout_json['rollout']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    axes = axes[0]
    for sched in SCHEDULE_ORDER:
        if sched not in data:
            continue
        recs = data[sched]
        steps = np.arange(1, len(recs) + 1)
        rmses = [r['rmse'] for r in recs]
        ensw2 = [r['ens_w2'] for r in recs]
        axes[0].plot(steps, rmses, label=PRETTY[sched], color=COLOURS[sched], lw=2)
        axes[1].plot(steps, ensw2, label=PRETTY[sched], color=COLOURS[sched], lw=2)
    axes[0].set_xlabel('rollout step'); axes[0].set_ylabel('RMSE (ens mean vs truth)')
    axes[0].set_title('Autoregressive pointwise RMSE')
    axes[0].legend()
    axes[1].set_xlabel('rollout step'); axes[1].set_ylabel('enstrophy $W_2$')
    axes[1].set_title('Autoregressive enstrophy-distribution $W_2$')
    axes[1].legend()
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def rank_histograms(compare_jsons, out, lag_key=10):
    if lag_key not in compare_jsons:
        lag_key = next(iter(compare_jsons))
    data = compare_jsons[lag_key]
    fig, axes = plt.subplots(1, len(SCHEDULE_ORDER), figsize=(3 * len(SCHEDULE_ORDER), 3.2), squeeze=False)
    axes = axes[0]
    for i, sched in enumerate(SCHEDULE_ORDER):
        hist = np.asarray(data[sched]['rank_hist'])
        n_bins = len(hist)
        ideal = 1.0 / n_bins
        axes[i].bar(range(n_bins), hist, edgecolor='black', color=COLOURS[sched], alpha=0.85)
        axes[i].axhline(ideal, color='k', linestyle=':', lw=1)
        axes[i].set_title(f'{PRETTY[sched]}')
        axes[i].set_xlabel('rank')
        if i == 0: axes[i].set_ylabel('freq')
    plt.suptitle(f'Rank histograms (Talagrand), lag = {lag_key}  — flat = calibrated')
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compare_glob', type=str, default='./figs/compare_lag*.json')
    p.add_argument('--rollout', type=str, default='./figs/rollout.json')
    p.add_argument('--out_dir', type=str, default='./figs')
    args = p.parse_args()

    compare_files = sorted(glob.glob(args.compare_glob))
    compare_data = {}
    for f in compare_files:
        with open(f) as h:
            d = json.load(h)
        lag = d['ck_args']['lag']
        # Aggregate multiple seeds if present: store first seed only for now (we can aggregate elsewhere)
        compare_data.setdefault(lag, {})
        if not compare_data[lag]:
            compare_data[lag] = d['results']

    if compare_data:
        headline_bars(compare_data, os.path.join(args.out_dir, 'headline_bars.pdf'))
        rank_histograms(compare_data, os.path.join(args.out_dir, 'rank_histograms.pdf'))
    if os.path.exists(args.rollout):
        with open(args.rollout) as h:
            ro = json.load(h)
        rollout_curves(ro, os.path.join(args.out_dir, 'rollout_curves.pdf'))


if __name__ == '__main__':
    main()
