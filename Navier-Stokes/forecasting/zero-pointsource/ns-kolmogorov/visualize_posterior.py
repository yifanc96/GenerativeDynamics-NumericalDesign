"""Paper-grade visualization for posterior SMC results on 2D Kolmogorov forecasting.

Reads figs/posterior/lag{1,10,40}_seed{0,1}.json and emits:
  - ess_curves.{pdf,png}       : ESS vs EM step, mean ± std over (IC × seed), per schedule
  - metric_bars.{pdf,png}      : 2 rows (lag 10, lag 40) × 4 cols (RMSE, spread, SSR, final ESS)
  - summary_posterior.md       : full table with ± std
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCHEDULES = ['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero']
PRETTY = {
    'follmer':  r'Föllmer $\sqrt{1-t^2}$',
    'baseline': r'baseline $1-t$',
    'triangle': r'$\sqrt{t(1-t)}$',
    'const':    r'const $g{=}1$',
    'sqrt_t':   r'$\sqrt{t}$',
    'zero':     r'ODE ($g{=}0$)',
}
COLOURS = {
    'follmer':  'tab:orange', 'baseline': 'tab:blue',
    'triangle': 'tab:green',  'const': 'tab:red',
    'sqrt_t': 'tab:purple',   'zero': 'tab:gray',
}


def load_jsons(in_dir):
    """Group posterior JSONs by (lag, seed)."""
    out = {}
    pat = re.compile(r'lag(?P<lag>\d+)_seed(?P<seed>\d+)\.json')
    for f in sorted(glob.glob(os.path.join(in_dir, 'lag*_seed*.json'))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        key = (int(m.group('lag')), int(m.group('seed')))
        with open(f) as h:
            out[key] = json.load(h)
    return out


def pooled_per_schedule(data, lag, schedule, key):
    """Across (seed, IC): gather all values of `key`. Returns np array."""
    vals = []
    for (lg, sd), d in data.items():
        if lg != lag:
            continue
        vals.extend([m[key] for m in d['results'][schedule]['per_ic']])
    return np.asarray(vals)


def pooled_ess_curve(data, lag, schedule):
    """Return (mean, std) of ESS curves stacked across (seed, IC)."""
    curves = []
    for (lg, sd), d in data.items():
        if lg != lag:
            continue
        curves.extend(d['results'][schedule]['ess_curve'])
    if not curves:
        return None, None
    arr = np.asarray(curves)           # (n_ic_total, n_steps)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else 0*arr.mean(axis=0)


def plot_ess_curves(data, lags, out):
    n = len(lags)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.6), squeeze=False)
    axes = axes[0]
    N_ens = data[next(iter(data))]['args']['n_particles']
    for j, lag in enumerate(lags):
        ax = axes[j]
        for sc in SCHEDULES:
            mean, std = pooled_ess_curve(data, lag, sc)
            if mean is None:
                continue
            steps = np.arange(len(mean))
            ax.plot(steps, mean, color=COLOURS[sc], lw=2, label=PRETTY[sc])
            ax.fill_between(steps, mean - std, mean + std, color=COLOURS[sc], alpha=0.15)
        ax.set_title(f'lag = {lag}')
        ax.set_xlabel('EM step')
        if j == 0:
            ax.set_ylabel(f'ESS (out of {N_ens})')
        ax.grid(alpha=0.3)
        if j == n - 1:
            ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def plot_metric_bars(data, lags, out):
    """Metric bars per schedule, 1 row per lag × 4 cols (RMSE, spread, SSR = spread/RMSE, finalESS).
    Pooled mean ± std across (seed, IC)."""
    metrics = [('rmse', 'posterior RMSE'),
               ('spread', 'posterior spread'),
               ('ssr', 'SSR = spread / RMSE'),
               ('final_ess', 'final ESS')]
    n_rows = len(lags); n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows), squeeze=False)
    N_ens = data[next(iter(data))]['args']['n_particles']
    for ri, lag in enumerate(lags):
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri, ci]
            means, stds = [], []
            for sc in SCHEDULES:
                if mkey == 'ssr':
                    r = pooled_per_schedule(data, lag, sc, 'rmse')
                    s = pooled_per_schedule(data, lag, sc, 'spread')
                    ssr_vals = s / np.clip(r, 1e-6, None)
                    means.append(float(ssr_vals.mean())); stds.append(float(ssr_vals.std(ddof=1)))
                else:
                    vals = pooled_per_schedule(data, lag, sc, mkey)
                    means.append(float(vals.mean())); stds.append(float(vals.std(ddof=1)))
            # highlight best: min for RMSE, max for final_ess, closest-to-1 for SSR
            if mkey == 'final_ess':
                best = int(np.argmax(means))
            elif mkey == 'ssr':
                best = int(np.argmin(np.abs(np.asarray(means) - 1.0)))
            else:
                best = int(np.argmin(means))
            colors = ['lightsteelblue'] * len(SCHEDULES)
            colors[best] = 'tab:orange'
            x = np.arange(len(SCHEDULES))
            ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels([PRETTY[sc] for sc in SCHEDULES], rotation=30, ha='right', fontsize=8)
            if ri == 0:
                ax.set_title(mlabel)
            if ci == 0:
                ax.set_ylabel(f'lag = {lag}')
            if mkey == 'ssr':
                ax.axhline(1.0, color='k', linestyle=':', lw=0.8, alpha=0.5)
            if mkey == 'final_ess':
                ax.set_ylim(0, N_ens * 1.05)
                ax.axhline(N_ens, color='k', linestyle=':', lw=0.8, alpha=0.3)
            ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def summary_table(data, lags):
    lines = ['# NS posterior-SMC summary (pooled across seeds × ICs)\n',
             'Metrics per (lag × schedule): **RMSE** (posterior mean vs truth), **spread** (ensemble std),',
             '**SSR** (spread/RMSE, 1.0 = calibrated), **final ESS** (out of N), **log-Z** (mean ± std across seeds×ICs).\n']
    for lag in lags:
        lines.append(f'\n## lag = {lag}\n')
        lines.append('| schedule | RMSE | spread | SSR | final ESS | log-Z |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        # Determine best per column
        rmse_means = [pooled_per_schedule(data, lag, sc, 'rmse').mean() for sc in SCHEDULES]
        ess_means = [pooled_per_schedule(data, lag, sc, 'final_ess').mean() for sc in SCHEDULES]
        ssr_means = [pooled_per_schedule(data, lag, sc, 'spread').mean() /
                     max(pooled_per_schedule(data, lag, sc, 'rmse').mean(), 1e-6) for sc in SCHEDULES]
        best_rmse = SCHEDULES[int(np.argmin(rmse_means))]
        best_ess = SCHEDULES[int(np.argmax(ess_means))]
        best_ssr = SCHEDULES[int(np.argmin(np.abs(np.asarray(ssr_means) - 1.0)))]
        for sc in SCHEDULES:
            r = pooled_per_schedule(data, lag, sc, 'rmse')
            sp = pooled_per_schedule(data, lag, sc, 'spread')
            fe = pooled_per_schedule(data, lag, sc, 'final_ess')
            lz = pooled_per_schedule(data, lag, sc, 'logZ')
            ssr = sp / np.clip(r, 1e-6, None)
            rmse_str = f"{r.mean():.3e} ± {r.std(ddof=1):.1e}"
            spread_str = f"{sp.mean():.3e} ± {sp.std(ddof=1):.1e}"
            ssr_str = f"{ssr.mean():.2f} ± {ssr.std(ddof=1):.2f}"
            ess_str = f"{fe.mean():.1f} ± {fe.std(ddof=1):.1f}"
            lz_str = f"{lz.mean():.1f} ± {lz.std(ddof=1):.1f}"
            if sc == best_rmse: rmse_str = f'**{rmse_str}**'
            if sc == best_ess: ess_str = f'**{ess_str}**'
            if sc == best_ssr: ssr_str = f'**{ssr_str}**'
            lines.append(f'| {PRETTY[sc]} | {rmse_str} | {spread_str} | {ssr_str} | {ess_str} | {lz_str} |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=str, default='./figs/posterior')
    p.add_argument('--lags', type=int, nargs='+', default=[10, 40])
    args = p.parse_args()
    data = load_jsons(args.in_dir)
    if not data:
        print(f'No data in {args.in_dir}'); return
    plot_ess_curves(data, args.lags, os.path.join(args.in_dir, 'ess_curves.pdf'))
    plot_metric_bars(data, args.lags, os.path.join(args.in_dir, 'metric_bars.pdf'))
    md = summary_table(data, args.lags)
    with open(os.path.join(args.in_dir, 'summary_posterior.md'), 'w') as f:
        f.write(md)
    print('\n' + md)
    print(f"\n[saved] {args.in_dir}/summary_posterior.md")


if __name__ == '__main__':
    main()
