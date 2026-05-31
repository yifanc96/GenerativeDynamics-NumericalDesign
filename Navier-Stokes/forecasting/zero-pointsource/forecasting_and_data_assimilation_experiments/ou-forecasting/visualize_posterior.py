"""Produce comprehensive posterior-SMC figures + tables for the 1D zps study.

Reads all JSONs in --in_dir and emits:
  - <in>/summary_table.md        — full metric table (schedule × sigma × proposal)
  - <in>/ess_curves_<target>_<proposal>.pdf    — ESS vs sampler step, one panel per sigma
  - <in>/w2_bars_<target>.pdf   — W2-to-analytic bar chart, grouped by sigma
  - <in>/metric_grid.pdf        — 2x3 panel: {gaussian,bimodal} x {W2, log-Z var, final ESS}
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
    """Group JSON files by (target, proposal, sigma)."""
    data = {}
    pat = re.compile(r'posterior_(?P<target>[a-z0-9]+)_(?P<proposal>guided|uncond)_sigma(?P<sigma>[0-9.]+)\.json')
    for f in sorted(glob.glob(os.path.join(in_dir, 'posterior_*_*_sigma*.json'))):
        name = os.path.basename(f)
        m = pat.match(name)
        if not m:
            continue
        key = (m.group('target'), m.group('proposal'), float(m.group('sigma')))
        with open(f) as h:
            data[key] = json.load(h)
    return data


def summary_table(data):
    """Return a markdown-formatted summary table."""
    lines = ['# Posterior-sampling summary\n',
             'Metrics: **W2 to analytic posterior** (lower better), **final ESS** (higher better),',
             '**logZ std** (lower better = more consistent estimator across seeds), RMSE on posterior mean.\n']
    targets = sorted({k[0] for k in data})
    for target in targets:
        lines.append(f'\n## Target: {target}\n')
        for proposal in ['guided', 'uncond']:
            sigmas = sorted({k[2] for k in data if k[0] == target and k[1] == proposal})
            if not sigmas:
                continue
            lines.append(f'\n### proposal = {proposal}\n')
            cols = ['schedule'] + [f'σ={s}: W2' for s in sigmas] + [f'σ={s}: fESS' for s in sigmas] + [f'σ={s}: logZσ' for s in sigmas]
            lines.append('| ' + ' | '.join(cols) + ' |')
            lines.append('|---|' + '|'.join(['---:'] * (len(cols) - 1)) + '|')
            # For highlighting: per column (metric × sigma), find best schedule.
            best = {}
            for s in sigmas:
                d_s = data[(target, proposal, s)]['summary']
                # best W2
                w2 = [d_s[sc]['w2_mean'] for sc in SCHEDULES]
                best[('w2', s)] = SCHEDULES[int(np.argmin(w2))]
                ess_v = [d_s[sc]['final_ess_mean'] for sc in SCHEDULES]
                best[('fess', s)] = SCHEDULES[int(np.argmax(ess_v))]
            for sc in SCHEDULES:
                row = [PRETTY[sc]]
                for s in sigmas:
                    d_s = data[(target, proposal, s)]['summary'][sc]
                    v = d_s['w2_mean']
                    bold = sc == best[('w2', s)]
                    cell = f"{v:.3f}±{d_s['w2_std']:.3f}"
                    if bold: cell = f'**{cell}**'
                    row.append(cell)
                for s in sigmas:
                    d_s = data[(target, proposal, s)]['summary'][sc]
                    v = d_s['final_ess_mean']
                    bold = sc == best[('fess', s)]
                    cell = f"{v:.1f}"
                    if bold: cell = f'**{cell}**'
                    row.append(cell)
                for s in sigmas:
                    d_s = data[(target, proposal, s)]['summary'][sc]
                    v = d_s['logZ_std']
                    row.append(f"{v:.3f}")
                lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def ess_curves_figure(data, target, proposal, out):
    """One panel per sigma, showing ESS-vs-step curves per schedule (mean ± std across seeds)."""
    sigmas = sorted({k[2] for k in data if k[0] == target and k[1] == proposal})
    n = len(sigmas)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    axes = axes[0]
    for j, sig in enumerate(sigmas):
        ax = axes[j]
        d = data[(target, proposal, sig)]['summary']
        for sc in SCHEDULES:
            curves = d[sc]['ess_curves']
            if not curves:
                continue
            arr = np.asarray(curves)   # (seeds, steps)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else 0 * mean
            steps = np.arange(len(mean))
            ax.plot(steps, mean, color=COLOURS[sc], lw=2, label=PRETTY[sc])
            ax.fill_between(steps, mean - std, mean + std, color=COLOURS[sc], alpha=0.15)
        ax.set_title(f'$\\sigma_y={sig}$')
        ax.set_xlabel('EM step')
        if j == 0:
            ax.set_ylabel('effective sample size')
        ax.grid(alpha=0.3)
        if j == n - 1:
            ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{target} — ESS curves — {proposal} proposal')
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def w2_bar_figure(data, target, out):
    """Bars of W2-to-analytic, grouped by sigma, one panel per proposal."""
    props = ['guided', 'uncond']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    axes = axes[0]
    for pi, prop in enumerate(props):
        sigmas = sorted({k[2] for k in data if k[0] == target and k[1] == prop})
        ax = axes[pi]
        n_s = len(sigmas); n_sc = len(SCHEDULES)
        width = 0.8 / n_sc
        x = np.arange(n_s)
        for i, sc in enumerate(SCHEDULES):
            means = []; stds = []
            for s in sigmas:
                if (target, prop, s) in data:
                    d = data[(target, prop, s)]['summary'][sc]
                    means.append(d['w2_mean']); stds.append(d['w2_std'])
                else:
                    means.append(0); stds.append(0)
            pos = x + (i - (n_sc - 1) / 2) * width
            ax.bar(pos, means, yerr=stds, width=width, color=COLOURS[sc],
                   edgecolor='black', capsize=2, label=PRETTY[sc])
        ax.set_xticks(x)
        ax.set_xticklabels([f'$\\sigma_y={s}$' for s in sigmas])
        ax.set_title(f'{target} — {prop} proposal')
        ax.set_ylabel('$W_2$ to analytic posterior')
        ax.grid(alpha=0.3, axis='y')
        if pi == 1:
            ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def metric_grid(data, out):
    """2×4 grid: rows = {gaussian, bimodal}, cols = {W2, posterior-mean RMSE, final ESS, logZ std}.
    Lowest-sigma (strong obs) slice with guided proposal as headline."""
    targets = ['gaussian1d', 'bimodal1d']
    metrics = [('w2', '$W_2$ to analytic posterior'),
               ('rmse', 'posterior-mean |Δμ|'),
               ('fess', 'final ESS'),
               ('logZsig', 'log-$Z$ std across seeds')]
    fig, axes = plt.subplots(len(targets), len(metrics),
                             figsize=(4.2 * len(metrics), 3.2 * len(targets)), squeeze=False)
    for ri, target in enumerate(targets):
        sigmas = sorted({k[2] for k in data if k[0] == target and k[1] == 'guided'})
        sig = sigmas[0] if sigmas else None
        if sig is None or (target, 'guided', sig) not in data:
            continue
        summary = data[(target, 'guided', sig)]['summary']
        for ci, (m_key, m_label) in enumerate(metrics):
            ax = axes[ri, ci]
            vals = []; stds = []
            for sc in SCHEDULES:
                d = summary[sc]
                if m_key == 'w2':
                    vals.append(d['w2_mean']); stds.append(d['w2_std'])
                elif m_key == 'rmse':
                    vals.append(d['rmse_mean']); stds.append(d['rmse_std'])
                elif m_key == 'fess':
                    vals.append(d['final_ess_mean']); stds.append(d['final_ess_std'])
                elif m_key == 'logZsig':
                    vals.append(d['logZ_std']); stds.append(0)
            if m_key == 'fess':
                best_idx = int(np.argmax(vals))
            else:
                best_idx = int(np.argmin(vals))
            bars = ax.bar(np.arange(len(SCHEDULES)), vals, yerr=stds, color='lightsteelblue',
                          edgecolor='black', capsize=3)
            bars[best_idx].set_color('tab:orange')
            ax.set_xticks(np.arange(len(SCHEDULES)))
            ax.set_xticklabels([PRETTY[sc] for sc in SCHEDULES], rotation=30, ha='right', fontsize=8)
            ax.set_title(f'{target}: {m_label}   ($\\sigma_y={sig}$)' if ri == 0 else f'{m_label}')
            if ri == 0 and ci == 0:
                ax.annotate(f'{target} row', xy=(-0.35, 0.5), xycoords='axes fraction', rotation=90, fontsize=9)
            ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def sigma_sweep_figure(data, out):
    """1x4 grid: {gaussian W2, gaussian RMSE, bimodal W2, bimodal RMSE} vs sigma_y.
    Solid = guided proposal, dashed = uncond."""
    targets = ['gaussian1d', 'bimodal1d']
    metrics = [('w2_mean', 'w2_std', '$W_2$ to analytic'),
               ('rmse_mean', 'rmse_std', 'posterior-mean $|\\Delta\\mu|$')]
    n_panels = len(targets) * len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.4), squeeze=False)
    axes = axes[0]
    p_i = 0
    for target in targets:
        for mkey, mstd, mlabel in metrics:
            ax = axes[p_i]
            for sc in SCHEDULES:
                for prop, ls in [('guided', '-'), ('uncond', '--')]:
                    sigmas = sorted({k[2] for k in data if k[0] == target and k[1] == prop})
                    if not sigmas: continue
                    ys, es = [], []
                    for s in sigmas:
                        if (target, prop, s) not in data: ys.append(float('nan')); es.append(0); continue
                        d = data[(target, prop, s)]['summary'][sc]
                        ys.append(d[mkey]); es.append(d[mstd])
                    ax.errorbar(sigmas, ys, yerr=es, color=COLOURS[sc], linestyle=ls,
                                lw=1.6, capsize=3, alpha=0.9,
                                label=f'{PRETTY[sc]} ({prop})' if p_i == 0 else None)
            ax.set_xlabel('$\\sigma_y$ (obs noise)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{target}\n{mlabel}')
            ax.grid(alpha=0.3, which='both')
            p_i += 1
    fig.legend(loc='center right', bbox_to_anchor=(1.18, 0.5), fontsize=7, ncol=1)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=str, default='./figs_posterior')
    args = p.parse_args()
    data = load_jsons(args.in_dir)
    if not data:
        print(f'No data found in {args.in_dir}'); return
    # Save table
    md = summary_table(data)
    with open(os.path.join(args.in_dir, 'summary_table.md'), 'w') as f:
        f.write(md)
    print(md)
    print(f'\n[saved] {args.in_dir}/summary_table.md')
    # ESS curves
    for target in ['gaussian1d', 'bimodal1d']:
        for prop in ['guided', 'uncond']:
            if any(k[0] == target and k[1] == prop for k in data):
                out = os.path.join(args.in_dir, f'ess_curves_{target}_{prop}.pdf')
                ess_curves_figure(data, target, prop, out)
    # W2 bars
    for target in ['gaussian1d', 'bimodal1d']:
        if any(k[0] == target for k in data):
            out = os.path.join(args.in_dir, f'w2_bars_{target}.pdf')
            w2_bar_figure(data, target, out)
    # Headline 2x4
    metric_grid(data, os.path.join(args.in_dir, 'metric_grid.pdf'))
    # Sigma-sweep
    sigma_sweep_figure(data, os.path.join(args.in_dir, 'sigma_sweep.pdf'))


if __name__ == '__main__':
    main()
