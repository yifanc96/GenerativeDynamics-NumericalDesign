"""Render a markdown table of terminal marginal accuracy metrics per
(target × schedule). Reads zps_<target>_results.json written by compare_zps.py.

Usage:
    python report_table.py --out figs/zps_table.md
"""
import argparse
import json
import os

import numpy as np


METRIC_KEYS = [
    ('kl_path',   'path KL',     '.2e'),   # filled from 'kl'
    ('kl_hist',   'marg KL',     '.2e'),
    ('w1',        'W1',          '.3e'),
    ('w2',        'W2',          '.3e'),
    ('mmd2',      'MMD$^2$',     '.2e'),
    ('mom_mean',  '$\\Delta\\mu$', '.2e'),
    ('mom_std',   '$\\Delta\\sigma$', '.2e'),
    ('mom_skew',  '$\\Delta$skew', '.2e'),
    ('mom_kurt',  '$\\Delta$kurt', '.2e'),
]

SCHEDULE_ORDER = ['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero']
PRETTY_SCHED = {
    'follmer':   'Föllmer $\\sqrt{1-t^2}$',
    'baseline':  'baseline $1-t$',
    'triangle':  '$\\sqrt{t(1-t)}$',
    'const':     'const ($g{=}1$)',
    'sqrt_t':    '$\\sqrt{t}$',
    'zero':      'ODE ($g{=}0$)',
}
PRETTY_TARGET = {
    'gaussian1d':      r'Target A: $\mathcal{N}(\mu,\sigma^2)$',
    'bimodal1d':       r'Target B: bimodal GMM',
    'ou_forecast':     r'Target C: OU forecasting (marginal over $Y_s$)',
    'gmm2d_forecast':  r'Target D: 2D GMM jump-forecast (conditional, marginalised over $x_0$)',
}


def fmt(value, std, style):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '—'
    return f'{value:{style}} ± {std:{style}}'


def render(results_by_target):
    lines = []
    lines.append('# Terminal marginal accuracy across schedules (5-seed mean ± std)\n')
    lines.append('All metrics are computed from 40 000 EM samples at '
                 '$t = 1 - 10^{-3}$ vs. the analytic target. Path-KL is the '
                 'Girsanov MC estimator (see README). ODE ($g{=}0$) has no '
                 'path-KL (Girsanov diverges).\n')
    for target, agg in results_by_target.items():
        lines.append(f'## {PRETTY_TARGET.get(target, target)}\n')
        header = '| schedule | ' + ' | '.join(m[1] for m in METRIC_KEYS) + ' |'
        sep = '|---|' + '|'.join(['---:'] * len(METRIC_KEYS)) + '|'
        lines += [header, sep]
        # find best (min) per metric (excluding ODE for path-KL)
        bests = {}
        for key, _, _ in METRIC_KEYS:
            jkey = 'kl' if key == 'kl_path' else key
            vals = []
            for s in SCHEDULE_ORDER:
                if s == 'zero' and key == 'kl_path':
                    vals.append(float('inf'))
                    continue
                v = agg[s].get(f'{jkey}_mean', float('nan'))
                vals.append(v if not np.isnan(v) else float('inf'))
            bests[key] = SCHEDULE_ORDER[int(np.argmin(vals))]
        for s in SCHEDULE_ORDER:
            row = [PRETTY_SCHED[s]]
            for key, _, style in METRIC_KEYS:
                jkey = 'kl' if key == 'kl_path' else key
                if s == 'zero' and key == 'kl_path':
                    row.append('—')
                    continue
                m = agg[s].get(f'{jkey}_mean', float('nan'))
                sd = agg[s].get(f'{jkey}_std', float('nan'))
                cell = fmt(m, sd, style)
                if bests[key] == s and cell != '—':
                    cell = f'**{cell}**'
                row.append(cell)
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=str, default='./figs')
    p.add_argument('--targets', type=str, nargs='+',
                   default=['gaussian1d', 'bimodal1d', 'ou_forecast', 'gmm2d_forecast'])
    p.add_argument('--out', type=str, default='./figs/zps_table.md')
    args = p.parse_args()
    results_by_target = {}
    for t in args.targets:
        with open(os.path.join(args.in_dir, f'zps_{t}_results.json')) as f:
            results_by_target[t] = json.load(f)['agg']
    md = render(results_by_target)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(md)
    print(md)
    print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()
