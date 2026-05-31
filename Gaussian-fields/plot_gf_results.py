"""Generate the three figures for GF G=64 in multiscale_experiments.tex.

Reads results/eval_gf_unified.npz produced by eval_gf_unified.py.
Outputs:
  figs/gf_spectrum_compare.png    - E(k) for truth + best-NFE per method
  figs/gf_relerror_perbin.png     - per-bin rel error with std bands
  figs/gf_nfe_sweep.png           - mean rel <=30 vs NFE per method
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'figure.figsize': (8, 5)})

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, '..', 'figs')
os.makedirs(FIGS, exist_ok=True)

# load
d = np.load(os.path.join(ROOT, 'results', 'eval_gf_unified.npz'), allow_pickle=True)
spec_truth = d['spec_truth_mean']
labels = json.loads(str(d['method_labels']))  # [(label, mtype, nfe_list), ...]
n_k = len(spec_truth)
kvals = np.arange(1, n_k + 1)

# pretty names
NAMES = {
    'mfm_long':  'Multiscale FM (long, 470k)',
    'mmf':       'Multiscale meanflow (250k)',
    'msc':       'Multiscale shortcut (250k)',
    'smf_gauss': 'Single-scale meanflow, gauss (50k)',
    'mfm_short': 'Multiscale FM (orig, 100k)',
}
COLORS = {
    'mfm_long':  'tab:blue',
    'mmf':       'tab:green',
    'msc':       'tab:purple',
    'smf_gauss': 'tab:orange',
    'mfm_short': 'tab:cyan',
}

# ================ Figure 1: E(k) at best NFE ================
fig, ax = plt.subplots()
ax.loglog(kvals, spec_truth, 'k--', lw=2.5, label='Truth')
# pick a representative NFE per method
best_nfe = {'mfm_long': 64, 'mmf': 8, 'msc': 32, 'smf_gauss': 8}
for label, mtype, _ in labels:
    nfe = best_nfe.get(label)
    if nfe is None: continue
    key = f'spec_{label}_NFE{nfe}'
    if key not in d: continue
    spec = d[key]  # (n_seeds, n_k)
    mean = spec.mean(0); std = spec.std(0)
    ax.loglog(kvals, mean, color=COLORS.get(label, 'gray'),
              label=f'{NAMES.get(label, label)} (NFE={nfe})')
    ax.fill_between(kvals, np.maximum(mean - std, 1e-12), mean + std,
                    color=COLORS.get(label, 'gray'), alpha=0.2)
ax.set_xlabel('Wavenumber $k$'); ax.set_ylabel('Energy spectrum $E(k)$')
ax.set_title('Gaussian field $G=64$: spectrum at best NFE')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'gf_spectrum_compare.png'), dpi=130)
plt.close()
print('wrote gf_spectrum_compare.png')

# ================ Figure 2: per-bin rel error ================
fig, ax = plt.subplots()
for label, mtype, _ in labels:
    nfe = best_nfe.get(label)
    if nfe is None: continue
    key = f'rel_{label}_NFE{nfe}'
    if key not in d: continue
    rel = d[key]  # (n_seeds, n_k)
    mean = rel.mean(0); std = rel.std(0)
    ax.semilogy(kvals, mean, color=COLORS.get(label, 'gray'),
                label=f'{NAMES.get(label, label)} (NFE={nfe})')
    ax.fill_between(kvals, np.maximum(mean - std, 1e-6), mean + std,
                    color=COLORS.get(label, 'gray'), alpha=0.2)
ax.axhline(0.1, color='r', ls='--', alpha=0.5, label='10% rel error')
ax.axvline(31, color='gray', ls='-.', alpha=0.7)
ax.text(31.2, ax.get_ylim()[1] * 0.4 if ax.get_ylim()[1] > 0 else 1, 'Nyquist edge', rotation=0, fontsize=9, color='gray')
ax.set_xlabel('Wavenumber $k$')
ax.set_ylabel('Relative spectrum error')
ax.set_title('Gaussian field $G=64$: per-bin relative error vs.\\ wavenumber')
ax.set_xlim(1, n_k); ax.set_ylim(1e-3, 100)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'gf_relerror_perbin.png'), dpi=130)
plt.close()
print('wrote gf_relerror_perbin.png')

# ================ Figure 3: NFE sweep ================
fig, ax = plt.subplots()
for label, mtype, nfe_list in labels:
    means, stds = [], []
    for nfe in nfe_list:
        key = f'rel_{label}_NFE{nfe}'
        if key not in d:
            means.append(np.nan); stds.append(np.nan); continue
        rel = d[key]  # (n_seeds, n_k)
        mr30 = rel[:, :30].mean(axis=1)
        means.append(mr30.mean()); stds.append(mr30.std())
    means = np.array(means); stds = np.array(stds)
    valid = ~np.isnan(means)
    ax.errorbar(np.array(nfe_list)[valid], means[valid], yerr=stds[valid],
                color=COLORS.get(label, 'gray'),
                label=NAMES.get(label, label), marker='o', capsize=4)
ax.axhline(0.028, color='k', ls='--', alpha=0.6, label='Oracle ($0.028$)')
ax.axhline(0.1, color='r', ls=':', alpha=0.5, label='10% threshold')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('NFE')
ax.set_ylabel(r'$\overline{\mathrm{rel}}_{k\leq 30}$')
ax.set_title('Gaussian field $G=64$: accuracy vs.\\ NFE (mean $\\pm$ std over seeds)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'gf_nfe_sweep.png'), dpi=130)
plt.close()
print('wrote gf_nfe_sweep.png')

# print a tab-formatted table
print('\nLaTeX-friendly result table:')
print(f"{'Method':<35}", end='')
all_nfes = sorted({nfe for _, _, nl in labels for nfe in nl})
for nfe in all_nfes:
    print(f' NFE={nfe:<4}', end='')
print()
for label, mtype, nfe_list in labels:
    print(f"{NAMES.get(label, label):<35}", end='')
    for nfe in all_nfes:
        if nfe in nfe_list:
            key = f'rel_{label}_NFE{nfe}'
            if key in d:
                rel = d[key]
                mr = rel[:, :30].mean(axis=1)
                print(f' {mr.mean():.3f}±{mr.std():.3f}', end='')
            else:
                print('  ---     ', end='')
        else:
            print('  ---     ', end='')
    print()
