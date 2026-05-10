"""Generate the three NS G=128 figures for multiscale_experiments.tex.

Reads results/eval_ns_unified.npz produced by eval_ns_unified.py.
Outputs:
  figs/ns_spectrum_compare.png    - E(k) for truth + best-NFE per method
  figs/ns_relerror_perbin.png     - per-bin rel error with std bands
  figs/ns_nfe_sweep.png           - mean rel <=60 vs NFE per method
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'figure.figsize': (8, 5)})

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, '..', 'figs')
os.makedirs(FIGS, exist_ok=True)

d = np.load(os.path.join(ROOT, 'results', 'eval_ns_unified.npz'), allow_pickle=True)
spec_truth = d['spec_truth_mean']
labels = json.loads(str(d['method_labels']))
kvals = d['kvals']
n_k = len(spec_truth)

NAMES = {
    'mfm':       'Multiscale FM (v1, 150k)',
    'mmf':       'Multiscale meanflow (250k)',
    'msc':       'Multiscale shortcut (250k)',
    'sfm_gauss': 'Single-scale FM, gauss (50k)',
    'sfm_std':   'Single-scale FM standard (50k)',
}
COLORS = {
    'mfm':       'tab:blue',
    'mmf':       'tab:green',
    'msc':       'tab:purple',
    'sfm_gauss': 'tab:orange',
    'sfm_std':   'tab:red',
}

# ============== Figure 1: E(k) ==============
fig, ax = plt.subplots()
ax.loglog(kvals, spec_truth, 'k--', lw=2.5, label='Truth')
best_nfe = {'mfm': 64, 'mmf': 32, 'msc': 8, 'sfm_gauss': 64, 'sfm_std': 64}
for label, mtype, _ in labels:
    nfe = best_nfe.get(label)
    if nfe is None: continue
    key = f'spec_{label}_NFE{nfe}'
    if key not in d: continue
    spec = d[key]
    mean = spec.mean(0); std = spec.std(0)
    ax.loglog(kvals, mean, color=COLORS.get(label, 'gray'),
              label=f'{NAMES.get(label, label)} (NFE={nfe})')
    ax.fill_between(kvals, np.maximum(mean - std, 1e-15), mean + std,
                    color=COLORS.get(label, 'gray'), alpha=0.2)
ax.set_xlabel('Wavenumber $k$'); ax.set_ylabel('Enstrophy spectrum')
ax.set_title('NS $G=128$: enstrophy spectrum at best NFE')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'ns_spectrum_compare.png'), dpi=130)
plt.close()
print('wrote ns_spectrum_compare.png')

# ============== Figure 2: per-bin rel error ==============
fig, ax = plt.subplots()
for label, mtype, _ in labels:
    nfe = best_nfe.get(label)
    if nfe is None: continue
    key = f'rel_{label}_NFE{nfe}'
    if key not in d: continue
    rel = d[key]
    mean = rel.mean(0); std = rel.std(0)
    ax.semilogy(kvals, mean, color=COLORS.get(label, 'gray'),
                label=f'{NAMES.get(label, label)} (NFE={nfe})')
    ax.fill_between(kvals, np.maximum(mean - std, 1e-6), mean + std,
                    color=COLORS.get(label, 'gray'), alpha=0.2)
ax.axhline(0.1, color='r', ls='--', alpha=0.5, label='10% rel error')
ax.axvline(60, color='gray', ls='-.', alpha=0.7)
ax.text(60.5, 1.5, 'k=60 cap', rotation=90, fontsize=9, color='gray', va='top')
ax.set_xlabel('Wavenumber $k$')
ax.set_ylabel('Relative spectrum error')
ax.set_title('NS $G=128$: per-bin relative error vs.\\ wavenumber')
ax.set_xlim(1, n_k); ax.set_ylim(1e-3, 100)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'ns_relerror_perbin.png'), dpi=130)
plt.close()
print('wrote ns_relerror_perbin.png')

# ============== Figure 3: NFE sweep ==============
fig, ax = plt.subplots()
for label, mtype, nfe_list in labels:
    means, stds = [], []
    for nfe in nfe_list:
        key = f'rel_{label}_NFE{nfe}'
        if key not in d:
            means.append(np.nan); stds.append(np.nan); continue
        rel = d[key]
        mr = rel[:, :60].mean(axis=1)
        means.append(mr.mean()); stds.append(mr.std())
    means = np.array(means); stds = np.array(stds)
    valid = ~np.isnan(means)
    ax.errorbar(np.array(nfe_list)[valid], means[valid], yerr=stds[valid],
                color=COLORS.get(label, 'gray'),
                label=NAMES.get(label, label), marker='o', capsize=4)
ax.axhline(0.1, color='r', ls=':', alpha=0.5, label='10% threshold')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('NFE')
ax.set_ylabel(r'$\overline{\mathrm{rel}}_{k\leq 60}$')
ax.set_title('NS $G=128$: accuracy vs.\\ NFE (mean $\\pm$ std over seeds)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'ns_nfe_sweep.png'), dpi=130)
plt.close()
print('wrote ns_nfe_sweep.png')

# Print table
print('\nNS results table:')
all_nfes = sorted({nfe for _, _, nl in labels for nfe in nl})
print(f"{'Method':<35}", end='')
for nfe in all_nfes: print(f' NFE={nfe:<4}', end='')
print()
for label, mtype, nfe_list in labels:
    print(f"{NAMES.get(label, label):<35}", end='')
    for nfe in all_nfes:
        if nfe in nfe_list:
            key = f'rel_{label}_NFE{nfe}'
            if key in d:
                rel = d[key]; mr = rel[:, :60].mean(axis=1)
                print(f' {mr.mean():.3f}±{mr.std():.3f}', end='')
            else:
                print('  ---     ', end='')
        else:
            print('  ---     ', end='')
    print()
