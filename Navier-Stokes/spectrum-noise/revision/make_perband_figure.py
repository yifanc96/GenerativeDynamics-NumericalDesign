"""
Produce the 2x3 per-band error figure for the paper using already-computed values.

Hard-coded values are taken from prior runs:
  - 64x64 noise=10:  auto_lip_ns.py output (b58l1e6zx) + earlier eval_lip_transfer
  - 128x128 noise=10: auto_lip_ns.py output (b4vsydt63 / bu95f3gvl)

For 'Designed', we use the auto-tuned r values:
  - 64x64:  r=3.0e-4  (auto, c=1.0; sigma_min=4.4 vs sigma_train=10)
  - 128x128: r=1.1e-5 (auto, c=1.0; sigma_min=8.8 vs sigma_train=10)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 64×64, noise=10 ──
# (mean, std) over 3 seeds for {Linear, Designed} × {10, 20, 50} × {low, mid, high}
data_64 = {
    ('Linear', 10):   {'low': (0.0806, 0.0054), 'mid': (0.0603, 0.0010), 'high': (0.4210, 0.0110)},
    ('Linear', 20):   {'low': (0.0819, 0.0053), 'mid': (0.0559, 0.0057), 'high': (0.0540, 0.0074)},
    ('Linear', 50):   {'low': (0.0830, 0.0053), 'mid': (0.0658, 0.0057), 'high': (0.0323, 0.0115)},
    ('Designed', 10): {'low': (0.0978, 0.0055), 'mid': (0.0331, 0.0059), 'high': (0.0414, 0.0042)},
    ('Designed', 20): {'low': (0.0974, 0.0054), 'mid': (0.0363, 0.0060), 'high': (0.0144, 0.0057)},
    ('Designed', 50): {'low': (0.0988, 0.0054), 'mid': (0.0333, 0.0060), 'high': (0.0148, 0.0015)},
}

# ── 128×128, noise=10 ──
data_128 = {
    ('Linear', 10):   {'low': (0.0716, 0.0118), 'mid': (0.0390, 0.0101), 'high': (3.5491, 0.0493)},
    ('Linear', 20):   {'low': (0.0728, 0.0118), 'mid': (0.0494, 0.0108), 'high': (0.8032, 0.0129)},
    ('Linear', 50):   {'low': (0.0732, 0.0118), 'mid': (0.0513, 0.0108), 'high': (0.2546, 0.0035)},
    ('Designed', 10): {'low': (0.0978, 0.0099), 'mid': (0.0189, 0.0030), 'high': (0.1629, 0.0089)},
    ('Designed', 20): {'low': (0.0899, 0.0122), 'mid': (0.0174, 0.0057), 'high': (0.1603, 0.0046)},
    ('Designed', 50): {'low': (0.0887, 0.0123), 'mid': (0.0174, 0.0061), 'high': (0.1790, 0.0049)},
}

steps = [10, 20, 50]
band_titles = {'mid': r'Mid band ($8 \leq k < 24$)',
               'high': r'High band ($k \geq 24$)'}
band_keys = ['mid', 'high']
res_data = [(64, data_64), (128, data_128)]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)

for row, (hi, data) in enumerate(res_data):
    for col, bn in enumerate(band_keys):
        ax = axes[row, col]
        for method, color, marker, ls in [('Linear', 'tab:red', 's', '--'),
                                          ('Designed', 'tab:blue', 'o', '-')]:
            ms = [data[(method, s)][bn][0] for s in steps]
            sds = [data[(method, s)][bn][1] for s in steps]
            ax.errorbar(steps, ms, yerr=sds, marker=marker, color=color, ls=ls,
                        lw=1.8, ms=8, capsize=4, label=method)
        ax.set_yscale('log')
        ax.set_xticks(steps)
        ax.grid(True, which='both', alpha=0.3)
        if row == 0:
            ax.set_title(band_titles[bn], fontsize=13)
        if col == 0:
            ax.set_ylabel(f"${hi}\\times{hi}$\nrelative error", fontsize=12)
        if row == 1:
            ax.set_xlabel('Number of RK4 steps', fontsize=12)
        if row == 0 and col == 1:
            ax.legend(fontsize=11, loc='best')

plt.tight_layout()
out_path = '../images/NS-perband-error.pdf'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
