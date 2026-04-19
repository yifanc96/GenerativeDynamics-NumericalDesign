"""Generate vorticity-field ensemble grids + energy-spectrum overlays
for a chosen (lag, seed) checkpoint."""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.fft as fft

from data import load_snapshots, split_train_val_test, PairDataset
from interpolant_ns import ZPSInterpolant
from network_ns import DriftNet
from schedules_ns import list_schedules, make_g
from drift_compose_ns import compose_drift
from sampler_ns import em_sample
from metrics_ns import radial_spectrum


SCHEDULE_ORDER = ['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero']
PRETTY = {
    'follmer':  r'Föllmer $\sqrt{1-t^2}$',
    'baseline': r'baseline $1-t$',
    'triangle': r'$\sqrt{t(1-t)}$',
    'const':    r'const $g{=}1$',
    'sqrt_t':   r'$\sqrt{t}$',
    'zero':     r'ODE ($g{=}0$)',
}


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']
    net = DriftNet(unet_channels=args['unet_channels'],
                   unet_dim_mults=tuple(args['unet_dim_mults'])).to(device)
    net.load_state_dict(ck['state'])
    net.eval()
    return net, args, ck['norm']


@torch.no_grad()
def generate_ensemble(ckpt_path, n_test, n_ens, n_em, device):
    net, ck_args, norm = load_ckpt(ckpt_path, device)
    omega_all, _ = load_snapshots(ck_args['data'])
    _, _, test_idx = split_train_val_test(
        omega_all.shape[0], ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])
    test_ds = PairDataset(omega_all[test_idx], ck_args['lag'],
                          coarsen_factor=ck_args['coarsen'])
    H = omega_all.shape[-1]
    # pick n_test pairs
    idxs = torch.linspace(0, len(test_ds) - 1, n_test).long()
    x0 = torch.stack([test_ds[i]['x0'] for i in idxs])       # (n, 1, H, H)
    x0_up = torch.stack([test_ds[i]['x0_up'] for i in idxs])
    x1 = torch.stack([test_ds[i]['x1'] for i in idxs])

    def b_fn(x, t, x0): return net(x, t, x0)
    outs = {}
    x0_up_dev = x0_up.to(device) / norm
    for name in SCHEDULE_ORDER:
        g_fn = make_g(name)
        bg = compose_drift(b_fn, g_fn)
        ens = torch.zeros(n_ens, n_test, 1, H, H, device=device)
        for k in range(n_ens):
            ens[k] = em_sample(bg, g_fn, (n_test, 1, H, H),
                               n_steps=n_em, t_min=ck_args.get('t_eps', 1e-3),
                               t_max=1.0 - ck_args.get('t_eps', 1e-3),
                               cond=(x0_up_dev,), device=device)
        outs[name] = (ens * norm).cpu()
    return x0, x0_up, x1, outs, ck_args, H


def plot_vorticity_grid(x0, x0_up, x1, ens_dict, n_test, out, vlim=None):
    ncols = 1 + 1 + 1 + len(SCHEDULE_ORDER)  # input, coarse, truth, + 6 schedules
    nrows = n_test
    H = x0.shape[-1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.1 * nrows), squeeze=False)
    if vlim is None:
        vlim = float(x1.abs().quantile(0.995))
    cmap = sns.cm.icefire
    titles = [r'$\omega_t$', r'$\tilde\omega_t$ (coarse)', r'$\omega_{t+\tau}$ (truth)'] + [PRETTY[s] for s in SCHEDULE_ORDER]
    for i in range(nrows):
        for j, title in enumerate(titles):
            if j == 0:
                im = x0[i, 0].numpy()
            elif j == 1:
                im = x0_up[i, 0].numpy()
            elif j == 2:
                im = x1[i, 0].numpy()
            else:
                s = SCHEDULE_ORDER[j - 3]
                im = ens_dict[s][0, i, 0].numpy()         # first ensemble member
            axes[i, j].imshow(im, cmap=cmap, vmin=-vlim, vmax=vlim)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def plot_spectrum_overlay(x1, ens_dict, out, lag):
    # Use ensemble-average spectrum per schedule; truth spectrum.
    COLOURS = {'follmer': 'tab:orange', 'baseline': 'tab:blue', 'triangle': 'tab:green',
               'const': 'tab:red', 'sqrt_t': 'tab:purple', 'zero': 'tab:gray'}
    k, spec_truth = radial_spectrum(x1.squeeze(1))
    truth_mean = spec_truth.mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(k[1:].numpy(), truth_mean[1:], 'k-', lw=2, label='truth')
    for name in SCHEDULE_ORDER:
        ens = ens_dict[name]                    # (K, n, 1, H, H)
        flat = ens.reshape(-1, *ens.shape[-2:])
        _, spec = radial_spectrum(flat)
        mean_s = spec.mean(dim=0).numpy()
        ax.loglog(k[1:].numpy(), mean_s[1:], color=COLOURS[name], lw=1.4, label=PRETTY[name], alpha=0.85)
    ax.set_xlabel('$k$'); ax.set_ylabel(r'$E(k)$')
    ax.set_title(f'Ensemble-averaged energy spectrum, forecast lag = {lag}')
    ax.grid(which='both', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.savefig(out.replace('.pdf', '.png'), dpi=130)
    plt.close(fig)
    print(f'[saved] {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./figs')
    p.add_argument('--n_test', type=int, default=4)
    p.add_argument('--n_ens', type=int, default=20)
    p.add_argument('--n_em', type=int, default=100)
    p.add_argument('--tag', type=str, default='lag10')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    x0, x0_up, x1, ens, ck_args, H = generate_ensemble(
        args.ckpt, args.n_test, args.n_ens, args.n_em, device)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_vorticity_grid(x0, x0_up, x1, ens, args.n_test,
                        os.path.join(args.out_dir, f'vorticity_grid_{args.tag}.pdf'))
    plot_spectrum_overlay(x1, ens,
                          os.path.join(args.out_dir, f'spectrum_overlay_{args.tag}.pdf'),
                          lag=ck_args['lag'])


if __name__ == '__main__':
    main()
