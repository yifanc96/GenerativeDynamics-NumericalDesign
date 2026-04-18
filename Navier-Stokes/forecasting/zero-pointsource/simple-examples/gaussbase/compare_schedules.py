"""Load a trained (v_theta, s_theta) checkpoint, sweep g(t) at sample time,
aggregate path-KL and marginal W2 across seeds, and make summary plots.

Usage:
    python compare_schedules.py --target gaussian1d --seeds 0 1 2 3 4 --out_dir ./figs
"""
import argparse
import json
import os

import numpy as np
import torch

from interpolant import GaussianBaseInterpolant
from networks import MLPNet
from targets import make_target
from schedules import list_schedules, make_g, compose_drift
from sampler import em_sample
from metrics import wasserstein_1d, path_kl_girsanov


def load_checkpoint(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck['args']
    target = make_target(args['target'], device=device)
    cond_dim = 1 if target.conditional else 0
    v_net = MLPNet(x_dim=target.dim, cond_dim=cond_dim, hidden=args['hidden'],
                   n_layers=args['n_layers'], time_embed=args['time_embed']).to(device)
    s_net = MLPNet(x_dim=target.dim, cond_dim=cond_dim, hidden=args['hidden'],
                   n_layers=args['n_layers'], time_embed=args['time_embed']).to(device)
    v_net.load_state_dict(ck['v_state'])
    s_net.load_state_dict(ck['s_state'])
    v_net.eval(); s_net.eval()
    return v_net, s_net, target, args


def evaluate_one(ckpt_path, args, device):
    v_net, s_net, target, ck_args = load_checkpoint(ckpt_path, device)
    ip = GaussianBaseInterpolant()
    epsilon = args.epsilon if args.epsilon > 0 else ck_args['epsilon']
    t_eps = ck_args.get('t_eps', 1e-3)

    def v_fn(x, t, *c): return v_net(x, t, *c)
    def s_fn(x, t, *c): return s_net(x, t, *c)

    def av(x, t, *c):
        return target.v_star(x, t, ip, *c) if target.conditional else target.v_star(x, t, ip)

    def as_(x, t, *c):
        return target.s_star(x, t, ip, *c) if target.conditional else target.s_star(x, t, ip)

    torch.manual_seed(args.eval_seed)
    if target.conditional:
        (y_ref,) = target.sample_cond(args.n_samples)
        x1_ref, _ = target.sample_x1(args.n_samples, y=y_ref)
        truth = x1_ref
    else:
        truth = target.sample_x1(args.n_samples)

    out = {}
    for name in list_schedules():
        g_fn = make_g(name, epsilon=epsilon, device=device)
        b_theta = compose_drift(v_fn, s_fn, g_fn)
        b_star = compose_drift(av, as_, g_fn)
        rec = {}
        if name != 'ode':
            rec['kl'] = path_kl_girsanov(b_theta, b_star, g_fn, ip, target,
                                         n_mc=args.n_mc, t_min=0.0, t_max=1.0 - t_eps,
                                         device=device, dtype=torch.float32)
        else:
            rec['kl'] = float('nan')
        if target.conditional:
            (y_s,) = target.sample_cond(args.n_samples)
            x1_s, _ = target.sample_x1(args.n_samples, y=y_s)
            samples = em_sample(b_theta, g_fn, args.n_samples, target.dim,
                                n_steps=args.n_em, t_min=0.0, t_max=1.0 - t_eps,
                                cond=(y_s,), device=device, init='gaussian')
            rec['w2'] = wasserstein_1d(samples, x1_s, p=2) if target.dim == 1 else float('nan')
        else:
            samples = em_sample(b_theta, g_fn, args.n_samples, target.dim,
                                n_steps=args.n_em, t_min=0.0, t_max=1.0 - t_eps,
                                device=device, init='gaussian')
            rec['w2'] = wasserstein_1d(samples, truth, p=2) if target.dim == 1 else float('nan')
        out[name] = rec
    return out


def aggregate(results_per_seed):
    schedules = list_schedules()
    agg = {s: {'kl_mean': [], 'kl_std': [], 'w2_mean': [], 'w2_std': []} for s in schedules}
    for s in schedules:
        kls = np.array([r[s]['kl'] for r in results_per_seed], dtype=float)
        w2s = np.array([r[s]['w2'] for r in results_per_seed], dtype=float)
        agg[s]['kl_mean'] = float(np.nanmean(kls))
        agg[s]['kl_std'] = float(np.nanstd(kls, ddof=1) if len(kls) > 1 else 0.0)
        agg[s]['w2_mean'] = float(np.nanmean(w2s))
        agg[s]['w2_std'] = float(np.nanstd(w2s, ddof=1) if len(w2s) > 1 else 0.0)
    return agg


def plot_bars(agg, metric, target_name, out_path, label_map=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    schedules = [s for s in list_schedules() if (metric != 'kl' or s != 'ode')]
    means = [agg[s][f'{metric}_mean'] for s in schedules]
    stds = [agg[s][f'{metric}_std'] for s in schedules]
    labels = [label_map.get(s, s) if label_map else s for s in schedules]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, edgecolor='black')
    # Highlight the lowest
    idx_min = int(np.argmin(means))
    bars[idx_min].set_color('tab:orange')
    ax.set_ylabel(f'path KL' if metric == 'kl' else 'marginal W2')
    ax.set_title(f'{target_name}: {metric}')
    ax.set_yscale('log' if metric == 'kl' else 'linear')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=str, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--runs_dir', type=str, default='./runs')
    p.add_argument('--out_dir', type=str, default='./figs')
    p.add_argument('--epsilon', type=float, default=-1.0, help='-1 = use ckpt epsilon')
    p.add_argument('--n_samples', type=int, default=40000)
    p.add_argument('--n_mc', type=int, default=80000)
    p.add_argument('--n_em', type=int, default=200)
    p.add_argument('--eval_seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        ckpt = os.path.join(args.runs_dir, f"{args.target}_seed{seed}.pt")
        print(f"[eval] {ckpt}")
        results.append(evaluate_one(ckpt, args, device))

    agg = aggregate(results)
    print("\n[summary]")
    for s, r in agg.items():
        print(f"  {s:12s}  KL={r['kl_mean']:.4e}±{r['kl_std']:.2e}  "
              f"W2={r['w2_mean']:.4e}±{r['w2_std']:.2e}")

    # Save raw + aggregate
    with open(os.path.join(args.out_dir, f"{args.target}_results.json"), 'w') as f:
        json.dump({'per_seed': results, 'agg': agg, 'seeds': args.seeds}, f, indent=2)

    plot_bars(agg, 'kl', args.target, os.path.join(args.out_dir, f"{args.target}_kl.pdf"))
    plot_bars(agg, 'w2', args.target, os.path.join(args.out_dir, f"{args.target}_w2.pdf"))
    print(f"[plots] saved to {args.out_dir}")


if __name__ == '__main__':
    main()
