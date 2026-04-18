"""Load trained hat_b_theta checkpoints, sweep g schedules via eq (3.9),
aggregate path-KL and marginal W2 across seeds, save JSON + per-target plot.

Usage:
    python compare_zps.py --target gaussian1d --seeds 0 1 2 3 4
"""
import argparse
import json
import os

import numpy as np
import torch

from networks import MLPNet
from targets_zps import make_target
from schedules_zps import list_schedules, make_g
from drift_compose import compose_drift
from sampler import em_sample
from metrics import (wasserstein_1d, mmd_rbf, moment_errors, kl_analytic_1d,
                     path_kl_girsanov)


def load_checkpoint(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck['args']
    target = make_target(args['target'], device=device)
    cond_dim = 1 if target.conditional else 0
    b_net = MLPNet(x_dim=target.dim, cond_dim=cond_dim, hidden=args['hidden'],
                   n_layers=args['n_layers'], time_embed=args['time_embed']).to(device)
    b_net.load_state_dict(ck['b_state'])
    b_net.eval()
    return b_net, target, args


def evaluate_one(ckpt_path, args, device):
    b_net, target, ck_args = load_checkpoint(ckpt_path, device)
    t_eps = ck_args.get('t_eps', 1e-3)

    def b_theta_fn(x, t, *c):
        return b_net(x, t, *c)

    def b_star_fn(x, t, *c):
        return target.b_star(x, t, *c) if target.conditional else target.b_star(x, t)

    torch.manual_seed(args.eval_seed)
    if target.conditional:
        (y_ref,) = target.sample_cond(args.n_samples)
        x1_ref, _ = target.sample_x1(args.n_samples, y=y_ref)
        truth = x1_ref
    else:
        truth = target.sample_x1(args.n_samples)

    out = {}
    for name in list_schedules():
        g_fn = make_g(name, scale=1.0)
        bg_theta = compose_drift(b_theta_fn, g_fn)
        bg_star = compose_drift(b_star_fn, g_fn)
        rec = {}
        if name != 'zero':
            rec['kl'] = path_kl_girsanov(bg_theta, bg_star, g_fn, target,
                                         n_mc=args.n_mc, t_min=t_eps, t_max=1.0 - t_eps,
                                         device=device, dtype=torch.float32)
        else:
            rec['kl'] = float('nan')
        if target.conditional:
            (y_s,) = target.sample_cond(args.n_samples)
            x1_s, _ = target.sample_x1(args.n_samples, y=y_s)
            samples = em_sample(bg_theta, g_fn, args.n_samples, target.dim,
                                n_steps=args.n_em, t_min=t_eps, t_max=1.0 - t_eps,
                                cond=(y_s,), device=device)
            ref = x1_s
        else:
            samples = em_sample(bg_theta, g_fn, args.n_samples, target.dim,
                                n_steps=args.n_em, t_min=t_eps, t_max=1.0 - t_eps,
                                device=device)
            ref = truth

        if target.dim == 1:
            rec['w1'] = wasserstein_1d(samples, ref, p=1)
            rec['w2'] = wasserstein_1d(samples, ref, p=2)
            rec['mmd2'] = mmd_rbf(samples, ref)
            rec.update({f'mom_{k}': v for k, v in moment_errors(samples, ref).items()})
            # KL via KDE vs analytic density, if available
            if hasattr(target, 'density'):
                rec['kl_hist'] = kl_analytic_1d(samples, target.density)
            else:
                rec['kl_hist'] = float('nan')
        else:
            for k in ['w1', 'w2', 'mmd2', 'kl_hist']:
                rec[k] = float('nan')
        out[name] = rec
    return out


def aggregate(results_per_seed):
    schedules = list_schedules()
    agg = {}
    keys = list(results_per_seed[0][schedules[0]].keys())
    for s in schedules:
        rec = {}
        for k in keys:
            vals = np.array([r[s].get(k, float('nan')) for r in results_per_seed], dtype=float)
            rec[f'{k}_mean'] = float(np.nanmean(vals))
            rec[f'{k}_std']  = float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0)
        agg[s] = rec
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=str, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--runs_dir', type=str, default='./runs')
    p.add_argument('--out_dir', type=str, default='./figs')
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
        print(f"  {s:10s}  KL={r['kl_mean']:.4e}±{r['kl_std']:.2e}  "
              f"W2={r['w2_mean']:.4e}±{r['w2_std']:.2e}")

    with open(os.path.join(args.out_dir, f"zps_{args.target}_results.json"), 'w') as f:
        json.dump({'per_seed': results, 'agg': agg, 'seeds': args.seeds}, f, indent=2)


if __name__ == '__main__':
    main()
