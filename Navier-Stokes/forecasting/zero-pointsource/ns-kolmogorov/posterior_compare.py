"""Sweep schedules for twisted-SMC posterior sampling.

Usage:
    python posterior_compare.py --ckpt runs/lag40_seed0.pt --out figs/posterior/lag40.json
"""
import argparse
import json
import os
import time

import torch

from data import load_snapshots, split_train_val_test, PairDataset
from interpolant_ns import INTERPOLANTS
from network_ns import DriftNet
from schedules_ns import list_schedules, make_g
from drift_compose_ns import compose_drift
from observation import make_avgpool_operator, log_likelihood as obs_log_likelihood
from twisted_smc import twisted_smc, ess


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']; norm = ck['norm']
    ip = INTERPOLANTS[args.get('interpolant', 'linlin')]()
    net = DriftNet(unet_channels=args['unet_channels'],
                   unet_dim_mults=tuple(args['unet_dim_mults'])).to(device)
    net.load_state_dict(ck['state']); net.eval()
    return net, args, norm, ip


def run_one(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    net, ck_args, norm, ip = load_ckpt(args.ckpt, device)
    omega_all, _ = load_snapshots(ck_args['data'])
    _, _, test_idx = split_train_val_test(
        omega_all.shape[0], ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])
    test_ds = PairDataset(omega_all[test_idx], ck_args['lag'], coarsen_factor=ck_args['coarsen'])
    H = omega_all.shape[-1]

    obs_A = make_avgpool_operator(args.obs_factor)
    sigma_y = args.sigma_y

    # Pick n_ic test inputs
    idxs = torch.linspace(0, len(test_ds) - 1, args.n_ic).long()

    def b_theta_fn(x, s, *cond):
        # Reshape s for network if needed
        ss = s.reshape(-1, 1) if s.dim() == 1 else s[..., :1] if s.dim() > 2 else s
        return net(x, ss, *cond)

    out = {}
    for sched_name in list_schedules():
        print(f"[schedule] {sched_name}")
        sched_metrics = {'ess_curve': [], 'posterior_rmse': [], 'posterior_spread': [],
                          'log_marginal': [], 'per_ic': []}
        g_fn = make_g(sched_name, interpolant=ip)
        def compose_b_g_fn(x, s, *cond):
            bg = compose_drift(b_theta_fn, g_fn, ip)
            return bg(x, s, *cond)

        for ic_idx in idxs:
            item = test_ds[int(ic_idx)]
            x0_up = item['x0_up'].unsqueeze(0).to(device) / norm       # (1, 1, H, H)
            x1_truth = item['x1'].unsqueeze(0).to(device) / norm       # (1, 1, H, H)
            y_obs_clean = obs_A(x1_truth)
            # Add obs noise (using deterministic RNG per IC)
            rng = torch.Generator(device=device).manual_seed(12345 + int(ic_idx))
            y_obs = y_obs_clean + sigma_y * torch.randn(y_obs_clean.shape, device=device, generator=rng)
            # Run SMC
            t0 = time.time()
            smc_rng = torch.Generator(device=device).manual_seed(7777 + int(ic_idx))
            particles, log_w, logZ, trace = twisted_smc(
                b_theta_fn=b_theta_fn,
                compose_b_g_fn=compose_b_g_fn,
                g_fn=g_fn,
                obs_A=obs_A,
                y_obs=y_obs,
                sigma_y=sigma_y,
                n_particles=args.n_particles,
                dim_tuple=(1, H, H),
                cond=(x0_up,),
                ip=ip,
                n_steps=args.n_em,
                t_min=args.t_eps,
                t_max=1.0 - args.t_eps,
                resample_thresh=args.resample_thresh,
                device=device,
                generator=smc_rng,
                return_trace=True,
                proposal=args.proposal,
                guidance_type=args.guidance_type,
                guidance_eta=args.guidance_eta,
            )
            elapsed = time.time() - t0
            # Posterior mean (weighted)
            lw = log_w - log_w.max()
            w = lw.exp(); w = w / w.sum()
            mean = (w.view(-1, 1, 1, 1) * particles).sum(dim=0, keepdim=True)   # (1, 1, H, H)
            sd = torch.sqrt((w.view(-1, 1, 1, 1) * (particles - mean) ** 2).sum(dim=0, keepdim=True))
            # Metrics (in normalised units)
            rmse = ((mean - x1_truth) ** 2).mean().sqrt().item()
            spread = sd.mean().item()
            sched_metrics['posterior_rmse'].append(rmse)
            sched_metrics['posterior_spread'].append(spread)
            sched_metrics['log_marginal'].append(logZ)
            sched_metrics['ess_curve'].append(trace['ess'])
            sched_metrics['per_ic'].append({
                'ic': int(ic_idx), 'rmse': rmse, 'spread': spread, 'logZ': logZ,
                'final_ess': trace['ess'][-1], 'time': elapsed,
            })
            print(f"  ic={int(ic_idx):4d}  rmse={rmse:.4f}  spread={spread:.4f}  "
                  f"logZ={logZ:.2f}  finalESS={trace['ess'][-1]:.1f}  ({elapsed:.1f}s)",
                  flush=True)
        # Aggregate
        rmses = sched_metrics['posterior_rmse']
        spreads = sched_metrics['posterior_spread']
        logZs = sched_metrics['log_marginal']
        import numpy as np
        sched_metrics['agg'] = {
            'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses, ddof=1)),
            'spread_mean': float(np.mean(spreads)),
            'logZ_mean': float(np.mean(logZs)), 'logZ_std': float(np.std(logZs, ddof=1)),
            'final_ess_mean': float(np.mean([m['final_ess'] for m in sched_metrics['per_ic']])),
        }
        out[sched_name] = sched_metrics
    return out, ck_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--n_particles', type=int, default=32)
    p.add_argument('--n_em', type=int, default=100)
    p.add_argument('--n_ic', type=int, default=8)
    p.add_argument('--t_eps', type=float, default=0.01)
    p.add_argument('--obs_factor', type=int, default=8)
    p.add_argument('--sigma_y', type=float, default=0.3)
    p.add_argument('--resample_thresh', type=float, default=0.5)
    p.add_argument('--proposal', type=str, default='guided', choices=['guided', 'uncond'])
    p.add_argument('--guidance_type', type=str, default='doob', choices=['doob', 'natural'])
    p.add_argument('--guidance_eta', type=float, default=1.0)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    print(f"[posterior] {args.ckpt}  obs=avgpool{args.obs_factor}  sigma_y={args.sigma_y}  "
          f"N={args.n_particles}  n_ic={args.n_ic}")
    out, ck_args = run_one(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Save
    with open(args.out, 'w') as f:
        json.dump({'results': out, 'ck_args': ck_args, 'args': vars(args)}, f, indent=2)
    print(f"[saved] {args.out}")
    # Print headline
    print("\n=== Summary ===")
    for name, rec in out.items():
        agg = rec['agg']
        print(f"  {name:10s}  RMSE={agg['rmse_mean']:.3e}±{agg['rmse_std']:.2e}   "
              f"spread={agg['spread_mean']:.3e}   logZ={agg['logZ_mean']:.2f}±{agg['logZ_std']:.2f}   "
              f"finalESS={agg['final_ess_mean']:.1f}/{args.n_particles}")


if __name__ == '__main__':
    main()
