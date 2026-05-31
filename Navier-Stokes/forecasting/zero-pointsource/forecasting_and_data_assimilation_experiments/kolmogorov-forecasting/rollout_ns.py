"""Autoregressive rollout: use the short-lag hat_b to forecast N_ar steps.

At each rollout step:
  1. Coarsen the current sample -> conditioning input
  2. Sample new state x_{t+tau} ~ model(. | coarse(x_t))
  3. Advance
Compare RMSE and enstrophy-W2 vs ground-truth rolled forward via the simulator.
"""
import argparse
import json
import os

import torch

from data import load_snapshots, split_train_val_test, coarsen, upsample_bilinear
from interpolant_ns import ZPSInterpolant
from network_ns import DriftNet
from schedules_ns import list_schedules, make_g
from drift_compose_ns import compose_drift
from sampler_ns import em_sample
from metrics_ns import (rmse_ensemble_mean, enstrophy_w2, spectrum_rmse,
                        ensemble_spread, spread_skill_ratio)


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']
    net = DriftNet(unet_channels=args['unet_channels'],
                   unet_dim_mults=tuple(args['unet_dim_mults'])).to(device)
    net.load_state_dict(ck['state'])
    net.eval()
    return net, args, ck['norm']


@torch.no_grad()
def rollout(ckpt_path, n_steps, schedules, n_test, n_ens, n_em, device, coarsen_factor):
    net, ck_args, norm = load_ckpt(ckpt_path, device)
    omega_all, _ = load_snapshots(ck_args['data'])
    N_traj = omega_all.shape[0]
    _, _, test_idx = split_train_val_test(
        N_traj, ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])
    # Pick n_test starting points at time index 10 (allow n_steps*lag rollout)
    lag = ck_args['lag']
    H = omega_all.shape[-1]
    T = omega_all.shape[1]
    if 10 + n_steps * lag >= T:
        max_start = T - n_steps * lag - 1
        t_start = max(0, max_start)
        print(f"[rollout] trimming start to {t_start} to fit {n_steps} steps of lag {lag}")
    else:
        t_start = 10
    # (test_idx[:n_test], t_start .. t_start + n_steps*lag)
    tj = test_idx[:n_test]
    traj = omega_all[tj, t_start:t_start + (n_steps + 1) * lag:lag].clone()   # (n_test, n_steps+1, H, H)
    x0 = traj[:, 0:1]                                                    # initial full state (on the starting timestep)

    def b_fn(x, t, x0): return net(x, t, x0)

    out = {}
    for sched in schedules:
        g_fn = make_g(sched)
        bg = compose_drift(b_fn, g_fn)
        # per-step ensemble forecast
        # Initialise the rollout state = real x0 (no noise yet)
        cur = x0.to(device).repeat(n_ens, 1, 1, 1, 1) / norm  # (n_ens, n_test, 1, H, H)
        step_metrics = []
        for step in range(n_steps):
            # Coarsen each ensemble member's current full state -> conditioning
            coarse_up = upsample_bilinear(coarsen(cur, coarsen_factor), (H, H))
            # Flatten (n_ens, n_test) -> batch for network
            B = n_ens * n_test
            x_cond = coarse_up.reshape(B, 1, H, H)
            x_next = em_sample(bg, g_fn, (B, 1, H, H),
                               n_steps=n_em, t_min=ck_args.get('t_eps', 1e-3),
                               t_max=1.0 - ck_args.get('t_eps', 1e-3),
                               cond=(x_cond,), device=device)
            cur = x_next.reshape(n_ens, n_test, 1, H, H)
            # Metric vs truth at this step
            truth_step = traj[:, step + 1].to(device).unsqueeze(1) / norm  # (n_test, 1, H, H)
            rec = {
                'rmse': rmse_ensemble_mean(cur, truth_step),
                'spread': ensemble_spread(cur),
                'ssr': spread_skill_ratio(cur, truth_step),
                'ens_w2': enstrophy_w2(cur.squeeze(2), truth_step.squeeze(1)),
                'spec_rmse': spectrum_rmse(cur.squeeze(2), truth_step.squeeze(1)),
            }
            step_metrics.append(rec)
            print(f"  [{sched:10s} step {step + 1:3d}/{n_steps}] "
                  f"RMSE={rec['rmse']:.3e}  ens_W2={rec['ens_w2']:.3e}  "
                  f"SSR={rec['ssr']:.3e}", flush=True)
        out[sched] = step_metrics
    return out, ck_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--n_steps', type=int, default=40)
    p.add_argument('--n_test', type=int, default=8)
    p.add_argument('--n_ens', type=int, default=8)
    p.add_argument('--n_em', type=int, default=100)
    p.add_argument('--coarsen', type=int, default=4)
    p.add_argument('--schedules', type=str, nargs='+',
                   default=['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero'])
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out, ck_args = rollout(args.ckpt, args.n_steps, args.schedules,
                           args.n_test, args.n_ens, args.n_em, device, args.coarsen)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'rollout': out, 'ck_args': ck_args, 'args': vars(args)}, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == '__main__':
    main()
