"""Autoregressive filter on Kolmogorov flow: K cycles of (roll R steps, then SMC-reweight on y_k).

Setup (using lag=1 network):
  - At cycle k (k=0..K-1), particles hold 128² vorticity fields x_{k*R}.
  - For r=1..R: propagate one lag=1 step (one EM forecast) conditional on coarsened state.
  - At r=R: observe y_{k+1} = A x_{(k+1)*R} + eta, eta ~ N(0, sigma_y^2).
  - Reweight by p(y_{k+1} | x_{(k+1)*R}).
  - Resample if ESS < N/2.
  - Continue to cycle k+1.

At every internal step (each lag-1 EM sample), the ensemble evolves; at every R'th
step we get a new observation constraint. Comparable schedules produce different
drift/diffusion profiles during the R-step rollouts.
"""
import argparse
import json
import math
import os
import time

import torch

from data import load_snapshots, split_train_val_test, coarsen, upsample_bilinear
from interpolant_ns import INTERPOLANTS
from network_ns import DriftNet
from schedules_ns import list_schedules, make_g
from drift_compose_ns import compose_drift
from sampler_ns import em_sample
from observation import make_avgpool_operator, log_likelihood as obs_log_lik
from twisted_smc import ess, systematic_resample


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']
    norm = ck['norm']
    ip = INTERPOLANTS[args.get('interpolant', 'linlin')]()
    net = DriftNet(unet_channels=args['unet_channels'],
                   unet_dim_mults=tuple(args['unet_dim_mults'])).to(device)
    net.load_state_dict(ck['state']); net.eval()
    return net, args, norm, ip


@torch.no_grad()
def run_ar_filter(ckpt, args_eval, device):
    net, ck_args, norm, ip = load_ckpt(ckpt, device)
    omega_all, _ = load_snapshots(ck_args['data'])
    _, _, test_idx = split_train_val_test(
        omega_all.shape[0], ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])

    # Pick n_ic test trajectories; starting index ts0 = 10 (after warm-up margin).
    tj_list = test_idx[:args_eval.n_ic]
    H = omega_all.shape[-1]
    R = args_eval.rollout_len
    K = args_eval.n_cycles
    lag = ck_args['lag']
    assert lag == 1, "Use the lag=1 network for AR rollout."
    ts0 = 10
    assert ts0 + K * R < omega_all.shape[1], "Not enough snapshots for K cycles."

    N = args_eval.n_particles
    obs_A = make_avgpool_operator(args_eval.obs_factor)
    sigma_y = args_eval.sigma_y
    t_eps = args_eval.t_eps

    def b_theta_fn(x, s, *cond): return net(x, s, *cond)

    out = {}
    for sched_name in list_schedules():
        print(f"[schedule] {sched_name}", flush=True)
        g_fn = make_g(sched_name, interpolant=ip)
        bg = compose_drift(b_theta_fn, g_fn, ip)

        per_ic_results = []
        for ic_i, tj in enumerate(tj_list):
            # True trajectory at the relevant time steps
            true_traj = omega_all[tj, ts0 : ts0 + (K + 1) * R : R].clone().to(device)   # (K+1, H, H)
            # Normalise
            true_traj_n = true_traj / norm
            # Initial particles: replicate the (known) x0 across N particles
            x = true_traj_n[0].unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1)     # (N, 1, H, H)
            log_w = torch.zeros(N, device=device)

            # Store per-step ensemble statistics
            step_traj = {'x_mean': [], 'x_std': [], 'rmse': [], 'ess': [], 'reweight_step': [],
                         'truth': []}
            step_traj['x_mean'].append((x.mean(dim=0) * norm).squeeze(0).cpu().numpy())
            step_traj['x_std'].append((x.std(dim=0) * norm).squeeze(0).cpu().numpy())
            step_traj['truth'].append(true_traj[0].cpu().numpy())
            step_traj['rmse'].append(0.0)
            step_traj['ess'].append(float(N))

            for cycle in range(K):
                # Propagate R steps of lag-1 forecasting
                for r in range(R):
                    # Conditioning: coarsened current ensemble
                    coarse = coarsen(x, factor=ck_args['coarsen'])
                    cond_up = upsample_bilinear(coarse, (H, H))
                    # One full EM sample: ts ∈ [t_eps, 1-t_eps], n_steps=args_eval.n_em
                    x_next = em_sample(bg, g_fn, (N, 1, H, H),
                                       n_steps=args_eval.n_em,
                                       t_min=t_eps, t_max=1.0 - t_eps,
                                       cond=(cond_up,), device=device)
                    x = x_next
                    step_traj['x_mean'].append((x.mean(dim=0) * norm).squeeze(0).cpu().numpy())
                    step_traj['x_std'].append((x.std(dim=0) * norm).squeeze(0).cpu().numpy())
                    is_obs_step = (r == R - 1)
                    if is_obs_step:
                        step_traj['truth'].append(true_traj[cycle + 1].cpu().numpy())
                        rmse_n = ((x.mean(dim=0, keepdim=True) - true_traj_n[cycle + 1].unsqueeze(0).unsqueeze(0))
                                  .pow(2).mean().sqrt().item() * norm)
                        step_traj['rmse'].append(rmse_n)
                    else:
                        step_traj['truth'].append(None)
                        step_traj['rmse'].append(None)
                    step_traj['ess'].append(float(ess(log_w)))

                # End-of-cycle: reweight on observation
                rng = torch.Generator(device=device).manual_seed(123 + ic_i * K + cycle)
                y_true_n = true_traj_n[cycle + 1].unsqueeze(0).unsqueeze(0)           # (1, 1, H, H)
                y_obs_n = obs_A(y_true_n) + sigma_y * torch.randn(
                    obs_A(y_true_n).shape, device=device, generator=rng) / norm
                # Compute log-likelihood per particle (all at terminal step, so no Tweedie needed)
                y_particles = obs_A(x)                                                # (N, 1, h, w)
                diff = y_particles - y_obs_n
                log_lik = -0.5 * diff.pow(2).sum(dim=(-1, -2, -3)) / ((sigma_y / norm) ** 2)
                log_w = log_w + log_lik
                step_traj['reweight_step'].append(len(step_traj['x_mean']) - 1)

                # Resample if ESS < N/2
                cur_ess = ess(log_w)
                if cur_ess < args_eval.resample_thresh * N:
                    idx = systematic_resample(log_w, generator=rng)
                    x = x[idx]
                    log_w = torch.zeros(N, device=device)

            import numpy as np
            per_ic_results.append({
                'tj': int(tj),
                'rmse_at_obs': [r for r in step_traj['rmse'] if r is not None],
                'ess_curve': step_traj['ess'],
                'reweight_steps': step_traj['reweight_step'],
                # For animation: store mean/std/truth only for the first IC
                **({'_traj': {
                    'x_mean': step_traj['x_mean'],
                    'x_std': step_traj['x_std'],
                    'truth': [t.tolist() if t is not None else None for t in step_traj['truth']],
                }} if ic_i == 0 else {})
            })
        out[sched_name] = per_ic_results
        import numpy as np
        rmses = np.array([[r for r in ic['rmse_at_obs']] for ic in per_ic_results])  # (n_ic, K)
        print(f"  RMSE per cycle mean: {rmses.mean(axis=0).round(3).tolist()}")
    return out, ck_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='./runs/lag1_seed0.pt')
    p.add_argument('--out', type=str, default='./figs/ar_filter/results.pt')
    p.add_argument('--n_particles', type=int, default=16)
    p.add_argument('--rollout_len', type=int, default=10,
                   help='# of lag=1 steps per cycle')
    p.add_argument('--n_cycles', type=int, default=4)
    p.add_argument('--n_em', type=int, default=100)
    p.add_argument('--n_ic', type=int, default=4)
    p.add_argument('--t_eps', type=float, default=0.01)
    p.add_argument('--obs_factor', type=int, default=8)
    p.add_argument('--sigma_y', type=float, default=0.3)
    p.add_argument('--resample_thresh', type=float, default=0.5)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[AR filter] ckpt={args.ckpt}  R={args.rollout_len} x K={args.n_cycles} cycles")
    t0 = time.time()
    results, ck_args = run_ar_filter(args.ckpt, args, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'results': results, 'ck_args': ck_args, 'args': vars(args)}, args.out)
    print(f'[saved] {args.out}  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
