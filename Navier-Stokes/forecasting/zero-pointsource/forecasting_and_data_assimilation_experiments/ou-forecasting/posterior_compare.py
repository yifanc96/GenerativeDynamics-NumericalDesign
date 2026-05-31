"""Sweep schedules for 1D posterior-sampling via twisted SMC.

Targets: gaussian1d (analytic Gaussian posterior), bimodal1d (GMM posterior).
For each, compare ESS, posterior W2-to-analytic, log-Z variance across schedules
in two proposal modes: 'uncond' and 'guided'.

Usage:
  python posterior_compare.py --target gaussian1d --seeds 0 1 2 3 4 --sigma_y 0.3
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from interpolant_zps import ZPSInterpolant
from networks import MLPNet
from targets_zps import make_target
from schedules_zps import list_schedules, make_g
from drift_compose import compose_drift
from metrics import wasserstein_1d
from posterior_smc import twisted_smc_1d, ess


# ---------- analytic posteriors ----------

def analytic_gaussian_posterior(mu, sigma, y, sigma_y):
    """x1 ~ N(mu, sigma^2), y = x1 + eta, eta ~ N(0, sigma_y^2). Posterior N(mu_post, sigma_post^2)."""
    inv = 1.0 / sigma ** 2 + 1.0 / sigma_y ** 2
    mu_post = (mu / sigma ** 2 + y / sigma_y ** 2) / inv
    sigma_post = math.sqrt(1.0 / inv)
    return mu_post, sigma_post


def analytic_bimodal_posterior_samples(m, tau, y, sigma_y, n):
    """p(x_1) = 1/2 N(-m, tau^2) + 1/2 N(m, tau^2). y = x_1 + eta, eta ~ N(0, sigma_y^2).
    Posterior: 1/2 w_+ N(mu_+, s^2) + 1/2 w_- N(mu_-, s^2), weights by Bayes."""
    s2_post = 1.0 / (1.0 / tau ** 2 + 1.0 / sigma_y ** 2)
    s_post = math.sqrt(s2_post)
    mu_p = s2_post * (m / tau ** 2 + y / sigma_y ** 2)
    mu_n = s2_post * (-m / tau ** 2 + y / sigma_y ** 2)
    # Mixture weights (marginal p(y|mode))
    v = tau ** 2 + sigma_y ** 2
    log_pos = -0.5 * (y - m) ** 2 / v
    log_neg = -0.5 * (y + m) ** 2 / v
    # w_+ = sigmoid(log_pos - log_neg)
    w_p = 1.0 / (1.0 + math.exp(log_neg - log_pos))
    mask = np.random.rand(n) < w_p
    eps = np.random.randn(n)
    samples = np.where(mask, mu_p + s_post * eps, mu_n + s_post * eps)
    return torch.as_tensor(samples, dtype=torch.float32).reshape(-1, 1), (mu_p, mu_n, s_post, w_p)


def analytic_ou_posterior(decay, cond_sig2, y_cond, y_obs, sigma_y):
    """OU forecasting conditional: X_1 | Y_s ~ N((decay-1) Y_s, cond_sig2).
    Observation y_obs = X_1 + eta, eta ~ N(0, sigma_y^2). Posterior Gaussian."""
    prior_mu = (decay - 1.0) * y_cond
    inv = 1.0 / cond_sig2 + 1.0 / sigma_y ** 2
    mu_post = (prior_mu / cond_sig2 + y_obs / sigma_y ** 2) / inv
    s_post = math.sqrt(1.0 / inv)
    return mu_post, s_post


# ---------- runner ----------

def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']
    target = make_target(args['target'], device=device)
    cond_dim = getattr(target, 'cond_dim', target.dim) if getattr(target, 'conditional', False) else 0
    net = MLPNet(x_dim=target.dim, cond_dim=cond_dim,
                 hidden=args['hidden'], n_layers=args['n_layers'],
                 time_embed=args['time_embed']).to(device)
    # Load b-net weights if saved under that key; otherwise combine v/s heads.
    # For zps 1D study we trained v_net + s_net; here we use the b_net that combines them
    # using interpolant's Rb target. Actually zps 1D trained v_theta, s_theta separately.
    # For compatibility with the existing gaussbase training (which did train b_theta directly via Rb),
    # we check which key is present.
    if 'b_state' in ck:
        net.load_state_dict(ck['b_state'])
    elif 'v_state' in ck and 's_state' in ck:
        # Use the v_net only as a baseline-drift proxy? For Gaussian base this works.
        net.load_state_dict(ck['v_state'])
    else:
        raise KeyError(f"Checkpoint {path} has neither 'b_state' nor 'v_state'.")
    net.eval()
    return net, args


def run_for_target(target_name, seeds, runs_dir, sigma_y, n_particles, n_em, n_ic,
                   t_eps, resample_thresh, device, proposal,
                   guidance_type='doob', guidance_eta=1.0):
    """Returns dict[schedule -> {metric -> [per-seed list]}]."""
    ip = ZPSInterpolant()
    agg = {name: {'final_ess': [], 'rmse': [], 'w2_to_analytic': [],
                  'logZ': [], 'ess_curve_mean': []} for name in list_schedules()}
    # Fix observation y and analytic posterior across seeds to compare apples-to-apples.
    # Use a random but fixed set of "true x1" samples → derive y.
    torch.manual_seed(0)
    y_cond_batch = None   # conditioning tensor for conditional targets
    if target_name == 'gaussian1d':
        tg = make_target('gaussian1d', device=device)
        truth_x1 = tg.sample_x1(n_ic)                                 # (n_ic, 1)
    elif target_name == 'bimodal1d':
        tg = make_target('bimodal1d', device=device)
        truth_x1 = tg.sample_x1(n_ic)
    elif target_name == 'ou_forecast':
        tg = make_target('ou_forecast', device=device)
        (y_cond_batch,) = tg.sample_cond(n_ic)                        # (n_ic, 1) stationary Y_s
        truth_x1, _ = tg.sample_x1(n_ic, y=y_cond_batch)              # (n_ic, 1) X_1 | Y_s
    else:
        raise NotImplementedError(f"{target_name} not yet implemented")
    truth_x1 = truth_x1.to(device)
    # Observation: y = truth_x1 + eta
    y_obs_batch = (truth_x1 + sigma_y * torch.randn_like(truth_x1)).to(device)

    # Analytic posteriors
    analytic_samples = []
    for i in range(n_ic):
        y_i = float(y_obs_batch[i, 0].item())
        if target_name == 'gaussian1d':
            mu_p, s_p = analytic_gaussian_posterior(1.5, 0.5, y_i, sigma_y)
            samples = torch.as_tensor(np.random.randn(20000) * s_p + mu_p, dtype=torch.float32).reshape(-1, 1)
        elif target_name == 'bimodal1d':
            samples, _ = analytic_bimodal_posterior_samples(1.0, 0.3, y_i, sigma_y, 20000)
        elif target_name == 'ou_forecast':
            y_cond_i = float(y_cond_batch[i, 0].item())
            mu_p, s_p = analytic_ou_posterior(tg.decay, tg.cond_sig2, y_cond_i, y_i, sigma_y)
            samples = torch.as_tensor(np.random.randn(20000) * s_p + mu_p, dtype=torch.float32).reshape(-1, 1)
        analytic_samples.append(samples)

    for seed in seeds:
        ckpt = os.path.join(runs_dir, f"{target_name}_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"[warn] missing {ckpt}, skipping")
            continue
        net, ck_args = load_ckpt(ckpt, device)

        def b_theta_fn(x, s, *cond):
            return net(x, s, *cond)

        for sched_name in list_schedules():
            if sched_name == 'ode' and proposal == 'uncond':
                # ODE + uncond doesn't produce any spread; skip to save time
                pass
            g_fn = make_g(sched_name, scale=1.0)
            def compose_b_g_fn(x, s, *cond):
                bg = compose_drift(b_theta_fn, g_fn)
                return bg(x, s, *cond)
            ess_curves_this_sched = []
            final_ess_this_sched = []
            w2s_this_sched = []
            rmses_this_sched = []
            logZs_this_sched = []
            for ic in range(n_ic):
                y_obs = y_obs_batch[ic:ic+1]    # (1, 1)
                cond_tup = (y_cond_batch[ic:ic+1],) if y_cond_batch is not None else ()
                rng = torch.Generator(device=device).manual_seed(7777 + ic + 100 * seed)
                x_final, log_w, logZ, trace = twisted_smc_1d(
                    b_theta_fn=b_theta_fn, compose_b_g_fn=compose_b_g_fn,
                    g_fn=g_fn, y_obs=y_obs, sigma_y=sigma_y,
                    n_particles=n_particles, dim=1, cond=cond_tup,
                    n_steps=n_em, t_min=t_eps, t_max=1.0 - t_eps,
                    resample_thresh=resample_thresh, proposal=proposal,
                    guidance_type=guidance_type, guidance_eta=guidance_eta,
                    device=device, generator=rng, return_trace=True,
                )
                # Posterior samples: weight-resample
                lw = log_w - log_w.max()
                w = lw.exp(); w = w / w.sum()
                idx = torch.multinomial(w, 20000, replacement=True, generator=rng)
                posterior_samples = x_final[idx]
                # W2 to analytic posterior samples (on CPU to avoid device mismatch)
                ps_cpu = posterior_samples.cpu()
                w2 = wasserstein_1d(ps_cpu, analytic_samples[ic], p=2)
                mean_post = ps_cpu.mean().item()
                rmse = abs(mean_post - analytic_samples[ic].mean().item())
                ess_curves_this_sched.append(trace['ess'])
                final_ess_this_sched.append(trace['ess'][-1])
                w2s_this_sched.append(w2)
                rmses_this_sched.append(rmse)
                logZs_this_sched.append(logZ)
            agg[sched_name]['final_ess'].append(np.mean(final_ess_this_sched))
            agg[sched_name]['rmse'].append(np.mean(rmses_this_sched))
            agg[sched_name]['w2_to_analytic'].append(np.mean(w2s_this_sched))
            agg[sched_name]['logZ'].append(np.mean(logZs_this_sched))
            # Store ESS curve averaged across ICs (one curve per seed)
            ec = np.mean(np.asarray(ess_curves_this_sched), axis=0)
            agg[sched_name]['ess_curve_mean'].append(ec.tolist())

        print(f"  seed {seed} done")

    # Aggregate across seeds
    summary = {}
    for name, rec in agg.items():
        summary[name] = {
            'final_ess_mean': float(np.mean(rec['final_ess'])) if rec['final_ess'] else float('nan'),
            'final_ess_std': float(np.std(rec['final_ess'], ddof=1)) if len(rec['final_ess']) > 1 else 0.0,
            'rmse_mean': float(np.mean(rec['rmse'])) if rec['rmse'] else float('nan'),
            'rmse_std': float(np.std(rec['rmse'], ddof=1)) if len(rec['rmse']) > 1 else 0.0,
            'w2_mean': float(np.mean(rec['w2_to_analytic'])) if rec['w2_to_analytic'] else float('nan'),
            'w2_std': float(np.std(rec['w2_to_analytic'], ddof=1)) if len(rec['w2_to_analytic']) > 1 else 0.0,
            'logZ_mean': float(np.mean(rec['logZ'])) if rec['logZ'] else float('nan'),
            'logZ_std': float(np.std(rec['logZ'], ddof=1)) if len(rec['logZ']) > 1 else 0.0,
            'ess_curves': rec['ess_curve_mean'],
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=str, default='gaussian1d',
                   choices=['gaussian1d', 'bimodal1d', 'ou_forecast'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--runs_dir', type=str, default='./runs')
    p.add_argument('--out_dir', type=str, default='./figs_posterior')
    p.add_argument('--sigma_y', type=float, default=0.3)
    p.add_argument('--n_particles', type=int, default=100)
    p.add_argument('--n_em', type=int, default=200)
    p.add_argument('--n_ic', type=int, default=20)
    p.add_argument('--t_eps', type=float, default=1e-3)
    p.add_argument('--resample_thresh', type=float, default=0.5)
    p.add_argument('--proposal', type=str, default='guided', choices=['guided', 'uncond'])
    p.add_argument('--guidance_type', type=str, default='doob', choices=['doob', 'natural'])
    p.add_argument('--guidance_eta', type=float, default=1.0)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    summary = run_for_target(
        args.target, args.seeds, args.runs_dir, args.sigma_y,
        args.n_particles, args.n_em, args.n_ic, args.t_eps,
        args.resample_thresh, device, args.proposal,
        guidance_type=args.guidance_type, guidance_eta=args.guidance_eta)
    tag = f'{args.target}_{args.proposal}'
    if args.proposal == 'guided':
        tag += f'_{args.guidance_type}_eta{args.guidance_eta}'
    tag += f'_sigma{args.sigma_y}'
    out_path = os.path.join(args.out_dir, f'posterior_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'args': vars(args)}, f, indent=2)
    print(f"\n[saved] {out_path}")
    print("\n=== Summary ===")
    for name, rec in summary.items():
        print(f"  {name:12s}  finalESS={rec['final_ess_mean']:.1f}±{rec['final_ess_std']:.1f}/{args.n_particles}"
              f"   W2={rec['w2_mean']:.3e}±{rec['w2_std']:.2e}"
              f"   RMSE={rec['rmse_mean']:.3e}"
              f"   logZ={rec['logZ_mean']:.2f}±{rec['logZ_std']:.2f}")


if __name__ == '__main__':
    main()
