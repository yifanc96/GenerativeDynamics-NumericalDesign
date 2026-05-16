"""Autoregressive rollout → save per-step frames for a later animation.

Layout per frame (2x2, no captions — user will overlay labels when animating):
  truth  |  ens mean
  |err|  |  ens std

Uses Follmer schedule on the lag=1 (or lag=10) checkpoint. One test IC,
small ensemble (n_ens members) advanced autoregressively for n_steps.
Each step: coarsen current ensemble-mean (or each particle) -> conditioning,
sample new state per particle via EM-SDE, advance, dump PNG.

Careful about memory: close every matplotlib figure immediately.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from data import load_snapshots, split_train_val_test, coarsen, upsample_bilinear
from drift_compose_ns import compose_drift
from enkf import enkf_update
from interpolant_ns import INTERPOLANTS
from network_ns import DriftNet
from observation import make_avgpool_operator
from sampler_ns import em_sample
from schedules_ns import make_g
from twisted_smc import twisted_smc


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']
    net = DriftNet(unet_channels=args['unet_channels'],
                   unet_dim_mults=tuple(args['unet_dim_mults'])).to(device)
    net.load_state_dict(ck['state'])
    net.eval()
    ip = INTERPOLANTS[args.get('interpolant', 'linlin')]()
    return net, args, ck['norm'], ip


def save_frame(path, truth, mean, err, sd, vlim, vlim_err, vlim_sd):
    """2x2 panel: truth, mean, |err|, std. No captions. Close fig on exit."""
    sns.set_theme(context='paper', style='white', font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    im0 = axes[0, 0].imshow(truth, cmap=sns.cm.icefire, vmin=-vlim, vmax=vlim, origin='lower')
    im1 = axes[0, 1].imshow(mean,  cmap=sns.cm.icefire, vmin=-vlim, vmax=vlim, origin='lower')
    im2 = axes[1, 0].imshow(err,   cmap='magma',        vmin=0,     vmax=vlim_err, origin='lower')
    im3 = axes[1, 1].imshow(sd,    cmap='magma',        vmin=0,     vmax=vlim_sd,  origin='lower')
    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax, im in zip(axes.flatten(), (im0, im1, im2, im3)):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=22)
        cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    net, ck_args, norm, ip = load_ckpt(args.ckpt, device)
    omega_all, _ = load_snapshots(ck_args['data'])
    H = omega_all.shape[-1]
    T = omega_all.shape[1]
    lag = ck_args['lag']
    coarsen_f = ck_args['coarsen']
    if ck_args.get('interpolant', 'linlin') == 'sqlin':
        t_eps = 0.05
    else:
        t_eps = ck_args.get('t_eps', 1e-3)

    # Split; pick test trajectory
    _, _, test_idx = split_train_val_test(
        omega_all.shape[0], ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])
    tj = int(test_idx[args.traj_idx].item())
    max_start = T - args.n_steps * lag - 1
    t_start = max(0, min(args.t_start, max_start))
    # traj[k] is the truth at step k: shape (H, H)
    truth_traj = omega_all[tj, t_start:t_start + (args.n_steps + 1) * lag:lag].clone().to(device)
    assert truth_traj.shape[0] == args.n_steps + 1, \
        f'need {args.n_steps + 1} truth frames, got {truth_traj.shape[0]}'

    # Model schedule
    g_fn = make_g(args.schedule, interpolant=ip)

    def b_fn(x, t, xcond):
        return net(x, t, xcond)

    bg = compose_drift(b_fn, g_fn, ip)

    # Ensemble state: (n_ens, 1, H, H), initialise at truth step 0
    n_ens = args.n_ens
    cur = (truth_traj[0].unsqueeze(0).unsqueeze(0) / norm).repeat(n_ens, 1, 1, 1)  # (n_ens, 1, H, H)

    # Global color scales (from truth)
    truth_cpu = (truth_traj * 1.0).cpu().numpy()    # already un-normed
    vlim = float(np.percentile(np.abs(truth_cpu), 99.5))

    # Error and std scale — fixed so frames are comparable. Use a rough estimate.
    vlim_err = args.err_scale if args.err_scale > 0 else vlim * 0.6
    vlim_sd  = args.sd_scale  if args.sd_scale  > 0 else vlim * 0.5

    os.makedirs(args.out_dir, exist_ok=True)
    # Frame 0: truth vs. deterministic start (zero err/std).
    step0_truth = truth_cpu[0]
    step0_mean = step0_truth.copy()
    save_frame(os.path.join(args.out_dir, 'frame_000.png'),
               step0_truth, step0_mean, np.zeros_like(step0_truth), np.zeros_like(step0_truth),
               vlim, vlim_err, vlim_sd)

    print(f'[ar-anim] traj={tj}, lag={lag}, schedule={args.schedule}, n_steps={args.n_steps}, n_ens={n_ens}')
    print(f'[ar-anim] vorticity vlim = ±{vlim:.2f};  err scale = {vlim_err:.2f};  std scale = {vlim_sd:.2f}')

    # Collect arrays so they can be re-rendered later without resampling.
    arrs = {
        'truth': [step0_truth],
        'mean':  [step0_mean],
        'err':   [np.zeros_like(step0_truth)],
        'sd':    [np.zeros_like(step0_truth)],
        'rmse':  [0.0],
    }

    obs_A = make_avgpool_operator(args.obs_factor) if args.assimilate else None

    da_method = args.da_method if args.assimilate else 'none'

    for step in range(args.n_steps):
        # Per-particle conditioning: each particle uses its own coarsened state.
        coarse_up = upsample_bilinear(coarsen(cur, coarsen_f), (H, H))   # (n_ens, 1, H, H)
        if args.assimilate and da_method in ('enkf', 'enkf_loc'):
            # 1) Pure-AR forecast step (no observation), Follmer prior sampler.
            x_fcst = em_sample(bg, g_fn, cur.shape,
                               n_steps=args.n_em, t_min=t_eps, t_max=1.0 - t_eps,
                               cond=(coarse_up,), device=device)
            # 2) Build observation
            truth_next = truth_traj[step + 1].unsqueeze(0).unsqueeze(0) / norm
            y_obs_clean = obs_A(truth_next)
            obs_rng = torch.Generator(device=device).manual_seed(54321 + step)
            y_obs = y_obs_clean + args.sigma_y * torch.randn_like(y_obs_clean, generator=obs_rng)
            # 3) EnKF update on the ensemble
            enkf_rng = torch.Generator(device=device).manual_seed(8888 + step)
            cur = enkf_update(x_fcst, y_obs, obs_factor=args.obs_factor,
                              sigma_y=args.sigma_y, inflation=args.inflation,
                              localise=(da_method == 'enkf_loc'),
                              loc_radius=args.loc_radius, generator=enkf_rng)
        elif args.assimilate:
            # Observation of truth at next step
            truth_next = truth_traj[step + 1].unsqueeze(0).unsqueeze(0) / norm  # (1, 1, H, H)
            y_obs_clean = obs_A(truth_next)
            obs_rng = torch.Generator(device=device).manual_seed(54321 + step)
            y_obs = y_obs_clean + args.sigma_y * torch.randn_like(y_obs_clean, generator=obs_rng)

            def b_theta_fn(x, s, *cond):
                ss = s.reshape(-1, 1) if s.dim() == 1 else s[..., :1]
                return net(x, ss, *cond)
            def compose_b_g_fn(x, s, *cond):
                return bg(x, s, *cond)

            smc_rng = torch.Generator(device=device).manual_seed(7777 + step)
            particles, log_w, _, _ = twisted_smc(
                b_theta_fn=b_theta_fn, compose_b_g_fn=compose_b_g_fn, g_fn=g_fn,
                obs_A=obs_A, y_obs=y_obs, sigma_y=args.sigma_y,
                n_particles=n_ens, dim_tuple=(1, H, H),
                cond=(coarse_up,), ip=ip,
                n_steps=args.n_em, t_min=t_eps, t_max=1.0 - t_eps,
                resample_thresh=0.5, device=device, generator=smc_rng,
                return_trace=False, proposal='guided',
                guidance_type='doob', guidance_eta=args.guidance_eta,
            )
            # Weighted resample to n_ens equally-weighted particles for the next step
            lw = log_w - log_w.max()
            w = lw.exp(); w = w / w.sum()
            idx = torch.multinomial(w, n_ens, replacement=True, generator=smc_rng)
            cur = particles[idx]
        else:
            x_next = em_sample(bg, g_fn, cur.shape,
                               n_steps=args.n_em, t_min=t_eps, t_max=1.0 - t_eps,
                               cond=(coarse_up,), device=device)
            cur = x_next
        # De-normalise for metrics/plot
        mean_cpu = (cur.mean(dim=0)[0] * norm).cpu().numpy()
        sd_cpu = (cur.std(dim=0)[0] * norm).cpu().numpy()
        truth_cpu_step = truth_cpu[step + 1]
        err_cpu = np.abs(mean_cpu - truth_cpu_step)

        rmse = float(np.sqrt(np.mean(err_cpu ** 2)))
        print(f'  step {step + 1:3d}/{args.n_steps}  rmse={rmse:.3f}  '
              f'mean_spread={float(sd_cpu.mean()):.3f}', flush=True)
        save_frame(os.path.join(args.out_dir, f'frame_{step + 1:03d}.png'),
                   truth_cpu_step, mean_cpu, err_cpu, sd_cpu,
                   vlim, vlim_err, vlim_sd)
        arrs['truth'].append(truth_cpu_step)
        arrs['mean'].append(mean_cpu)
        arrs['err'].append(err_cpu)
        arrs['sd'].append(sd_cpu)
        arrs['rmse'].append(rmse)

    # Cache the full trajectory so we can re-render with different colour scales
    np.savez_compressed(os.path.join(args.out_dir, 'arrays.npz'),
                        truth=np.asarray(arrs['truth']),
                        mean=np.asarray(arrs['mean']),
                        err=np.asarray(arrs['err']),
                        sd=np.asarray(arrs['sd']),
                        rmse=np.asarray(arrs['rmse']),
                        vlim=np.asarray(vlim))
    print(f'[saved] {args.n_steps + 1} frames + arrays.npz in {args.out_dir}/')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='runs/lag1_seed0.pt')
    p.add_argument('--out_dir', type=str, default='figs/ar_anim')
    p.add_argument('--schedule', type=str, default='follmer')
    p.add_argument('--n_steps', type=int, default=80)
    p.add_argument('--n_ens', type=int, default=8)
    p.add_argument('--n_em', type=int, default=50)
    p.add_argument('--traj_idx', type=int, default=0, help='index into test_idx')
    p.add_argument('--t_start', type=int, default=10, help='snapshot time index to start from')
    p.add_argument('--err_scale', type=float, default=0.0, help='cap for |err|; 0 = auto')
    p.add_argument('--sd_scale',  type=float, default=0.0, help='cap for std;    0 = auto')
    p.add_argument('--assimilate', action='store_true',
                   help='enable observation-based update each step')
    p.add_argument('--da_method', type=str, default='smc',
                   choices=['smc', 'enkf', 'enkf_loc'],
                   help='smc = twisted-SMC + Tweedie + Follmer guidance; enkf = vanilla stochastic EnKF; '
                        'enkf_loc = stochastic EnKF + Gaspari-Cohn localisation')
    p.add_argument('--obs_factor', type=int, default=8)
    p.add_argument('--sigma_y', type=float, default=0.3)
    p.add_argument('--guidance_eta', type=float, default=1.0)
    p.add_argument('--inflation', type=float, default=1.05,
                   help='multiplicative inflation factor (EnKF only)')
    p.add_argument('--loc_radius', type=float, default=8.0,
                   help='localisation radius in state-grid pixels (enkf_loc only)')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
