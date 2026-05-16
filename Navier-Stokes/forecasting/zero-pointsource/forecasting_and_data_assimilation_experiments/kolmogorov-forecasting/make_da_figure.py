"""Paper-ready figure illustrating the data-assimilation setting.

Layout (single row, 6 panels):
  1. conditioning omega_t (coarse 4x, upsampled)  -- 'current state (coarse)'
  2. truth omega_{t+tau}                          -- 'truth'
  3. observation y = AvgPool_8(truth) + noise     -- 'observation (8x coarser, noisy)'
  4. Follmer posterior mean                       -- 'posterior mean (ours)'
  5. Follmer posterior std                        -- 'posterior std'
  6. |posterior mean - truth|                     -- '|error|'

Shared vorticity colourmap for panels 1-4; own scale for std and error.
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

from data import load_snapshots, split_train_val_test, PairDataset
from drift_compose_ns import compose_drift
from interpolant_ns import INTERPOLANTS
from network_ns import DriftNet
from observation import make_avgpool_operator
from posterior_compare import load_ckpt
from schedules_ns import make_g
from twisted_smc import twisted_smc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default='runs/sqlin/lag40_seed0.pt',
                   help='checkpoint; sqlin lag40 is the most dramatic DA setting')
    p.add_argument('--out', type=str,
                   default='figs/da_illustration.pdf')
    p.add_argument('--ic', type=int, default=2,
                   help='which test IC (0..n_test-1)')
    p.add_argument('--n_particles', type=int, default=64)
    p.add_argument('--n_em', type=int, default=100)
    p.add_argument('--t_eps', type=float, default=None,
                   help='auto: 0.05 for sqlin, 0.01 otherwise')
    p.add_argument('--obs_factor', type=int, default=8)
    p.add_argument('--sigma_y', type=float, default=0.3)
    p.add_argument('--guidance_eta', type=float, default=1.0)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    net, ck_args, norm, ip = load_ckpt(args.ckpt, device)
    if args.t_eps is None:
        args.t_eps = 0.05 if ck_args.get('interpolant', 'linlin') == 'sqlin' else 0.01

    # Pick test IC
    omega_all, _ = load_snapshots(ck_args['data'])
    _, _, test_idx = split_train_val_test(
        omega_all.shape[0], ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])
    test_ds = PairDataset(omega_all[test_idx], ck_args['lag'],
                          coarsen_factor=ck_args['coarsen'])
    item = test_ds[args.ic]
    x0_full = item['x0'].unsqueeze(0).to(device) / norm        # (1, 1, H, H) -- current state
    x0_up = item['x0_up'].unsqueeze(0).to(device) / norm       # (1, 1, H, H) -- coarse conditioning
    x1_truth = item['x1'].unsqueeze(0).to(device) / norm       # (1, 1, H, H) -- truth
    H = x1_truth.shape[-1]

    # Observation
    obs_A = make_avgpool_operator(args.obs_factor)
    y_obs_clean = obs_A(x1_truth)
    torch.manual_seed(12345 + args.ic)
    y_obs = y_obs_clean + args.sigma_y * torch.randn_like(y_obs_clean)

    # Run Follmer posterior
    g_fn = make_g('follmer', interpolant=ip)

    def b_theta_fn(x, s, *cond):
        ss = s.reshape(-1, 1) if s.dim() == 1 else (s[..., :1] if s.dim() > 2 else s)
        return net(x, ss, *cond)

    def compose_b_g_fn(x, s, *cond):
        bg = compose_drift(b_theta_fn, g_fn, ip)
        return bg(x, s, *cond)

    rng = torch.Generator(device=device).manual_seed(7777 + args.ic)
    particles, log_w, logZ, _ = twisted_smc(
        b_theta_fn=b_theta_fn,
        compose_b_g_fn=compose_b_g_fn,
        g_fn=g_fn,
        obs_A=obs_A,
        y_obs=y_obs,
        sigma_y=args.sigma_y,
        n_particles=args.n_particles,
        dim_tuple=(1, H, H),
        cond=(x0_up,),
        ip=ip,
        n_steps=args.n_em,
        t_min=args.t_eps,
        t_max=1.0 - args.t_eps,
        resample_thresh=0.5,
        device=device,
        generator=rng,
        return_trace=False,
        proposal='guided',
        guidance_type='doob',
        guidance_eta=args.guidance_eta,
    )
    # Weighted posterior mean and std (de-normalised by `norm`)
    lw = log_w - log_w.max()
    w = lw.exp(); w = w / w.sum()
    w4 = w.view(-1, 1, 1, 1)
    mean = (w4 * particles).sum(dim=0, keepdim=True)
    sd = torch.sqrt((w4 * (particles - mean) ** 2).sum(dim=0, keepdim=True))
    rmse = ((mean - x1_truth) ** 2).mean().sqrt().item()
    print(f"[da-fig] ckpt={args.ckpt} ic={args.ic} rmse={rmse:.4f} "
          f"mean_spread={sd.mean().item():.4f}")

    # De-normalise for vorticity-unit plotting.
    # For the conditioning panel, show the coarsened-then-nearest-upsampled field so
    # the viewer sees the pixelation (the model internally trains on bilinear).
    coarsen_f = ck_args['coarsen']
    x0_coarse = F.avg_pool2d(x0_full, coarsen_f, stride=coarsen_f)
    cond_blocky = F.interpolate(x0_coarse, size=(H, H), mode='nearest')
    cond = (cond_blocky[0, 0] * norm).cpu().numpy()
    truth = (x1_truth[0, 0] * norm).cpu().numpy()
    # Observation: upsample blocky to 128 for side-by-side visual parity.
    y_obs_up = F.interpolate(y_obs, size=(H, H), mode='nearest')
    obs_field = (y_obs_up[0, 0] * norm).cpu().numpy()
    post_mean = (mean[0, 0] * norm).cpu().numpy()
    post_sd = (sd[0, 0] * norm).cpu().numpy()
    err = np.abs(post_mean - truth)

    # Set colour scales
    vlim = float(np.percentile(np.abs(np.concatenate([truth.flatten(),
                                                      post_mean.flatten(),
                                                      cond.flatten()])), 99.5))
    vlim_obs = float(np.percentile(np.abs(obs_field.flatten()), 99.5))

    # --- Figure (no titles / captions — user will add them in LaTeX) ---
    sns.set_theme(context='paper', style='white', font_scale=0.95)
    panels = [
        (cond,       sns.cm.icefire, -vlim,     vlim,     True),
        (truth,      sns.cm.icefire, -vlim,     vlim,     False),
        (obs_field,  sns.cm.icefire, -vlim_obs, vlim_obs, True),
        (post_mean,  sns.cm.icefire, -vlim,     vlim,     False),
        (post_sd,    'magma',        0,         float(post_sd.max()), False),
        (err,        'magma',        0,         float(err.max()),     False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    axes = axes.flatten()
    for ax, (field, cmap, vmin, vmax, is_blocky) in zip(axes, panels):
        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                       interpolation='nearest' if is_blocky else 'bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=24)
        cb.ax.tick_params(labelsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight', dpi=300)
    fig.savefig(args.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=220)
    print(f"[saved] {args.out}")
    print(f"[saved] {args.out.replace('.pdf', '.png')}")


if __name__ == '__main__':
    main()
