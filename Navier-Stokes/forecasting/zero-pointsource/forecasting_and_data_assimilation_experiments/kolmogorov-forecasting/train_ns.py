"""Train hat_b_theta for the §3.4.1 baseline drift on Kolmogorov forecasting.

One training per (lag, seed); sweep g at sample time via drift_compose_ns.compose_drift.

Usage:
    python train_ns.py --lag 10 --seed 0 --max_steps 20000
"""
import argparse
import os
import time

import torch

from data import load_snapshots, split_train_val_test, make_loaders
from interpolant_ns import INTERPOLANTS
from network_ns import DriftNet
from schedules_ns import list_schedules, make_g
from drift_compose_ns import compose_drift
from sampler_ns import em_sample
from metrics_ns import (crps_ensemble, rmse_ensemble_mean, ensemble_spread,
                        spread_skill_ratio, spectrum_rmse, enstrophy_w2, vorticity_pdf_w2)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.manual_seed(args.seed)
    ip = INTERPOLANTS[args.interpolant]()

    # Data
    print(f"[data] loading {args.data}")
    omega_all, sim_args = load_snapshots(args.data)
    N_traj = omega_all.shape[0]
    train_idx, val_idx, test_idx = split_train_val_test(
        N_traj, args.n_train, args.n_val, args.n_test, seed=args.data_seed)
    train_dl, val_dl, test_dl = make_loaders(
        omega_all, args.lag, train_idx, val_idx, test_idx,
        args.batch_size, coarsen_factor=args.coarsen, num_workers=0)
    H = W = omega_all.shape[-1]
    print(f"[data] H={H}, lag={args.lag}, train pairs={len(train_dl.dataset)}, "
          f"val={len(val_dl.dataset)}, test={len(test_dl.dataset)}")

    # Normalise — compute from training slice
    tr_std = omega_all[train_idx].std().item()
    print(f"[data] train std = {tr_std:.3f}")
    norm = tr_std

    # Model
    net = DriftNet(unet_channels=args.unet_channels,
                   unet_dim_mults=tuple(args.unet_dim_mults)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] params = {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb
        run_name = f"kolm_lag{args.lag}_seed{args.seed}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=run_name, config=vars(args))

    # Training loop
    start = time.time()
    step = 0
    data_iter = iter(train_dl)
    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)
        net.train()
        x1 = batch['x1'].to(device, non_blocking=True) / norm
        x0_up = batch['x0_up'].to(device, non_blocking=True) / norm
        B = x1.shape[0]
        s = torch.rand(B, 1, device=device) * (1.0 - args.t_eps) + args.t_eps
        z = torch.randn_like(x1)
        xs = ip.It(x1, z, s)
        rb = ip.Rb(x1, z, s)
        pred = net(xs, s, x0_up)
        loss = (pred - rb).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"[step {step:6d}] loss={loss.item():.4e}  "
                  f"({time.time() - start:.1f}s)", flush=True)
            if use_wandb:
                import wandb
                wandb.log({'loss': loss.item()}, step=step)

        if step > 0 and step % args.eval_every == 0:
            _eval(net, ip, val_dl, args, step, device, norm, use_wandb)

    _eval(net, ip, val_dl, args, args.max_steps, device, norm, use_wandb)

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"lag{args.lag}_seed{args.seed}.pt")
    torch.save({'state': net.state_dict(), 'args': vars(args), 'norm': norm}, path)
    print(f"[saved] {path}")
    if use_wandb:
        import wandb
        wandb.finish()


@torch.no_grad()
def _eval(net, ip, val_dl, args, step, device, norm, use_wandb):
    net.eval()

    # Small ensemble on a reference batch
    batch = next(iter(val_dl))
    x1_ref = batch['x1'].to(device) / norm
    x0_up_ref = batch['x0_up'].to(device) / norm
    Bref = min(args.eval_batch, x1_ref.shape[0])
    x1_ref = x1_ref[:Bref]; x0_up_ref = x0_up_ref[:Bref]

    def b_fn(x, t, x0): return net(x, t, x0)
    report = {}
    import math
    for name in list_schedules():
        g_fn = make_g(name, interpolant=ip)
        bg = compose_drift(b_fn, g_fn, ip)
        ens = torch.zeros(args.eval_ensemble, Bref, 1,
                          x1_ref.shape[-2], x1_ref.shape[-1], device=device)
        for k in range(args.eval_ensemble):
            ens[k] = em_sample(bg, g_fn, (Bref, 1, x1_ref.shape[-2], x1_ref.shape[-1]),
                               n_steps=args.eval_n_em, t_min=args.t_eps,
                               t_max=1.0 - args.t_eps, cond=(x0_up_ref,), device=device)
        truth = x1_ref
        report[f'crps/{name}'] = crps_ensemble(ens, truth)
        report[f'rmse/{name}'] = rmse_ensemble_mean(ens, truth)
        report[f'spread/{name}'] = ensemble_spread(ens)
        report[f'ssr/{name}'] = spread_skill_ratio(ens, truth)
        report[f'spec_rmse/{name}'] = spectrum_rmse(ens.squeeze(2), truth.squeeze(1))
        report[f'enstrophy_w2/{name}'] = enstrophy_w2(ens.squeeze(2), truth.squeeze(1))
        report[f'pdf_w2/{name}'] = vorticity_pdf_w2(ens.squeeze(2), truth.squeeze(1))

    print(f"  [eval step={step}]")
    for k in sorted(report):
        print(f"    {k:22s}  {report[k]:.3e}")
    if use_wandb:
        import wandb
        wandb.log(report, step=step)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str,
                   default='/home/yifanchen/research/GenerativeDynamics-NumericalDesign/'
                           'NSdata/kolmogorov_128/data.pt')
    p.add_argument('--lag', type=int, default=10)
    p.add_argument('--coarsen', type=int, default=4)
    p.add_argument('--n_train', type=int, default=500)
    p.add_argument('--n_val', type=int, default=100)
    p.add_argument('--n_test', type=int, default=100)
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--unet_channels', type=int, default=32)
    p.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 2, 2])
    p.add_argument('--t_eps', type=float, default=1e-3)
    p.add_argument('--eval_every', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--eval_batch', type=int, default=4)
    p.add_argument('--eval_ensemble', type=int, default=8)
    p.add_argument('--eval_n_em', type=int, default=100)
    p.add_argument('--save_dir', type=str, default='./runs')
    p.add_argument('--use_wandb', type=int, default=0)
    p.add_argument('--wandb_project', type=str, default='interpolants_follmer_kolm')
    p.add_argument('--wandb_entity', type=str, default='yifanc96')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--interpolant', type=str, default='linlin', choices=['linlin', 'sqlin'])
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    train(args)
