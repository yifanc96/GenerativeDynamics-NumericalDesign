"""One-time training of hat_b_theta for the §3.4.1 baseline drift
    b_t(x) = E[ x_star - sqrt(t) z | x_t = x ]
via standard L2 DSM. The same hat_b_theta is used for all g schedules at
sample time via eq (3.9). No per-schedule training.

Usage:
    python train_zps.py --target gaussian1d --seed 0 --max_steps 20000
"""
import argparse
import os
import time

import torch

from interpolant_zps import ZPSInterpolant
from networks import MLPNet
from targets_zps import make_target
from schedules_zps import list_schedules, make_g
from drift_compose import compose_drift
from sampler import em_sample
from metrics import wasserstein_1d, path_kl_girsanov


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.manual_seed(args.seed)
    ip = ZPSInterpolant()
    target = make_target(args.target, device=device)
    cond_dim = 1 if target.conditional else 0

    b_net = MLPNet(x_dim=target.dim, cond_dim=cond_dim, hidden=args.hidden,
                   n_layers=args.n_layers, time_embed=args.time_embed).to(device)
    opt = torch.optim.AdamW(b_net.parameters(), lr=args.lr)

    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb
        run_name = f"zps_{args.target}_seed{args.seed}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name,
                   config=vars(args))

    start = time.time()
    for step in range(args.max_steps):
        b_net.train()
        # t ~ U[t_eps, 1], Rb = x_star - sqrt(t) z is bounded
        t = torch.rand(args.batch_size, 1, device=device) * (1.0 - args.t_eps) + args.t_eps
        z = torch.randn(args.batch_size, target.dim, device=device)
        if target.conditional:
            x1, y = target.sample_x1(args.batch_size)
            cond = (y,)
        else:
            x1 = target.sample_x1(args.batch_size)
            cond = ()
        xt = ip.It(x1, z, t)
        rb = ip.Rb(x1, z, t)
        b_pred = b_net(xt, t, *cond)
        loss = ((b_pred - rb) ** 2).sum(dim=-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(b_net.parameters(), max_norm=5.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"[step {step:6d}] loss={loss.item():.4f}  ({time.time() - start:.1f}s)")
            if use_wandb:
                import wandb
                wandb.log({'loss': loss.item()}, step=step)

        if step > 0 and step % args.eval_every == 0:
            _eval(b_net, ip, target, args, step, device, use_wandb=use_wandb)

    _eval(b_net, ip, target, args, args.max_steps, device, use_wandb=use_wandb,
          n_samples=args.final_n_samples)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.target}_seed{args.seed}.pt")
    torch.save({'b_state': b_net.state_dict(), 'args': vars(args)}, save_path)
    print(f"[saved] {save_path}")
    if use_wandb:
        import wandb
        wandb.finish()


def _eval(b_net, ip, target, args, step, device, use_wandb=False, n_samples=None):
    n_samples = n_samples or args.eval_n_samples
    b_net.eval()

    def b_theta_fn(x, t, *cond):
        return b_net(x, t, *cond)

    # truth
    torch.manual_seed(args.seed + 1000 + step)
    if target.conditional:
        (y_ref,) = target.sample_cond(n_samples)
        x1_ref, _ = target.sample_x1(n_samples, y=y_ref)
        truth = x1_ref
    else:
        truth = target.sample_x1(n_samples)

    def b_star_fn(x, t, *cond):
        return target.b_star(x, t, *cond) if target.conditional else target.b_star(x, t)

    report = {}
    for name in list_schedules():
        g_fn = make_g(name, scale=1.0)
        bg_theta = compose_drift(b_theta_fn, g_fn)
        bg_star = compose_drift(b_star_fn, g_fn)
        if name != 'zero':
            kl = path_kl_girsanov(bg_theta, bg_star, g_fn, target,
                                  n_mc=args.eval_n_mc,
                                  t_min=args.t_eps, t_max=1.0 - args.t_eps,
                                  device=device, dtype=torch.float32)
            report[f'kl/{name}'] = kl
        if target.conditional:
            (y_s,) = target.sample_cond(n_samples)
            x1_s, _ = target.sample_x1(n_samples, y=y_s)
            samples = em_sample(bg_theta, g_fn, n_samples, target.dim,
                                n_steps=args.eval_n_em, t_min=args.t_eps,
                                t_max=1.0 - args.t_eps, cond=(y_s,), device=device)
            w2 = wasserstein_1d(samples, x1_s, p=2) if target.dim == 1 else float('nan')
        else:
            samples = em_sample(bg_theta, g_fn, n_samples, target.dim,
                                n_steps=args.eval_n_em, t_min=args.t_eps,
                                t_max=1.0 - args.t_eps, device=device)
            w2 = wasserstein_1d(samples, truth, p=2) if target.dim == 1 else float('nan')
        report[f'w2/{name}'] = w2

    print(f"  [eval step={step}]")
    for k in sorted(report):
        print(f"    {k:20s} {report[k]:.4e}")
    if use_wandb:
        import wandb
        wandb.log(report, step=step)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=str, default='gaussian1d',
                   choices=['gaussian1d', 'bimodal1d', 'ou_forecast'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--time_embed', type=int, default=64)
    p.add_argument('--t_eps', type=float, default=1e-3)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--log_every', type=int, default=500)
    p.add_argument('--eval_n_samples', type=int, default=10000)
    p.add_argument('--eval_n_mc', type=int, default=40000)
    p.add_argument('--eval_n_em', type=int, default=200)
    p.add_argument('--final_n_samples', type=int, default=40000)
    p.add_argument('--save_dir', type=str, default='./runs')
    p.add_argument('--use_wandb', type=int, default=0)
    p.add_argument('--wandb_project', type=str, default='interpolants_follmer_zps')
    p.add_argument('--wandb_entity', type=str, default='yifanc96')
    p.add_argument('--cpu', action='store_true')
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    train(args)
