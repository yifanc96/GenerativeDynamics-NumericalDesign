"""Train one (v_theta, s_theta) pair on a fixed Gaussian-base interpolant.
At sample time, compose b^(g) = v_theta + (g^2 / 2) s_theta for any schedule g(t).

Usage:
    python toy_train.py --target gaussian1d --seed 0 --max_steps 20000
"""
import argparse
import os
import time

import torch

from interpolant import GaussianBaseInterpolant
from networks import MLPNet
from targets import make_target
from schedules import list_schedules, make_g, compose_drift
from sampler import em_sample
from metrics import wasserstein_1d, path_kl_girsanov


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.manual_seed(args.seed)
    ip = GaussianBaseInterpolant()
    target = make_target(args.target, device=device)
    cond_dim = 1 if target.conditional else 0

    v_net = MLPNet(x_dim=target.dim, cond_dim=cond_dim, hidden=args.hidden,
                   n_layers=args.n_layers, time_embed=args.time_embed).to(device)
    s_net = MLPNet(x_dim=target.dim, cond_dim=cond_dim, hidden=args.hidden,
                   n_layers=args.n_layers, time_embed=args.time_embed).to(device)
    opt = torch.optim.AdamW(list(v_net.parameters()) + list(s_net.parameters()), lr=args.lr)

    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb
        run_name = f"{args.target}_seed{args.seed}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name,
                   config=vars(args))

    start = time.time()
    for step in range(args.max_steps):
        v_net.train(); s_net.train()
        # With Gaussian base, gamma(1) = 0; exclude a tiny slice near t=1 to avoid score singularity.
        t = torch.rand(args.batch_size, 1, device=device) * (1.0 - args.t_eps)
        z = torch.randn(args.batch_size, target.dim, device=device)
        if target.conditional:
            x1, y = target.sample_x1(args.batch_size)
            cond = (y,)
        else:
            x1 = target.sample_x1(args.batch_size)
            cond = ()
        xt = ip.It(x1, z, t)
        rv = ip.Rv(x1, z, t)                                      # velocity target
        gamma_t = ip.gamma(t).clamp_min(args.t_eps)
        v_pred = v_net(xt, t, *cond)
        s_pred = s_net(xt, t, *cond)
        loss_v = ((v_pred - rv) ** 2).sum(dim=-1).mean()
        # Weighted score loss: || gamma * s_theta + z ||^2, bounded at t -> 1.
        loss_s = ((gamma_t * s_pred + z) ** 2).sum(dim=-1).mean()
        loss = loss_v + loss_s

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(v_net.parameters()) + list(s_net.parameters()), max_norm=5.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"[step {step:6d}] loss={loss.item():.4f}  v={loss_v.item():.4f}  "
                  f"s={loss_s.item():.4f}  ({time.time() - start:.1f}s)")
            if use_wandb:
                import wandb
                wandb.log({'loss/total': loss.item(), 'loss/v': loss_v.item(),
                           'loss/s': loss_s.item()}, step=step)

        if step > 0 and step % args.eval_every == 0:
            _eval(v_net, s_net, ip, target, args, step, device, use_wandb=use_wandb)

    _eval(v_net, s_net, ip, target, args, args.max_steps, device, use_wandb=use_wandb,
          n_samples=args.final_n_samples)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.target}_seed{args.seed}.pt")
    torch.save({
        'v_state': v_net.state_dict(), 's_state': s_net.state_dict(),
        'args': vars(args),
    }, save_path)
    print(f"[saved] {save_path}")
    if use_wandb:
        import wandb
        wandb.finish()


def _eval(v_net, s_net, ip, target, args, step, device, use_wandb=False, n_samples=None):
    n_samples = n_samples or args.eval_n_samples
    v_net.eval(); s_net.eval()

    def v_fn(x, t, *cond): return v_net(x, t, *cond)
    def s_fn(x, t, *cond): return s_net(x, t, *cond)

    # Truth samples
    torch.manual_seed(args.seed + 1000 + step)
    if target.conditional:
        (y_ref,) = target.sample_cond(n_samples)
        x1_ref, _ = target.sample_x1(n_samples, y=y_ref)
        truth = x1_ref
    else:
        truth = target.sample_x1(n_samples)

    t_min_s, t_max_s = 0.0, 1.0 - args.t_eps

    # Common analytic drift closures per schedule
    def analytic_v(x, t, *cond):
        if target.conditional:
            return target.v_star(x, t, ip, *cond)
        return target.v_star(x, t, ip)

    def analytic_s(x, t, *cond):
        if target.conditional:
            return target.s_star(x, t, ip, *cond)
        return target.s_star(x, t, ip)

    report = {}
    for name in list_schedules():
        g_fn = make_g(name, epsilon=args.epsilon, device=device)
        b_theta = compose_drift(v_fn, s_fn, g_fn)
        b_star = compose_drift(analytic_v, analytic_s, g_fn)

        if name != 'ode':
            kl = path_kl_girsanov(b_theta, b_star, g_fn, ip, target,
                                  n_mc=args.eval_n_mc, t_min=0.0, t_max=1.0 - args.t_eps,
                                  device=device, dtype=torch.float32)
            report[f'kl/{name}'] = kl
        if target.conditional:
            (y_s,) = target.sample_cond(n_samples)
            x1_s, _ = target.sample_x1(n_samples, y=y_s)
            samples = em_sample(b_theta, g_fn, n_samples, target.dim,
                                n_steps=args.eval_n_em, t_min=t_min_s, t_max=t_max_s,
                                cond=(y_s,), device=device, init='gaussian')
            w2 = wasserstein_1d(samples, x1_s, p=2) if target.dim == 1 else float('nan')
        else:
            samples = em_sample(b_theta, g_fn, n_samples, target.dim,
                                n_steps=args.eval_n_em, t_min=t_min_s, t_max=t_max_s,
                                device=device, init='gaussian')
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
    p.add_argument('--epsilon', type=float, default=0.5)
    p.add_argument('--t_eps', type=float, default=1e-3)
    p.add_argument('--eval_every', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=500)
    p.add_argument('--eval_n_samples', type=int, default=20000)
    p.add_argument('--eval_n_mc', type=int, default=40000)
    p.add_argument('--eval_n_em', type=int, default=200)
    p.add_argument('--final_n_samples', type=int, default=40000)
    p.add_argument('--save_dir', type=str, default='./runs')
    p.add_argument('--use_wandb', type=int, default=1)
    p.add_argument('--wandb_project', type=str, default='interpolants_follmer_toy')
    p.add_argument('--wandb_entity', type=str, default='yifanc96')
    p.add_argument('--cpu', action='store_true')
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    train(args)
