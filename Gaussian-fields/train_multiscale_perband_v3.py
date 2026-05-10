"""
v3: EDM-flavoured preconditioning for per-band flow matching on Gaussian fields.

Recipe (synthesized from Karras EDM, SD3, CDM, WSGM):
  * EDM-style preconditioning translated to FM:
      - Network sees unit-variance input: x_t / sigma_x_t
      - Network output rescaled by sigma_target = sigma_F * sqrt(1+kappa)
      - Loss is MSE on unit-variance target (no explicit /sigma^2 needed)
  * Logit-normal time sampling t = sigmoid(m + s*z), default m=0, s=1
  * Per-band channel widening: ch=[32, 48, 64, 96] for R=[8,16,32,64]
  * EMA decay 0.999 (window ~1000 steps), matched to per-band step count
  * Grad clip 100 (catches outliers; natural gradnorm here is ~25)
  * Warmup 1000 + cosine

Reference: Karras EDM (2206.00364), SD3 (2403.03206), CDM (2106.15282).
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
from time import time as timer
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)


def ridge_regression(Y, X, ridge=1e-6):
    n = Y.shape[0]
    Xa = torch.cat([torch.ones(n, 1, device=Y.device), X], dim=1)
    beta = torch.linalg.solve(Xa.T @ Xa + ridge * torch.eye(Xa.shape[1], device=Y.device), Xa.T @ Y)
    sigma2 = (Y - Xa @ beta).var(dim=0).mean().item()
    return sigma2, beta[1:].T, beta[0]


def make_unet(R, ic, oc, ch=32):
    dm = (1, 2) if R <= 8 else (1, 2, 2) if R <= 32 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)


def embed_mask_normalized(zt_F, x_C, F, C, G, R, device, sigma_data, sigma_x_t):
    """1-channel embedding with per-sample input normalization.
    Both the F (interpolant) and C (already-generated coarse) values are
    normalized by sigma_data so the network sees roughly unit-variance input.
    """
    B = zt_F.shape[0]
    if len(C) == 0:
        # Just unit-variance input
        return (zt_F / sigma_x_t.view(B, 1)).view(B, 1, R, R)
    stride = G // R
    full = torch.zeros(B, G*G, device=device)
    full[:, F] = zt_F / sigma_x_t.view(B, 1)
    full[:, C] = x_C / sigma_data
    ch1 = full.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    ctx = torch.zeros(B, G*G, device=device); ctx[:, C] = x_C / sigma_data
    ch2 = ctx.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ch1, ch2], dim=1)


def extract_mask(pred, F, G, R, device):
    B = pred.shape[0]; stride = G // R
    if stride == 1: return pred.view(B, -1)[:, F]
    pf = torch.zeros(B, G, G, device=device)
    pf[:, ::stride, ::stride] = pred.view(B, R, R)
    return pf.view(B, -1)[:, F]


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore_from(self, model, backup):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(backup[n])


def sample_t(B, dist, mu, sigma, device):
    if dist == 'uniform':
        return torch.rand(B, device=device) * 0.998 + 0.001
    elif dist == 'logit_normal':
        z = torch.randn(B, device=device)
        return torch.sigmoid(mu + sigma * z).clamp(1e-3, 1 - 1e-3)
    raise ValueError(dist)


def compute_sigmas(t, sigma_F, kappa=1.0):
    """Per-sample sigma_x_t (std of interpolant) and sigma_target (std of target).

    For raw FM with z0~N(0, sigma_F^2 I) and Var(x_F | x_C) ~ sigma_F^2 * kappa,
    target = x_F - z0,  Var(target) = sigma_F^2 (1 + kappa).
    Var(x_t | x_C) = (1-t)^2 sigma_F^2 + t^2 sigma_F^2 * kappa.
    We pass kappa=1 (worst case, ensures preconditioning never blows up).
    """
    sigma_x_t = sigma_F * torch.sqrt((1 - t)**2 + t**2 * kappa)
    sigma_target = sigma_F * math.sqrt(1.0 + kappa)
    return sigma_x_t, sigma_target


def generate_band_edm(net_eval, F, C, G, R, device, sigma_F, n_samples, nF, n_rk4=4):
    """Generate one band via RK4 with EDM preconditioning at sample time."""
    zt = sigma_F * torch.randn(n_samples, nF, device=device)
    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    sigma_target = sigma_F * math.sqrt(2.0)  # kappa=1

    # x_C is captured externally; we'll pass it via closure
    return zt, nodes, sigma_target  # caller will run the loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--ch_list', type=str, default='32,48,64,96')
    p.add_argument('--steps', type=str, default='15000,20000,30000,60000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr_min_ratio', type=float, default=0.01)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--grad_clip', type=float, default=100.0)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--time_dist', type=str, default='logit_normal',
                   choices=['uniform', 'logit_normal'])
    p.add_argument('--time_mu', type=float, default=0.0)
    p.add_argument('--time_sigma', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=500)
    p.add_argument('--n_ridge', type=int, default=10000)
    p.add_argument('--tag', type=str, default='v3')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.G; K = args.K
    ch_list = [int(x) for x in args.ch_list.split(',')]
    steps_list = [int(x) for x in args.steps.split(',')]
    while len(ch_list) < K + 1: ch_list.append(ch_list[-1])
    while len(steps_list) < K + 1: steps_list.append(steps_list[-1])

    sigma_sq = 1.0 * ((2*math.pi)**2 + 1.0**2)**args.s
    amp = precompute_matern_amplitude(G, sigma_sq, 1.0, args.s).float()
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    # Bands (mask, coarse-to-fine)
    hier = HierarchicalMasks(G, K + 1, device='cpu')
    bands = []
    for k in range(K + 1):
        si = K - k
        F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten()
        Cl = [torch.nonzero(hier.masks[K - j].cpu().flatten()).flatten() for j in range(k)]
        C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
        R = G // (2**(K - k))
        bands.append(dict(F=F, C=C, R=R, in_ch=1 if k == 0 else 2, out_ch=1,
                          name=f'mask_s{k}_R{R}'))

    # Per-band sigma estimation
    est = sample_matern_batch(amp, args.n_ridge, device=device).float().view(args.n_ridge, -1)
    for b in bands:
        Ft = b['F'].to(device); Y = est[:, Ft]
        if len(b['C']) == 0:
            b['sigma2'] = Y.var(dim=0).mean().item()
            b['M_op'] = None; b['intercept'] = None
        else:
            Ct = b['C'].to(device); X = est[:, Ct]
            b['sigma2'], b['M_op'], b['intercept'] = ridge_regression(Y, X)
        b['scale'] = math.sqrt(b['sigma2'])
    del est

    print(f"v3 EDM-flavoured — mask raw, G={G} K={K}, t={args.time_dist}({args.time_mu},{args.time_sigma}), ema={args.ema_decay}, clip={args.grad_clip}, warmup={args.warmup}")
    for b, ch, ns in zip(bands, ch_list, steps_list):
        print(f"  {b['name']:12s} R={b['R']:2d} |F|={len(b['F']):5d} sigma_F={b['scale']:.4e} ch={ch} steps={ns}")

    run_name = f'v3_mask_K{K}_G{G}_{args.tag}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    save_dir = f'results/bench/v3_{args.tag}_K{K}'
    os.makedirs(save_dir, exist_ok=True)

    gen = torch.zeros(args.n_eval, G * G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']
        sigma_F = b['scale']
        sigma_target = sigma_F * math.sqrt(2.0)  # kappa=1 (z0+x_F std)
        ch = ch_list[bi]; n_steps = steps_list[bi]

        net = make_unet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
        warmup = max(0, min(args.warmup, n_steps - 1))
        floor = args.lr_min_ratio
        def lr_lambda(s, w=warmup, T=n_steps, fl=floor):
            if w > 0 and s < w: return (s + 1) / w
            if T - w <= 0: return fl
            x = (s - w) / (T - w)
            x = max(0.0, min(1.0, x))
            return fl + 0.5 * (1 - fl) * (1 + math.cos(math.pi * x))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        ema = EMA(net, args.ema_decay) if args.ema_decay > 0 else None

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, ch={ch}, {npar:,} params, {n_steps} steps, sigma_F={sigma_F:.3e}\n{'='*60}", flush=True)

        for step in range(1, n_steps + 1):
            net.train()
            data = sample_matern_batch(amp, args.batch, device=device).float().view(args.batch, -1)
            xF = data[:, F]; xC = data[:, C] if len(C) > 0 else None

            z0 = sigma_F * torch.randn(args.batch, nF, device=device)
            t = sample_t(args.batch, args.time_dist, args.time_mu, args.time_sigma, device)
            a = (1 - t).unsqueeze(1); bcoef = t.unsqueeze(1)
            zt_F = a * z0 + bcoef * xF
            target = xF - z0

            sigma_x_t, _ = compute_sigmas(t, sigma_F, kappa=1.0)

            inp = embed_mask_normalized(zt_F, xC, F, C, G, R, device, sigma_F, sigma_x_t)
            out_raw = net(inp, t, classes=None)
            if len(C) > 0:
                pred_normed = extract_mask(out_raw, F, G, R, device)
            else:
                pred_normed = out_raw.view(args.batch, -1)[:, :nF]

            # Predict normalized target: target / sigma_target
            target_normed = target / sigma_target
            loss = (pred_normed - target_normed).pow(2).mean()

            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip).item()
            else:
                gn = float('nan')
            opt.step(); sch.step()
            if ema is not None: ema.update(net)

            if step % max(1, n_steps // 20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]; el = (timer() - t0) / 60
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} gn={gn:.2f} lr={lr:.2e} [{el:.1f}m]", flush=True)
                if not args.no_wandb:
                    wandb.log({f'loss/{b["name"]}': loss.item(), f'gradnorm/{b["name"]}': gn, 'step': step + bi * 200000})

            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                backup = ema.copy_to(net) if ema is not None else None

                xC_eval = gen[:, C] if len(C) > 0 else None
                gen_F = sample_band(net, F, C, G, R, device, sigma_F, sigma_target, args.n_eval, nF, xC_eval, n_rk4=4)

                gen_temp = gen.clone(); gen_temp[:, F] = gen_F
                stride_e = G // R
                gen_R = gen_temp.view(args.n_eval, G, G)[:, ::stride_e, ::stride_e].cpu().numpy()
                tru_R = test_data[:, ::stride_e, ::stride_e].numpy()
                _, sg = get_fourier_spectrum(gen_R)
                _, st = get_fourier_spectrum(tru_R)
                kmax = min(R // 2 - 1, len(sg))
                rel = np.abs(sg[:kmax] - st[:kmax]) / (np.abs(st[:kmax]) + 1e-30)
                print(f"  [eval R={R}] mean_rel<={kmax}={rel.mean():.4f} max_rel<={kmax}={rel.max():.4f}", flush=True)
                if not args.no_wandb:
                    wandb.log({f'eval/mean_rel_R{R}': rel.mean(), f'eval/max_rel_R{R}': rel.max()})

                if ema is not None: ema.restore_from(net, backup)

        # Save EMA + commit final band
        net.eval()
        if ema is not None: ema.copy_to(net)
        torch.save(net.state_dict(), os.path.join(save_dir, f"{b['name']}.pt"))
        xC_eval = gen[:, C] if len(C) > 0 else None
        gen[:, F] = sample_band(net, F, C, G, R, device, sigma_F, sigma_target, args.n_eval, nF, xC_eval, n_rk4=4)

    # Final multi-RK4 eval
    print(f"\n{'='*60}\nFINAL — multi-RK4 eval\n{'='*60}", flush=True)
    summary = {}
    for n_rk4 in [2, 4, 8, 16]:
        gen2 = torch.zeros(args.n_eval, G * G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']; sigma_F2 = b['scale']
            sigma_target2 = sigma_F2 * math.sqrt(2.0)
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            net2 = make_unet(R2, b['in_ch'], b['out_ch'], ch_list[bi]).float().to(device)
            net2.load_state_dict(torch.load(os.path.join(save_dir, f"{b['name']}.pt"),
                                            map_location=device, weights_only=True))
            net2.eval()
            gen2[:, F2] = sample_band(net2, F2, C2, G, R2, device, sigma_F2, sigma_target2, args.n_eval, nF2, xC2, n_rk4=n_rk4)
        gn2 = gen2.view(args.n_eval, G, G).cpu().numpy()
        _, sg2 = get_fourier_spectrum(gn2)
        rel2 = np.abs(sg2 - spec_truth) / (np.abs(spec_truth) + 1e-30)
        nb1 = int(np.sum(rel2[:30] < 1))
        m = rel2[:30].mean(); mx = rel2[:30].max()
        # per-bin breakdown
        nb01 = int(np.sum(rel2[:30] < 0.1))
        print(f"  RK4={n_rk4:2d}: mean_rel<=30={m:.4f} max_rel<=30={mx:.4f} bins<1={nb1}/30 bins<0.1={nb01}/30", flush=True)
        summary[n_rk4] = (m, mx, nb1, nb01)
        if not args.no_wandb:
            wandb.log({f'final/mean_rel_RK{n_rk4}': m, f'final/max_rel_RK{n_rk4}': mx,
                       f'final/bins_lt_1_RK{n_rk4}': nb1, f'final/bins_lt_0p1_RK{n_rk4}': nb01})

    # Per-bin at best RK4
    best = min(summary, key=lambda k: summary[k][0])
    print(f"\nBest RK4={best}, mean_rel<=30={summary[best][0]:.4f}")
    print(f"\nTotal time: {(timer()-t0)/60:.1f} min")
    if not args.no_wandb:
        wandb.finish()


def sample_band(net, F, C, G, R, device, sigma_F, sigma_target, n_samples, nF, xC, n_rk4=4):
    """RK4 sampler with EDM-style normalized network input/output."""
    zt = sigma_F * torch.randn(n_samples, nF, device=device)
    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4 + 1)
    for i in range(len(nodes) - 1):
        sv = float(nodes[i]); ds = float(nodes[i + 1] - nodes[i])

        def vel(z, tv):
            tt = torch.full((n_samples,), tv, device=device)
            sigma_x_t, _ = compute_sigmas(tt, sigma_F, kappa=1.0)
            inp = embed_mask_normalized(z, xC, F, C, G, R, device, sigma_F, sigma_x_t)
            with torch.no_grad():
                out_raw = net(inp, tt, classes=None)
            if len(C) > 0:
                pred_normed = extract_mask(out_raw, F, G, R, device)
            else:
                pred_normed = out_raw.view(n_samples, -1)[:, :nF]
            return pred_normed * sigma_target  # un-normalize to get drift

        k1 = vel(zt, sv); k2 = vel(zt + .5 * ds * k1, sv + .5 * ds)
        k3 = vel(zt + .5 * ds * k2, sv + .5 * ds); k4 = vel(zt + ds * k3, sv + ds)
        zt = zt + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return zt


if __name__ == '__main__':
    main()
