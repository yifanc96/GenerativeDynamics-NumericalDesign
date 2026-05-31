"""
v2 of multiscale per-band flow matching for Gaussian fields. Same method as
train_multiscale_perband.py (mask, raw variant, loss/sigma2). Adds:
  * EMA of model weights (--ema_decay)
  * Logit-normal time sampling (--time_dist logit_normal --time_mu --time_sigma)
  * Per-band channel widths (--ch_list a,b,c,d)
  * Per-band step counts (--steps a,b,c,d)

Goal: close the trained-vs-oracle gap on Gaussian fields.
Reference baseline (current best with this script's defaults disabled): 0.21 on G=64 mask.
Oracle: 0.028.
"""
import os, sys, math, argparse, copy
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


# ---------- σ² + linear conditional mean ----------
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


# ---------- Mask embedding (1-channel raw) ----------
def embed_mask(zt_F, x_C, F, C, G, R, device):
    B = zt_F.shape[0]
    if len(C) == 0:
        return zt_F.view(B, 1, R, R)
    stride = G // R
    full = torch.zeros(B, G*G, device=device); full[:, F] = zt_F; full[:, C] = x_C
    ch1 = full.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    ctx = torch.zeros(B, G*G, device=device); ctx[:, C] = x_C
    ch2 = ctx.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ch1, ch2], dim=1)


def extract_mask(pred, F, G, R, device):
    B = pred.shape[0]; stride = G // R
    if stride == 1: return pred.view(B, -1)[:, F]
    pf = torch.zeros(B, G, G, device=device)
    pf[:, ::stride, ::stride] = pred.view(B, R, R)
    return pf.view(B, -1)[:, F]


# ---------- EMA ----------
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
        # returns a backup of original params; caller can restore_from(backup)
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


# ---------- Time sampling ----------
def sample_t(B, dist, mu, sigma, device):
    if dist == 'uniform':
        return torch.rand(B, device=device) * 0.998 + 0.001
    elif dist == 'logit_normal':
        # Stable Diffusion 3 / Esser et al.: t = sigmoid(mu + sigma * z), z~N(0,1)
        z = torch.randn(B, device=device)
        t = torch.sigmoid(mu + sigma * z)
        return t.clamp(1e-3, 1 - 1e-3)
    raise ValueError(dist)


# ---------- Per-band sample (for raw variant only) ----------
def generate_band_raw(net_eval, embed_fn, extract_fn, scale, n_samples, nF, device, n_rk4=4):
    zt = scale * torch.randn(n_samples, nF, device=device)
    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    for i in range(len(nodes)-1):
        sv = float(nodes[i]); ds = float(nodes[i+1]-nodes[i])
        def vel(z, tv):
            inp = embed_fn(z)
            tt = torch.full((n_samples,), tv, device=device)
            with torch.no_grad(): p = net_eval(inp, tt, classes=None)
            return extract_fn(p)
        k1=vel(zt, sv); k2=vel(zt+.5*ds*k1, sv+.5*ds)
        k3=vel(zt+.5*ds*k2, sv+.5*ds); k4=vel(zt+ds*k3, sv+ds)
        zt = zt + (ds/6)*(k1+2*k2+2*k3+k4)
    return zt


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--ch_list', type=str, default='32,32,32,32',
                   help='Per-band channel widths, comma-separated, coarse-to-fine')
    p.add_argument('--steps', type=str, default='20000,20000,40000,80000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr_min_ratio', type=float, default=0.01,
                   help='cosine min LR as fraction of peak LR')
    p.add_argument('--warmup', type=int, default=1000,
                   help='linear LR warmup steps (per band)')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='max gradient norm (set to 0 or large to disable)')
    p.add_argument('--ema_decay', type=float, default=0.0)
    p.add_argument('--time_dist', type=str, default='logit_normal', choices=['uniform', 'logit_normal'])
    p.add_argument('--time_mu', type=float, default=0.0)
    p.add_argument('--time_sigma', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=500)
    p.add_argument('--n_ridge', type=int, default=10000)
    p.add_argument('--tag', type=str, default='v2', help='Run tag for results dir / wandb')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.G; K = args.K
    ch_list = [int(x) for x in args.ch_list.split(',')]
    steps_list = [int(x) for x in args.steps.split(',')]
    while len(ch_list) < K + 1: ch_list.append(ch_list[-1])
    while len(steps_list) < K + 1: steps_list.append(steps_list[-1])

    amp = setup_amplitude(G, args.s)
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    # ---------- Bands: mask, coarse-to-fine ----------
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

    # ---------- σ² estimation ----------
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

    print(f"v2 sweep — mask raw, G={G} K={K}, ema={args.ema_decay}, t={args.time_dist}({args.time_mu},{args.time_sigma})")
    for b, ch, ns in zip(bands, ch_list, steps_list):
        print(f"  {b['name']:12s} R={b['R']:2d} |F|={len(b['F']):5d} σ={b['scale']:.4e} ch={ch} steps={ns}")

    run_name = f'v2_mask_K{K}_G{G}_{args.tag}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    save_dir = f'results/bench/mask_raw_K{K}_{args.tag}'
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Train coarse-to-fine ----------
    gen = torch.zeros(args.n_eval, G * G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        ch = ch_list[bi]; n_steps = steps_list[bi]

        net = make_unet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        # Warmup + cosine schedule combined via LambdaLR
        warmup = max(0, min(args.warmup, n_steps - 1))
        peak = 1.0
        floor = args.lr_min_ratio
        def lr_lambda(s, w=warmup, T=n_steps, fl=floor):
            if w > 0 and s < w:
                return (s + 1) / w
            # cosine from peak -> floor over (T - w) steps
            if T - w <= 0: return fl
            x = (s - w) / (T - w)
            x = max(0.0, min(1.0, x))
            return fl + 0.5 * (peak - fl) * (1 + math.cos(math.pi * x))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        ema = EMA(net, args.ema_decay) if args.ema_decay > 0 else None

        def make_embed(xC_loc):
            return lambda z: embed_mask(z, xC_loc, F, C, G, R, device)
        def make_extract():
            if len(C) > 0:
                return lambda p: extract_mask(p, F, G, R, device)
            return lambda p: p.view(p.shape[0], -1)[:, :nF]
        extract_fn = make_extract()

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, ch={ch}, {npar:,} params, {n_steps} steps\n{'='*60}", flush=True)

        for step in range(1, n_steps + 1):
            net.train()
            data = sample_matern_batch(amp, args.batch, device=device).float().view(args.batch, -1)
            xF = data[:, F]; xC = data[:, C] if len(C) > 0 else None

            z0 = scale * torch.randn(args.batch, nF, device=device)
            t = sample_t(args.batch, args.time_dist, args.time_mu, args.time_sigma, device)
            a = (1 - t).unsqueeze(1); bcoef = t.unsqueeze(1)
            zt_F = a * z0 + bcoef * xF
            target = xF - z0

            inp = make_embed(xC)(zt_F)
            pred = extract_fn(net(inp, t, classes=None))
            raw_mse = (pred - target).pow(2).mean()
            loss = raw_mse / b['sigma2']

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
                # use EMA weights for eval
                backup = ema.copy_to(net) if ema is not None else None

                xC_eval = gen[:, C] if len(C) > 0 else None
                embed_eval = make_embed(xC_eval)
                gen_F = generate_band_raw(net, embed_eval, extract_fn, scale, args.n_eval, nF, device, n_rk4=4)

                gen_temp = gen.clone(); gen_temp[:, F] = gen_F
                stride_e = G // R
                gen_R = gen_temp.view(args.n_eval, G, G)[:, ::stride_e, ::stride_e].cpu().numpy()
                tru_R = test_data[:, ::stride_e, ::stride_e].numpy()
                _, sg = get_fourier_spectrum(gen_R)
                _, st = get_fourier_spectrum(tru_R)
                kmax = min(R // 2 - 1, len(sg))
                rel = np.abs(sg[:kmax] - st[:kmax]) / (np.abs(st[:kmax]) + 1e-30)
                print(f"  [eval R={R}] mean_rel≤{kmax}={rel.mean():.4f} max_rel≤{kmax}={rel.max():.4f}", flush=True)
                if not args.no_wandb:
                    wandb.log({f'eval/mean_rel_R{R}': rel.mean(), f'eval/max_rel_R{R}': rel.max()})

                if ema is not None: ema.restore_from(net, backup)

        # save EMA weights
        net.eval()
        if ema is not None: ema.copy_to(net)
        torch.save(net.state_dict(), os.path.join(save_dir, f"{b['name']}.pt"))
        # commit final generated band (with EMA weights)
        xC_eval = gen[:, C] if len(C) > 0 else None
        embed_eval = make_embed(xC_eval)
        gen[:, F] = generate_band_raw(net, embed_eval, extract_fn, scale, args.n_eval, nF, device, n_rk4=4)

    # ---------- Final multi-RK4 eval ----------
    print(f"\n{'='*60}\nFINAL — multi-RK4 eval\n{'='*60}", flush=True)
    summary = {}
    for n_rk4 in [2, 4, 8, 16]:
        gen2 = torch.zeros(args.n_eval, G * G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']; scale2 = b['scale']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            net2 = make_unet(R2, b['in_ch'], b['out_ch'], ch_list[bi]).float().to(device)
            net2.load_state_dict(torch.load(os.path.join(save_dir, f"{b['name']}.pt"),
                                            map_location=device, weights_only=True))
            net2.eval()
            ef2 = lambda z, xC_=xC2, F_=F2, C_=C2, R_=R2: embed_mask(z, xC_, F_, C_, G, R_, device)
            xf2 = (lambda p, F_=F2, R_=R2: extract_mask(p, F_, G, R_, device)) if len(C2) > 0 else (lambda p: p.view(args.n_eval, -1))
            gen2[:, F2] = generate_band_raw(net2, ef2, xf2, scale2, args.n_eval, nF2, device, n_rk4=n_rk4)
        gn2 = gen2.view(args.n_eval, G, G).cpu().numpy()
        _, sg2 = get_fourier_spectrum(gn2)
        rel2 = np.abs(sg2 - spec_truth) / (np.abs(spec_truth) + 1e-30)
        nb1 = int(np.sum(rel2[:30] < 1))
        m = rel2[:30].mean(); mx = rel2[:30].max()
        print(f"  RK4={n_rk4:2d}: mean_rel≤30={m:.4f} max_rel≤30={mx:.4f} bins<1={nb1}/30", flush=True)
        summary[n_rk4] = (m, mx, nb1)
        if not args.no_wandb:
            wandb.log({f'final/mean_rel_RK{n_rk4}': m, f'final/max_rel_RK{n_rk4}': mx, f'final/bins_lt_1_RK{n_rk4}': nb1})

    print(f"\nTotal time: {(timer()-t0)/60:.1f} min")
    print(f"Summary: " + " ".join(f"RK{k}={v[0]:.3f}" for k, v in summary.items()))
    if not args.no_wandb:
        wandb.finish()


def setup_amplitude(G, s):
    sigma_sq = 1.0 * ((2 * math.pi)**2 + 1.0**2)**s
    return precompute_matern_amplitude(G, sigma_sq, 1.0, s).float()


if __name__ == '__main__':
    main()
