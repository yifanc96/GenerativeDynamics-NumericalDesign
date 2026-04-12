"""
Multiscale per-band flow matching for Gaussian random fields.

Supports both mask (pixel-space) and Haar wavelet decompositions.
Each band is trained independently with its own scale-appropriate UNet.
Training proceeds coarse-to-fine; each band conditions on previously resolved bands.

Recommended variant: "raw"
  - z0 ~ N(0, σ²I) where σ² = conditional variance (estimated via ridge regression)
  - Interpolant: zt = (1-t)*z0 + t*x_F
  - Target: x_F - z0
  - Loss: MSE / σ² (reweighted so all bands have O(1) loss)
  - Network input: raw-scale interpolant at F pixels + raw-scale coarse context at C pixels
  - At eval: start from z0, integrate with RK4, output = final state

Also supports variants: rescale, center, center_rescale (for comparison).

Results (G=64, Matérn s=3, K=3, 4 RK4 steps per band):
  mask raw: mean_rel≤30 = 0.21, all 30 bins below 1
  haar raw: mean_rel≤30 = 0.30, all 30 bins below 1

Usage:
  python train_multiscale_perband.py --decomp mask --variant raw --gpu 0
  python train_multiscale_perband.py --decomp haar --variant raw --gpu 1
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
from time import time as timer
from matplotlib import pyplot as plt
import pywt
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)


def setup(G, s):
    sigma_sq = 1.0 * ((2*math.pi)**2 + 1.0**2)**s
    return precompute_matern_amplitude(G, sigma_sq, 1.0, s).float()


def ridge_regression(Y, X, ridge=1e-6):
    """Returns sigma2, M_op (|F|×|C|), intercept (|F|,)."""
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


# ─── Embedding (same for train and eval) ──────────────────────────────────
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


def embed_haar(zt_detail, x_C_wav, C_w, W, G, R, device):
    B = zt_detail.shape[0]; ns = R*R; stride = G // R
    if len(C_w) == 0:
        return zt_detail.view(B, 1, R, R)
    LH = zt_detail[:, :ns].view(B, 1, R, R)
    HL = zt_detail[:, ns:2*ns].view(B, 1, R, R)
    HH = zt_detail[:, 2*ns:].view(B, 1, R, R)
    wc = torch.zeros(B, G*G, device=device); wc[:, C_w] = x_C_wav
    ll = (W.T @ wc.T).T.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ll, LH, HL, HH], dim=1)


# ─── Flow matching step ───────────────────────────────────────────────────
def flow_step(x_F, x_C, mu, scale, variant, batch_size, nF, device):
    """Compute zt, target, and the F values to embed, given variant."""
    if variant == 'raw':
        z0 = scale * torch.randn(batch_size, nF, device=device)
        zt_F = None  # will be set below
        target = x_F - z0
        # zt in raw scale
        t = torch.rand(batch_size, device=device) * 0.998 + 0.001
        a = (1-t).unsqueeze(1); b = t.unsqueeze(1)
        zt_F = a * z0 + b * x_F
        return zt_F, target, t, x_C  # embed F=zt_F, C=x_C, both raw

    elif variant == 'rescale':
        z0 = torch.randn(batch_size, nF, device=device)
        t = torch.rand(batch_size, device=device) * 0.998 + 0.001
        a = (1-t).unsqueeze(1); b = t.unsqueeze(1)
        zt_F = a * z0 + b * (x_F / scale)  # F normalized
        target = (x_F / scale) - z0
        return zt_F, target, t, x_C  # embed F=zt_F (O(xF/σ)), C=x_C raw

    elif variant == 'center':
        x_c = x_F - mu
        z0 = scale * torch.randn(batch_size, nF, device=device)
        t = torch.rand(batch_size, device=device) * 0.998 + 0.001
        a = (1-t).unsqueeze(1); b = t.unsqueeze(1)
        zt_F = a * z0 + b * x_c
        target = x_c - z0
        return zt_F, target, t, x_C  # embed F=zt_F (centered raw), C=x_C raw

    elif variant == 'center_rescale':
        x_c = x_F - mu
        z0 = torch.randn(batch_size, nF, device=device)
        t = torch.rand(batch_size, device=device) * 0.998 + 0.001
        a = (1-t).unsqueeze(1); b = t.unsqueeze(1)
        zt_F = a * z0 + b * (x_c / scale)  # centered + normalized
        target = (x_c / scale) - z0
        return zt_F, target, t, x_C  # embed F=zt_F (O(1)), C=x_C raw


def generate_band(net, embed_fn, extract_fn, scale, mu, variant, n_samples, nF, device, n_rk4=4):
    """Generate one band via RK4."""
    if variant in ('raw', 'center'):
        zt = scale * torch.randn(n_samples, nF, device=device)
    else:
        zt = torch.randn(n_samples, nF, device=device)

    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    for i in range(len(nodes)-1):
        sv = float(nodes[i]); ds = float(nodes[i+1]-nodes[i])
        def vel(z, tv):
            inp = embed_fn(z)
            tt = torch.full((n_samples,), tv, device=device)
            with torch.no_grad(): p = net(inp, tt, classes=None)
            return extract_fn(p)
        k1=vel(zt,sv); k2=vel(zt+.5*ds*k1,sv+.5*ds)
        k3=vel(zt+.5*ds*k2,sv+.5*ds); k4=vel(zt+ds*k3,sv+ds)
        zt = zt + (ds/6)*(k1+2*k2+2*k3+k4)

    # Un-normalize / un-center
    if variant == 'raw':
        return zt
    elif variant == 'rescale':
        return zt * scale
    elif variant == 'center':
        return zt + mu
    elif variant == 'center_rescale':
        return zt * scale + mu


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--decomp', type=str, default='mask', choices=['mask', 'haar'])
    p.add_argument('--variant', type=str, default='center_rescale',
                   choices=['raw', 'rescale', 'center', 'center_rescale'])
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--steps', type=str, default='20000,20000,20000,40000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=500)
    p.add_argument('--n_ridge', type=int, default=10000)
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.G; K = args.K; variant = args.variant
    steps_list = [int(x) for x in args.steps.split(',')]

    amp = setup(G, args.s)
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    # ── Build DWT matrix (for Haar) ────────────────────────────────────
    W_dev = None
    if args.decomp == 'haar':
        d = G*G
        W_np = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            delta = np.zeros((G, G)); delta.flat[i] = 1.0
            c = pywt.wavedec2(delta, 'haar', level=K, mode='periodization')
            flat = list(c[0].ravel())
            for j in range(1, len(c)):
                for sub in c[j]: flat.extend(sub.ravel())
            W_np[:, i] = flat
        W_dev = torch.from_numpy(W_np).to(device)

    # ── Build bands ────────────────────────────────────────────────────
    if args.decomp == 'mask':
        hier = HierarchicalMasks(G, K+1, device='cpu')
        bands = []
        for k in range(K+1):
            si = K - k
            F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten()
            Cl = [torch.nonzero(hier.masks[K-j].cpu().flatten()).flatten() for j in range(k)]
            C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
            R = G // (2**(K-k))
            bands.append(dict(F=F, C=C, R=R, in_ch=1 if k==0 else 2, out_ch=1,
                              name=f'mask_s{k}_R{R}'))
    else:
        c = pywt.wavedec2(np.zeros((G, G)), 'haar', level=K, mode='periodization')
        bands = []; idx = 0; coarse = []
        n_ll = int(np.prod(c[0].shape)); R_ll = c[0].shape[0]
        bands.append(dict(F=torch.arange(idx, idx+n_ll), C=torch.empty(0, dtype=torch.long),
                          R=R_ll, in_ch=1, out_ch=1, name='LL'))
        coarse.extend(range(idx, idx+n_ll)); idx += n_ll
        for j in range(1, len(c)):
            li = []; Rj = c[j][0].shape[0]
            for sub in c[j]:
                n = int(np.prod(sub.shape)); li.extend(range(idx, idx+n)); idx += n
            bands.append(dict(F=torch.tensor(li), C=torch.tensor(coarse[:]),
                              R=Rj, in_ch=4, out_ch=3, name=f'det_L{j}'))
            coarse.extend(li)

    # ── Estimate σ² and M_op per band ──────────────────────────────────
    est = sample_matern_batch(amp, args.n_ridge, device=device).float().view(args.n_ridge, -1)
    if args.decomp == 'haar':
        est = (W_dev @ est.T).T
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

    while len(steps_list) < len(bands):
        steps_list.append(steps_list[-1])

    print(f'{args.decomp} {variant} G={G} K={K}')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:2d} |F|={len(b['F']):5d} σ={b['scale']:.4e} steps={ns}")

    # ── Wandb ──────────────────────────────────────────────────────────
    run_name = f'{args.decomp}_{variant}_K{K}_G{G}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine ───────────────────────────────────────────
    gen = torch.zeros(args.n_eval, G*G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        M_op = b['M_op']; intercept = b['intercept']
        n_steps = steps_list[bi]

        net = make_unet(R, b['in_ch'], b['out_ch'], args.ch).float().to(device)
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        # Embedding closures
        def make_embed(xC_loc):
            if args.decomp == 'mask':
                return lambda z: embed_mask(z, xC_loc, F, C, G, R, device)
            else:
                return lambda z: embed_haar(z, xC_loc, C, W_dev, G, R, device) if len(C) > 0 else z.view(-1, 1, R, R)

        def make_extract():
            if args.decomp == 'mask' and len(C) > 0:
                return lambda p: extract_mask(p, F, G, R, device)
            else:
                return lambda p: p.view(p.shape[0], -1)[:, :nF]

        extract_fn = make_extract()

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, {n_steps} steps\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            data = sample_matern_batch(amp, args.batch, device=device).float().view(args.batch, -1)
            if args.decomp == 'haar': data = (W_dev @ data.T).T
            xF = data[:, F]
            xC = data[:, C] if len(C) > 0 else None

            # Conditional mean
            if M_op is not None and xC is not None:
                mu = intercept.unsqueeze(0) + xC @ M_op.T
            else:
                mu = torch.zeros(args.batch, nF, device=device)

            zt_F, target, t, xC_embed = flow_step(xF, xC, mu, scale, variant, args.batch, nF, device)
            inp = make_embed(xC_embed)(zt_F)
            pred = extract_fn(net(inp, t, classes=None))
            raw_mse = (pred - target).pow(2).mean()
            # Reweight: for raw/center targets are O(σ), for rescale/center_rescale targets are O(1)
            if variant in ('raw', 'center'):
                loss = raw_mse / b['sigma2']  # O(σ²)/σ² = O(1)
            else:
                loss = raw_mse  # already O(1)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e4)
            opt.step(); sch.step()

            if step % max(1, n_steps//20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]
                elapsed = (timer()-t0)/60
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} lr={lr:.2e} [{elapsed:.1f}m]")
                if not args.no_wandb:
                    wandb.log({f'loss/{b["name"]}': loss.item(), 'step': step+bi*100000})

            # ── Eval ──
            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                xC_eval = gen[:, C] if len(C) > 0 else None
                if M_op is not None and xC_eval is not None:
                    mu_eval = intercept.unsqueeze(0) + xC_eval @ M_op.T
                else:
                    mu_eval = torch.zeros(args.n_eval, nF, device=device)

                embed_eval = make_embed(xC_eval)
                gen_F = generate_band(net, embed_eval, extract_fn, scale, mu_eval,
                                       variant, args.n_eval, nF, device)

                gen_temp = gen.clone(); gen_temp[:, F] = gen_F
                if args.decomp == 'haar':
                    gpix = (W_dev.T @ gen_temp.T).T
                else:
                    gpix = gen_temp

                # Eval at current resolution R
                stride_e = G // R
                gen_R = gpix.view(args.n_eval, G, G)[:, ::stride_e, ::stride_e].cpu().numpy()
                tru_R = test_data[:, ::stride_e, ::stride_e].numpy()
                _, sg = get_fourier_spectrum(gen_R)
                _, st = get_fourier_spectrum(tru_R)
                kmax = min(R//2-1, len(sg))
                rel = np.abs(sg[:kmax]-st[:kmax])/(np.abs(st[:kmax])+1e-30)
                print(f"  [eval R={R}] mean_rel≤{kmax}={rel.mean():.4f} max_rel≤{kmax}={rel.max():.4f}")
                if not args.no_wandb:
                    wandb.log({f'eval/mean_rel_R{R}': rel.mean(), f'eval/max_rel_R{R}': rel.max()})

        # Save checkpoint and store generated band
        save_dir = f'results/bench/{args.decomp}_{variant}_K{K}'
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))
        gen[:, F] = gen_F

    # ── Final full-resolution eval at multiple RK4 steps ───────────────
    save_dir_final = f'results/bench/{args.decomp}_{variant}_K{K}'
    print(f"\n{'='*60}\nFINAL — multi-RK4 eval\n{'='*60}")
    for n_rk4 in [2, 4, 8, 16]:
        gen2 = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']; scale2 = b['scale']
            M2 = b['M_op']; I2 = b['intercept']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            if M2 is not None and xC2 is not None:
                mu2 = I2.unsqueeze(0) + xC2 @ M2.T
            else:
                mu2 = torch.zeros(args.n_eval, nF2, device=device)
            net2 = make_unet(R2, b['in_ch'], b['out_ch'], args.ch).float().to(device)
            net2.load_state_dict(torch.load(os.path.join(save_dir_final, b['name'], 'model.pt'),
                                            map_location=device, weights_only=True))
            net2.eval()
            # Inline embedding
            if args.decomp == 'mask':
                ef2 = lambda z, xC_=xC2, F_=F2, C_=C2, R_=R2: embed_mask(z, xC_, F_, C_, G, R_, device)
                xf2 = (lambda p, F_=F2, R_=R2: extract_mask(p, F_, G, R_, device)) if len(C2) > 0 else (lambda p: p.view(args.n_eval, -1))
            else:
                ef2 = (lambda z, xC_=xC2, C_=C2, R_=R2: embed_haar(z, xC_, C_, W_dev, G, R_, device)) if len(C2) > 0 else (lambda z, R_=R2: z.view(args.n_eval, 1, R_, R_))
                xf2 = lambda p, nF_=nF2: p.view(args.n_eval, -1)[:, :nF_]
            gen2[:, F2] = generate_band(net2, ef2, xf2, scale2, mu2, variant, args.n_eval, nF2, device, n_rk4=n_rk4)
        if args.decomp == 'haar':
            gpix2 = (W_dev.T @ gen2.T).T
        else:
            gpix2 = gen2
        gn2 = gpix2.view(args.n_eval, G, G).cpu().numpy()
        _, sg2 = get_fourier_spectrum(gn2)
        rel2 = np.abs(sg2 - spec_truth) / (np.abs(spec_truth) + 1e-30)
        nb1 = np.sum(rel2[:30] < 1)
        print(f"  RK4={n_rk4:2d}: mean_rel≤30={rel2[:30].mean():.4f} max_rel≤30={rel2[:30].max():.4f} bins<1={nb1}/30")
    if args.decomp == 'haar':
        gpix = (W_dev.T @ gen.T).T
    else:
        gpix = gen
    gen_np = gpix.view(args.n_eval, G, G).cpu().numpy()
    _, spec_gen = get_fourier_spectrum(gen_np)
    rel = np.abs(spec_gen-spec_truth)/(np.abs(spec_truth)+1e-30)
    print(f"{'k':>4} {'E_truth':>10} {'E_gen':>10} {'ratio':>8} {'rel':>8}")
    for i in range(len(kvals)):
        r = spec_gen[i]/(spec_truth[i]+1e-30)
        m = ' ***' if abs(r-1)>1 else (' * ' if abs(r-1)>0.1 else '   ')
        print(f"{kvals[i]:>4.0f} {spec_truth[i]:>10.3e} {spec_gen[i]:>10.3e} {r:>8.4f} {abs(r-1):>8.4f}{m}")
    print(f"\nmean_rel≤30={rel[:30].mean():.4f} max_rel≤30={rel[:30].max():.4f}")

    if not args.no_wandb:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        ax.loglog(kvals, spec_gen, 'r--', label='generated')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)'); ax.legend()
        ax.set_title(f'{args.decomp} {variant} final')
        wandb.log({'final_spectrum': wandb.Image(fig)})
        plt.close()
        wandb.finish()

    print(f"\nTotal time: {(timer()-t0)/60:.1f} min")
