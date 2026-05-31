"""
Multiscale per-band SHORTCUT MODELS for Gaussian random fields.

Same infrastructure as train_multiscale_perband_meanflow.py, but uses the
Shortcut Models loss (Frans et al. 2024) instead of MeanFlow JVP.

Shortcut self-consistency (no JVP):
  For d = r - s > 0: bootstrap target from two half-steps
    m = (s + r) / 2
    v_first = u(z_s, s, m)             (no_grad)
    z_m = z_s + (m-s) * v_first
    v_second = u(z_m, m, r)            (no_grad)
    target = (v_first + v_second) / 2
    loss = ||u(z_s, s, r) - stopgrad(target)||^2 / σ²

  For d = 0 (r = s): standard FM loss against v = x_F - z_0_F.

Memory advantage: 3 forward passes total (1 with grad + 2 no_grad), no JVP doubling.
This lets us train the finest band at full batch size.
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
from time import time as timer
from matplotlib import pyplot as plt
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)
from train_multiscale_perband import setup, ridge_regression, embed_mask, extract_mask
from train_multiscale_perband_meanflow import BandMFNet, generate_band_mf


# ─── Shortcut loss ──────────────────────────────────────────────────────────

def shortcut_loss_step(net, embed_fn, extract_fn, zt_F, target_v, s, r, sigma2):
    """
    Shortcut self-consistency loss.

    Args:
      net: per-band UNet with (s, r) conditioning
      embed_fn: zt_F -> 2-channel R×R image
      extract_fn: net output -> band pixels
      zt_F: interpolated band pixels (B, |F|)
      target_v: ground truth velocity for FM (B, |F|), used when r=s
      s, r: time tensors (B,) with r >= s
      sigma2: scalar variance for loss normalization

    Returns:
      scalar loss
    """
    # Full-step prediction (with grad)
    w_full = extract_fn(net(embed_fn(zt_F), s, r))

    # Bootstrap target via two half-steps (no grad)
    m = (s + r) / 2

    with torch.no_grad():
        # First half: (s, m)
        v_first = extract_fn(net(embed_fn(zt_F), s, m))
        d_first = (m - s).unsqueeze(1)
        zm_F = zt_F + d_first * v_first
        # Second half: (m, r) from new state
        v_second = extract_fn(net(embed_fn(zm_F), m, r))

    target_shortcut = (v_first + v_second) / 2

    # Where r == s (FM mode), use ground truth velocity instead
    is_fm = (r == s).unsqueeze(1).float()
    target_use = is_fm * target_v + (1.0 - is_fm) * target_shortcut

    error = w_full - target_use.detach()
    return error.pow(2).mean() / sigma2


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--steps', type=str, default='50000,50000,50000,100000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--n_eval', type=int, default=500)
    p.add_argument('--n_ridge', type=int, default=10000)
    p.add_argument('--flow_ratio', type=float, default=0.75,
                   help='fraction of batch using r=s (pure FM); shortcut paper uses 0.75')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.G; K = args.K
    steps_list = [int(x) for x in args.steps.split(',')]

    amp = setup(G, args.s)
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    # ── Bands ────────────────────────────────────────────────────────
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

    # ── σ² per band ─────────────────────────────────────────────────
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

    while len(steps_list) < len(bands):
        steps_list.append(steps_list[-1])

    print(f'Perband SHORTCUT: mask G={G} K={K} flow_ratio={args.flow_ratio}')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:2d} |F|={len(b['F']):5d} σ={b['scale']:.4e} steps={ns}")

    run_name = f'perband_shortcut_K{K}_G{G}_fr{args.flow_ratio}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine ────────────────────────────────────────
    gen = torch.zeros(args.n_eval, G*G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']; sigma2 = b['sigma2']
        n_steps = steps_list[bi]

        net = BandMFNet(R, b['in_ch'], b['out_ch'], args.ch).float().to(device)
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        def make_embed(xC_loc, F_=F, C_=C, G_=G, R_=R):
            return lambda z: embed_mask(z, xC_loc, F_, C_, G_, R_, device)

        def make_extract(F_=F, C_=C, G_=G, R_=R):
            if len(C_) > 0:
                return lambda p: extract_mask(p, F_, G_, R_, device)
            else:
                return lambda p: p.view(p.shape[0], -1)[:, :len(F_)]

        extract_fn = make_extract()
        warmup_steps = max(n_steps // 10, 500)

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, σ={scale:.4e}\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            data = sample_matern_batch(amp, args.batch, device=device).float().view(args.batch, -1)
            xF = data[:, F]
            xC = data[:, C] if len(C) > 0 else None

            z0_F = scale * torch.randn(args.batch, nF, device=device)
            target_v = xF - z0_F  # FM target

            # Sample (s, r) — same scheme as MeanFlow
            t1 = torch.rand(args.batch, device=device) * 0.998 + 0.001
            t2 = torch.rand(args.batch, device=device) * 0.998 + 0.001
            s = torch.minimum(t1, t2)
            r = torch.maximum(t1, t2)
            flow_mask = torch.rand(args.batch, device=device) < args.flow_ratio
            r = torch.where(flow_mask, s, r)

            zt_F = (1 - s).unsqueeze(1) * z0_F + s.unsqueeze(1) * xF

            embed_fn_train = make_embed(xC)

            if step <= warmup_steps:
                # Warmup: pure FM (force r=s, no shortcut bootstrap)
                w = extract_fn(net(embed_fn_train(zt_F), s, s))
                error = w - target_v
                loss = error.pow(2).mean() / sigma2
            else:
                # Shortcut self-consistency
                loss = shortcut_loss_step(net, embed_fn_train, extract_fn,
                                           zt_F, target_v, s, r, sigma2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            sch.step()

            if step % max(1, n_steps//20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]
                elapsed = (timer()-t0)/60
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} lr={lr:.2e} [{elapsed:.1f}m]")
                if not args.no_wandb:
                    wandb.log({f'loss/{b["name"]}': loss.item()}, step=step+bi*200000)

            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                xC_eval = gen[:, C] if len(C) > 0 else None
                embed_eval = make_embed(xC_eval)
                for n_mf in [1, 2, 4]:
                    gen_F = generate_band_mf(net, embed_eval, extract_fn, scale,
                                              args.n_eval, nF, device, n_mf_steps=n_mf)
                    gen_temp = gen.clone(); gen_temp[:, F] = gen_F
                    gpix = gen_temp.view(args.n_eval, G, G)
                    stride_e = G // R
                    gen_R = gpix[:, ::stride_e, ::stride_e].cpu().numpy()
                    tru_R = test_data[:, ::stride_e, ::stride_e].numpy()
                    _, sg = get_fourier_spectrum(gen_R)
                    _, st = get_fourier_spectrum(tru_R)
                    kmax = min(R//2-1, len(sg))
                    rel = np.abs(sg[:kmax]-st[:kmax])/(np.abs(st[:kmax])+1e-30)
                    print(f"  [eval R={R} MF{n_mf}] mean_rel≤{kmax}={rel.mean():.4f} max_rel≤{kmax}={rel.max():.4f}")
                    if not args.no_wandb:
                        wandb.log({f'eval/mean_rel_R{R}_MF{n_mf}': rel.mean()}, step=step+bi*200000)

        save_dir = f'results/perband_shortcut_K{K}'
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))

        net.eval()
        xC_final = gen[:, C] if len(C) > 0 else None
        gen[:, F] = generate_band_mf(net, make_embed(xC_final), extract_fn, scale,
                                      args.n_eval, nF, device, n_mf_steps=2)

    # ── Final eval ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nFINAL — multi-MF-step eval\n{'='*60}")
    for n_mf in [1, 2, 4]:
        gen2 = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            net2 = BandMFNet(R2, b['in_ch'], b['out_ch'], args.ch).float().to(device)
            net2.load_state_dict(torch.load(os.path.join(f'results/perband_shortcut_K{K}', b['name'], 'model.pt'),
                                            map_location=device, weights_only=True))
            net2.eval()
            ef2 = make_embed(xC2, F_=F2, C_=C2, G_=G, R_=R2)
            xf2 = make_extract(F_=F2, C_=C2, G_=G, R_=R2)
            gen2[:, F2] = generate_band_mf(net2, ef2, xf2, b['scale'], args.n_eval, nF2, device, n_mf_steps=n_mf)

        gpix2 = gen2.view(args.n_eval, G, G).cpu().numpy()
        _, sg2 = get_fourier_spectrum(gpix2)
        std_r = gpix2.std() / (truth_np.std() + 1e-12)
        rel2 = np.abs(sg2 - spec_truth) / (np.abs(spec_truth) + 1e-30)
        nfe = n_mf * len(bands)
        print(f"  MF{n_mf}/band (NFE={nfe}): mean_rel={rel2.mean():.4f} max_rel={rel2.max():.4f} std_ratio={std_r:.4f}")
        if not args.no_wandb:
            wandb.log({f'final/mean_rel_MF{n_mf}': rel2.mean(), f'final/std_ratio_MF{n_mf}': std_r})

    if not args.no_wandb:
        wandb.finish()
