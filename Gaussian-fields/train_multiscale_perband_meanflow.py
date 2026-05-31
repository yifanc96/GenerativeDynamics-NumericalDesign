"""
Multiscale per-band MEAN FLOW for Gaussian random fields.

Adapted from train_multiscale_perband.py — same sequential coarse-to-fine
structure, same per-band UNet at native resolution, same embedding/extraction.

Only change: velocity matching → JVP mean flow loss, and RK4 → 1-2 MF steps.

Usage:
  python train_multiscale_perband_meanflow.py --decomp mask --gpu 0
"""
import os, sys, math, argparse, copy
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
from train_multiscale_perband import (
    setup, ridge_regression, embed_mask, extract_mask,
)


# ─── Per-band mean flow UNet ────────────────────────────────────────────────

def make_mf_unet(R, ic, oc, ch=32):
    """Create a UNet + r_mlp for mean flow conditioning at resolution R."""
    dm = (1, 2) if R <= 8 else (1, 2, 2) if R <= 32 else (1, 2, 2, 2)
    net = Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
               resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
               random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
               attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)
    return net


class BandMFNet(nn.Module):
    """Per-band UNet with (s, r) mean flow conditioning."""
    def __init__(self, R, ic, oc, ch=32):
        super().__init__()
        self.net = make_mf_unet(R, ic, oc, ch)
        time_dim = ch * 4
        lsd = max(ch, 16)
        sinu = RandomOrLearnedSinusoidalPosEmb(lsd, is_random=False)
        fourier_dim = lsd + 1
        self.r_mlp = nn.Sequential(
            sinu, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )
        nn.init.zeros_(self.r_mlp[-1].weight)
        nn.init.zeros_(self.r_mlp[-1].bias)

    def forward(self, x, s, r):
        net = self.net
        s_emb = net.time_mlp(s)
        r_emb = self.r_mlp(r)
        t_emb = s_emb + r_emb
        x = net.init_conv(x)
        r_skip = x.clone()
        h = []
        for block1, block2, attn, downsample in net.downs:
            x = block1(x, t_emb, None); h.append(x)
            x = block2(x, t_emb, None); x = attn(x); h.append(x)
            x = downsample(x)
        x = net.mid_block1(x, t_emb, None); x = net.mid_attn(x); x = net.mid_block2(x, t_emb, None)
        for block1, block2, attn, upsample in net.ups:
            x = torch.cat((x, h.pop()), dim=1); x = block1(x, t_emb, None)
            x = torch.cat((x, h.pop()), dim=1); x = block2(x, t_emb, None); x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r_skip), dim=1)
        x = net.final_res_block(x, t_emb, None)
        return net.final_conv(x)


# ─── Mean flow generation (replaces RK4) ─────────────────────────────────────

def generate_band_mf(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_mf_steps=2):
    """Generate one band via mean flow steps (replaces RK4)."""
    zt = scale * torch.randn(n_samples, nF, device=device)
    s_vals = torch.linspace(0.0, 1.0, n_mf_steps + 1)
    for j in range(n_mf_steps):
        s_j = float(s_vals[j])
        r_j = float(s_vals[j + 1])
        s_batch = torch.full((n_samples,), s_j, device=device)
        r_batch = torch.full((n_samples,), r_j, device=device)
        inp = embed_fn(zt)
        with torch.no_grad():
            pred = net(inp, s_batch, r_batch)
        vel = extract_fn(pred)
        zt = zt + (r_j - s_j) * vel
    return zt


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--decomp', type=str, default='mask', choices=['mask'])
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--batch_fine', type=int, default=100, help='batch size for finest band (JVP needs more memory)')
    p.add_argument('--steps', type=str, default='50000,50000,50000,100000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=500)
    p.add_argument('--n_ridge', type=int, default=10000)
    p.add_argument('--flow_ratio', type=float, default=0.5)
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

    # ── Build bands (mask decomposition, same as original) ────────────
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

    # ── Estimate σ² per band (same as original) ──────────────────────
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

    print(f'Perband MEAN FLOW: mask G={G} K={K}')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:2d} |F|={len(b['F']):5d} σ={b['scale']:.4e} steps={ns}")

    run_name = f'perband_mf_mask_K{K}_G{G}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine (SEQUENTIAL, same as original) ───────────
    gen = torch.zeros(args.n_eval, G*G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']; sigma2 = b['sigma2']
        M_op = b['M_op']; intercept = b['intercept']
        n_steps = steps_list[bi]

        net = BandMFNet(R, b['in_ch'], b['out_ch'], args.ch).float().to(device)
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        # Embedding closures (same as original)
        def make_embed(xC_loc, F_=F, C_=C, G_=G, R_=R):
            return lambda z: embed_mask(z, xC_loc, F_, C_, G_, R_, device)

        def make_extract(F_=F, C_=C, G_=G, R_=R):
            if len(C_) > 0:
                return lambda p: extract_mask(p, F_, G_, R_, device)
            else:
                return lambda p: p.view(p.shape[0], -1)[:, :len(F_)]

        extract_fn = make_extract()

        # Use smaller batch for large bands (JVP doubles memory)
        band_batch = args.batch_fine if R >= 64 else args.batch

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, {n_steps} steps, σ={scale:.4e}, batch={band_batch}\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            data = sample_matern_batch(amp, band_batch, device=device).float().view(band_batch, -1)
            xF = data[:, F]
            xC = data[:, C] if len(C) > 0 else None

            # Noise and interpolant (raw variant: z0 ~ N(0, σ²I))
            z0_F = scale * torch.randn(band_batch, nF, device=device)
            target = xF - z0_F  # velocity

            # Sample (s, r) for mean flow
            t1 = torch.rand(band_batch, device=device) * 0.998 + 0.001
            t2 = torch.rand(band_batch, device=device) * 0.998 + 0.001
            s = torch.minimum(t1, t2)
            r = torch.maximum(t1, t2)
            flow_mask = torch.rand(band_batch, device=device) < args.flow_ratio
            r = torch.where(flow_mask, s, r)

            # Interpolate
            zt_F = (1 - s).unsqueeze(1) * z0_F + s.unsqueeze(1) * xF

            # Warmup: first 10% of steps use pure flow matching (no JVP)
            # to get the network output to the right scale before JVP kicks in
            warmup_steps = max(n_steps // 10, 500)
            use_jvp = (step > warmup_steps)

            embed_fn_train = make_embed(xC)

            if use_jvp:
                # Full mean flow with JVP
                def fn(zt_F_, s_, r_, _embed=embed_fn_train, _extract=extract_fn, _net=net):
                    inp = _embed(zt_F_)
                    pred = _net(inp, s_, r_)
                    return _extract(pred)

                primals = (zt_F, s, r)
                tangents = (target, torch.ones_like(s), torch.zeros_like(r))
                w, dw_ds = torch.func.jvp(fn, primals, tangents)

                dr = (r - s).unsqueeze(1)
                w_tgt = target + dr * dw_ds
                error = w - w_tgt.detach()
            else:
                # Pure flow matching warmup (r = s, so w_tgt = target)
                inp = embed_fn_train(zt_F)
                w = extract_fn(net(inp, s, s))  # r = s
                error = w - target

            # Loss reweighted by σ² (same as original: makes loss O(1))
            loss = error.pow(2).mean() / sigma2

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

            # ── Eval ──
            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                xC_eval = gen[:, C] if len(C) > 0 else None
                embed_eval = make_embed(xC_eval)

                # Mean flow generation at multiple step counts
                for n_mf in [1, 2, 4]:
                    gen_F = generate_band_mf(net, embed_eval, extract_fn, scale,
                                              args.n_eval, nF, device, n_mf_steps=n_mf)
                    gen_temp = gen.clone(); gen_temp[:, F] = gen_F
                    gpix = gen_temp

                    stride_e = G // R
                    gen_R = gpix.view(args.n_eval, G, G)[:, ::stride_e, ::stride_e].cpu().numpy()
                    tru_R = test_data[:, ::stride_e, ::stride_e].numpy()
                    _, sg = get_fourier_spectrum(gen_R)
                    _, st = get_fourier_spectrum(tru_R)
                    kmax = min(R//2-1, len(sg))
                    rel = np.abs(sg[:kmax]-st[:kmax])/(np.abs(st[:kmax])+1e-30)
                    print(f"  [eval R={R} MF{n_mf}] mean_rel≤{kmax}={rel.mean():.4f} max_rel≤{kmax}={rel.max():.4f}")
                    if not args.no_wandb:
                        wandb.log({f'eval/mean_rel_R{R}_MF{n_mf}': rel.mean(),
                                   f'eval/max_rel_R{R}_MF{n_mf}': rel.max()},
                                  step=step+bi*200000)

        # Save and store generated band (use MF 2-step as default)
        save_dir = f'results/perband_mf_K{K}'
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))

        # Generate final band with 2 mean flow steps for next band's context
        net.eval()
        xC_final = gen[:, C] if len(C) > 0 else None
        embed_final = make_embed(xC_final)
        gen[:, F] = generate_band_mf(net, embed_final, extract_fn, scale,
                                      args.n_eval, nF, device, n_mf_steps=2)

    # ── Final full-resolution eval ────────────────────────────────────
    print(f"\n{'='*60}\nFINAL — multi-MF-step eval\n{'='*60}")
    for n_mf in [1, 2, 4]:
        gen2 = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']; scale2 = b['scale']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            net2 = BandMFNet(R2, b['in_ch'], b['out_ch'], args.ch).float().to(device)
            net2.load_state_dict(torch.load(os.path.join(f'results/perband_mf_K{K}', b['name'], 'model.pt'),
                                            map_location=device, weights_only=True))
            net2.eval()
            ef2 = make_embed(xC2, F_=F2, C_=C2, G_=G, R_=R2)
            xf2 = make_extract(F_=F2, C_=C2, G_=G, R_=R2)
            gen2[:, F2] = generate_band_mf(net2, ef2, xf2, scale2, args.n_eval, nF2, device, n_mf_steps=n_mf)

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
