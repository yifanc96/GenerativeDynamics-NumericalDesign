"""
Multiscale per-band MEAN FLOW for 2D Navier-Stokes vorticity fields.

Adapted from Gaussian-fields/train_multiscale_perband_meanflow.py.
Same sequential coarse-to-fine structure with per-band UNets at native resolution.
Only data loading and eval metrics differ from the GF version.

Usage:
  python train_ns_perband_meanflow.py --gpu 0
"""
import os, sys, math, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from time import time as timer
from matplotlib import pyplot as plt
import scipy.stats as stats
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb

# Import shared infrastructure from GF code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Gaussian-fields'))
from train_multiscale_interpolation import HierarchicalMasks
from train_multiscale_perband import ridge_regression, embed_mask, extract_mask


# ─── NS data loading ────────────────────────────────────────────────────────

def load_ns_data(data_loc, hi_size, train_test_split=0.9):
    data_raw, _ = torch.load(data_loc)
    Ntj, Nts, Nx, Ny = data_raw.shape
    print(f"[Data] raw: {Ntj} traj × {Nts} snapshots × {Nx}×{Ny}")
    avg_pixel_norm = torch.norm(data_raw, dim=(2, 3), p='fro').mean() / np.sqrt(Nx * Ny)
    data_raw = data_raw / avg_pixel_norm
    data = data_raw.reshape(-1, Nx, Ny)
    if hi_size != Nx:
        data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size),
                                         mode='bilinear').squeeze(1)
    print(f"[Data] {data.shape[0]} samples @ {hi_size}×{hi_size}, std={data.std():.4f}")
    n_train = int(data.shape[0] * train_test_split)
    return data[:n_train], data[n_train:]


# ─── NS eval metrics ────────────────────────────────────────────────────────

def get_energy_spectrum(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(1, 2), norm='forward')
    fourier_amp = (torch.abs(fhat)**2 * (2 * np.pi)).mean(dim=0)
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    fourier_flat = fourier_amp.numpy().flatten()
    laplace = knrm**2; laplace[0] = 1.0
    energy_flat = fourier_flat / laplace
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    enstrophy, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    enstrophy *= area_weight
    energy, _, _ = stats.binned_statistic(knrm, energy_flat, statistic='mean', bins=kbins)
    energy *= area_weight
    return kvals, enstrophy, energy


# ─── Per-band mean flow UNet (same as GF version) ───────────────────────────

class BandMFNet(nn.Module):
    def __init__(self, R, ic, oc, ch=32):
        super().__init__()
        dm = (1, 2) if R <= 8 else (1, 2, 2) if R <= 32 else (1, 2, 2, 2)
        self.net = Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                        resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                        random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                        attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)
        time_dim = ch * 4; lsd = max(ch, 16)
        sinu = RandomOrLearnedSinusoidalPosEmb(lsd, is_random=False)
        self.r_mlp = nn.Sequential(
            sinu, nn.Linear(lsd + 1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )
        nn.init.zeros_(self.r_mlp[-1].weight)
        nn.init.zeros_(self.r_mlp[-1].bias)

    def forward(self, x, s, r):
        net = self.net
        t_emb = net.time_mlp(s) + self.r_mlp(r)
        x = net.init_conv(x); r_skip = x.clone(); h = []
        for b1, b2, attn, down in net.downs:
            x = b1(x, t_emb, None); h.append(x)
            x = b2(x, t_emb, None); x = attn(x); h.append(x)
            x = down(x)
        x = net.mid_block1(x, t_emb, None); x = net.mid_attn(x); x = net.mid_block2(x, t_emb, None)
        for b1, b2, attn, up in net.ups:
            x = torch.cat((x, h.pop()), 1); x = b1(x, t_emb, None)
            x = torch.cat((x, h.pop()), 1); x = b2(x, t_emb, None); x = attn(x)
            x = up(x)
        x = torch.cat((x, r_skip), 1)
        x = net.final_res_block(x, t_emb, None)
        return net.final_conv(x)


def generate_band_mf(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_mf_steps=2):
    zt = scale * torch.randn(n_samples, nF, device=device)
    s_vals = torch.linspace(0.0, 1.0, n_mf_steps + 1)
    for j in range(n_mf_steps):
        s_j, r_j = float(s_vals[j]), float(s_vals[j + 1])
        sb = torch.full((n_samples,), s_j, device=device)
        rb = torch.full((n_samples,), r_j, device=device)
        with torch.no_grad():
            vel = extract_fn(net(embed_fn(zt), sb, rb))
        zt = zt + (r_j - s_j) * vel
    return zt


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=128)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--batch', type=int, default=200)
    p.add_argument('--batch_fine', type=int, default=20)
    p.add_argument('--steps', type=str, default='50000,50000,50000,100000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--n_eval', type=int, default=200)
    p.add_argument('--flow_ratio', type=float, default=0.5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--data_loc', type=str, default='../NSdata/data_file.pt')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.G; K = args.K
    steps_list = [int(x) for x in args.steps.split(',')]

    # ── Load NS data ──────────────────────────────────────────────────
    train_data, test_data = load_ns_data(args.data_loc, G)
    truth_np = test_data[:args.n_eval].numpy()
    kvals, enst_truth, ener_truth = get_energy_spectrum(torch.from_numpy(truth_np))

    # ── Build bands (mask decomposition) ──────────────────────────────
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

    # ── Estimate σ² per band from training data ──────────────────────
    n_ridge = min(5000, train_data.shape[0])
    est = train_data[:n_ridge].to(device).float().view(n_ridge, -1)
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

    print(f'NS Perband MEAN FLOW: G={G} K={K}')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:3d} |F|={len(b['F']):6d} σ={b['scale']:.4e} steps={ns}")

    run_name = f'NS_perband_mf_K{K}_G{G}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine ──────────────────────────────────────────
    gen = torch.zeros(args.n_eval, G*G, device=device)
    n_train = train_data.shape[0]
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']; sigma2 = b['sigma2']
        M_op = b['M_op']; intercept = b['intercept']
        n_steps = steps_list[bi]
        band_batch = args.batch_fine if R >= G else args.batch

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

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, {n_steps} steps, "
              f"σ={scale:.4e}, batch={band_batch}\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            # Sample from training data
            idx = torch.randint(0, n_train, (band_batch,))
            data_batch = train_data[idx].to(device).float().view(band_batch, -1)
            xF = data_batch[:, F]
            xC = data_batch[:, C] if len(C) > 0 else None

            z0_F = scale * torch.randn(band_batch, nF, device=device)
            target = xF - z0_F

            t1 = torch.rand(band_batch, device=device) * 0.998 + 0.001
            t2 = torch.rand(band_batch, device=device) * 0.998 + 0.001
            s = torch.minimum(t1, t2)
            r = torch.maximum(t1, t2)
            flow_mask = torch.rand(band_batch, device=device) < args.flow_ratio
            r = torch.where(flow_mask, s, r)

            zt_F = (1 - s).unsqueeze(1) * z0_F + s.unsqueeze(1) * xF
            use_jvp = (step > warmup_steps)

            embed_fn_train = make_embed(xC)

            if use_jvp:
                def fn(zt_F_, s_, r_, _embed=embed_fn_train, _extract=extract_fn, _net=net):
                    return _extract(_net(_embed(zt_F_), s_, r_))
                w, dw_ds = torch.func.jvp(fn, (zt_F, s, r),
                                           (target, torch.ones_like(s), torch.zeros_like(r)))
                dr = (r - s).unsqueeze(1)
                w_tgt = target + dr * dw_ds
                error = w - w_tgt.detach()
            else:
                w = extract_fn(net(embed_fn_train(zt_F), s, s))
                error = w - target

            loss = error.pow(2).mean() / sigma2

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            sch.step()

            if step % max(1, n_steps//20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} lr={lr:.2e} [{(timer()-t0)/60:.1f}m]")
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
                    tru_R = test_data[:args.n_eval, ::stride_e, ::stride_e].numpy()
                    from train_multiscale_interpolation import get_fourier_spectrum as gfs
                    _, sg = gfs(gen_R); _, st = gfs(tru_R)
                    kmax = min(R//2-1, len(sg))
                    rel = np.abs(sg[:kmax]-st[:kmax])/(np.abs(st[:kmax])+1e-30)
                    print(f"  [eval R={R} MF{n_mf}] mean_rel≤{kmax}={rel.mean():.4f} max_rel≤{kmax}={rel.max():.4f}")
                    if not args.no_wandb:
                        wandb.log({f'eval/mean_rel_R{R}_MF{n_mf}': rel.mean()}, step=step+bi*200000)

        save_dir = f'results/ns_perband_mf_K{K}'
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))

        net.eval()
        xC_final = gen[:, C] if len(C) > 0 else None
        gen[:, F] = generate_band_mf(net, make_embed(xC_final), extract_fn, scale,
                                      args.n_eval, nF, device, n_mf_steps=2)

    # ── Final full-resolution eval ────────────────────────────────────
    print(f"\n{'='*60}\nFINAL — NS enstrophy/energy spectrum\n{'='*60}")
    for n_mf in [1, 2, 4]:
        gen2 = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            net2 = BandMFNet(R2, b['in_ch'], b['out_ch'], args.ch).float().to(device)
            net2.load_state_dict(torch.load(os.path.join(f'results/ns_perband_mf_K{K}', b['name'], 'model.pt'),
                                            map_location=device, weights_only=True))
            net2.eval()
            ef2 = make_embed(xC2, F_=F2, C_=C2, G_=G, R_=R2)
            xf2 = make_extract(F_=F2, C_=C2, G_=G, R_=R2)
            gen2[:, F2] = generate_band_mf(net2, ef2, xf2, b['scale'], args.n_eval, nF2, device, n_mf_steps=n_mf)

        gen_sq = gen2.view(args.n_eval, G, G).cpu()
        _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)
        std_ratio = gen_sq.numpy().std() / (truth_np.std() + 1e-12)

        bands_k = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}
        bm = {}
        for bn, mask in bands_k.items():
            bm[bn] = np.mean(np.abs(enst_truth[mask] - enst_gen[mask]) / (np.abs(enst_truth[mask]) + 1e-12))

        nfe = n_mf * len(bands)
        print(f"  MF{n_mf}/band (NFE={nfe}): low={bm['low']:.4f} mid={bm['mid']:.4f} high={bm['high']:.4f} std_ratio={std_ratio:.4f}")
        if not args.no_wandb:
            wandb.log({f'final/low_MF{n_mf}': bm['low'], f'final/mid_MF{n_mf}': bm['mid'],
                       f'final/high_MF{n_mf}': bm['high'], f'final/std_MF{n_mf}': std_ratio})

    if not args.no_wandb:
        wandb.finish()
