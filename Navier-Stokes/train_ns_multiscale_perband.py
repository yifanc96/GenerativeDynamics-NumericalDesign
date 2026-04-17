"""
Multiscale per-band flow matching for 2D Navier-Stokes.

Supports mask, Haar wavelet, and standard (1-mask) flow matching.
Uses the "raw" variant: z0 ~ N(0, σ²I), target = x_F - z0, loss = MSE/σ².

128×128 resolution, K=3 (4 bands). Loads all 5 NS datasets.
Trains coarse-to-fine, each band independently.

Usage:
  # Mask multiscale (4 bands)
  python train_ns_multiscale_perband.py --decomp mask --K 3 --gpu 0

  # Haar wavelet multiscale (4 bands)
  python train_ns_multiscale_perband.py --decomp haar --K 3 --gpu 1

  # Standard flow matching (1 mask, baseline)
  python train_ns_multiscale_perband.py --decomp mask --K 0 --gpu 2

  # Compare all three on same GPU sequentially
  python train_ns_multiscale_perband.py --decomp mask --K 3 --gpu 0
  python train_ns_multiscale_perband.py --decomp haar --K 3 --gpu 1
  python train_ns_multiscale_perband.py --decomp mask --K 0 --gpu 2
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from time import time as timer
from matplotlib import pyplot as plt
import scipy.stats as stats
import pywt
import wandb

from unet import Unet


# ─── Data ──────────────────────────────────────────────────────────────────
def load_ns_data(data_locs, hi_size, train_test_split=0.9):
    avg_pixel_norm = 3.0679163932800293
    all_data = []
    for loc in data_locs:
        data_raw, _ = torch.load(loc, weights_only=False)
        Ntj, Nts, Nx, Ny = data_raw.shape
        print(f"  {loc}: {Ntj}x{Nts}x{Nx}x{Ny}")
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, Nx, Ny)
        if hi_size != Nx:
            data = nn.functional.interpolate(
                data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        all_data.append(data)
    data = torch.cat(all_data, dim=0).float()
    N = data.shape[0]; n_train = int(N * train_test_split)
    print(f"  Total: {N}, train={n_train}, test={N-n_train}, std={data.std():.4f}")
    return data[:n_train], data[n_train:]


# ─── Spectrum ──────────────────────────────────────────────────────────────
def get_energy_spectrum(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    fhat = torch.fft.fftn(data.float(), dim=(1, 2), norm='forward')
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
    area = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    enstrophy, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    energy, _, _ = stats.binned_statistic(knrm, energy_flat, statistic='mean', bins=kbins)
    return kvals, enstrophy * area, energy * area


# ─── UNet ──────────────────────────────────────────────────────────────────
def make_unet(R, ic, oc, ch=32):
    dm = (1, 2) if R <= 16 else (1, 2, 2) if R <= 64 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)


# ─── Mask bands ────────────────────────────────────────────────────────────
def build_mask_bands(G, K):
    """K=0 gives standard 1-mask FM. K>=1 gives K+1 multiscale bands."""
    if K == 0:
        F = torch.arange(G * G)
        return [dict(F=F, C=torch.empty(0, dtype=torch.long), R=G,
                     in_ch=1, out_ch=1, name='full')]

    n = int(math.log2(G))
    y, x = torch.meshgrid(torch.arange(G), torch.arange(G), indexing='ij')
    all_masks = []
    for k in range(n):
        if k == n - 1:
            mask = (y % (2**k) == 0) & (x % (2**k) == 0)
        else:
            div_cur = (y % (2**k) == 0) & (x % (2**k) == 0)
            div_coarser = (y % (2**(k+1)) == 0) & (x % (2**(k+1)) == 0)
            mask = div_cur & ~div_coarser
        all_masks.append(mask.float())
    masks = []
    if K + 1 < n:
        for k in range(K):
            masks.append(all_masks[k])
        masks.append(sum(all_masks[K:]).clamp(max=1.0))
    else:
        masks = all_masks[:K+1]

    bands = []
    for k_phase in range(K+1):
        si = K - k_phase
        F = torch.nonzero(masks[si].flatten()).flatten()
        Cl = [torch.nonzero(masks[K-j].flatten()).flatten() for j in range(k_phase)]
        C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
        R = G // (2**(K - k_phase))
        bands.append(dict(F=F, C=C, R=R, in_ch=1 if k_phase == 0 else 2, out_ch=1,
                          name=f's{k_phase}_R{R}'))
    return bands


# ─── Haar bands ────────────────────────────────────────────────────────────
def build_haar_bands(G, K):
    d = G * G
    W_np = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        delta = np.zeros((G, G)); delta.flat[i] = 1.0
        c = pywt.wavedec2(delta, 'haar', level=K, mode='periodization')
        flat = list(c[0].ravel())
        for j in range(1, len(c)):
            for sub in c[j]: flat.extend(sub.ravel())
        W_np[:, i] = flat
    W = torch.from_numpy(W_np)

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
    return bands, W


# ─── Embedding ─────────────────────────────────────────────────────────────
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


def embed_haar(zt_detail, x_C_wav, C_w, W_dev, G, R, device):
    B = zt_detail.shape[0]; ns = R*R; stride = G // R
    if len(C_w) == 0:
        return zt_detail.view(B, 1, R, R)
    LH = zt_detail[:, :ns].view(B, 1, R, R)
    HL = zt_detail[:, ns:2*ns].view(B, 1, R, R)
    HH = zt_detail[:, 2*ns:].view(B, 1, R, R)
    wc = torch.zeros(B, G*G, device=device); wc[:, C_w] = x_C_wav
    ll = (W_dev.T @ wc.T).T.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ll, LH, HL, HH], dim=1)


# ─── Ridge regression σ² ──────────────────────────────────────────────────
def estimate_sigma2(data_flat, F, C, device, n_max=10000, ridge=1e-6):
    N = min(data_flat.shape[0], n_max)
    Y = data_flat[:N, F]
    if len(C) == 0:
        return Y.var(dim=0).mean().item()
    X = data_flat[:N, C]
    Xa = torch.cat([torch.ones(N, 1, device=device), X], dim=1)
    beta = torch.linalg.solve(Xa.T @ Xa + ridge * torch.eye(Xa.shape[1], device=device), Xa.T @ Y)
    return (Y - Xa @ beta).var(dim=0).mean().item()


# ─── RK4 generate one band ────────────────────────────────────────────────
def generate_band(net, embed_fn, extract_fn, sigma2, n_samples, nF, device, n_rk4=4):
    scale = math.sqrt(sigma2)
    zt = scale * torch.randn(n_samples, nF, device=device)
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
    return zt


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--decomp', type=str, default='mask', choices=['mask', 'haar'])
    p.add_argument('--K', type=int, default=3, help='0=standard FM, >=1=multiscale')
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--num_dataset', type=int, default=5)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--steps', type=str, default='30000,30000,30000,60000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--n_eval', type=int, default=200)
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.hi_size; K = args.K
    steps_list = [int(x) for x in args.steps.split(',')]

    # Load data
    suffixes = ['', '02', '03', '04', '05']
    data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes[:args.num_dataset]]
    print(f'Loading {args.num_dataset} NS datasets at {G}x{G}:')
    train_data, test_data = load_ns_data(data_locs, G)
    N_train = train_data.shape[0]
    train_dev = train_data.to(device)

    # Truth spectrum
    kvals, enstr_truth, energy_truth = get_energy_spectrum(test_data[:args.n_eval])

    # Build bands
    W_dev = None
    if args.decomp == 'haar' and K > 0:
        bands, W_t = build_haar_bands(G, K)
        W_dev = W_t.to(device)
        train_w = (W_dev @ train_dev.view(N_train, -1).T).T  # precompute wavelet transform
    else:
        bands = build_mask_bands(G, K)
        train_w = train_dev.view(N_train, -1)

    # Estimate σ² per band
    for b in bands:
        Ft = b['F'].to(device); Ct = b['C'].to(device) if len(b['C']) > 0 else b['C']
        b['sigma2'] = estimate_sigma2(train_w, Ft, Ct, device)
        b['scale'] = math.sqrt(b['sigma2'])

    while len(steps_list) < len(bands):
        steps_list.append(steps_list[-1])

    decomp_label = f'{args.decomp}_K{K}' if K > 0 else 'standard_1mask'
    print(f'\n{decomp_label} ({len(bands)} bands):')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:3d} |F|={len(b['F']):5d} σ={b['scale']:.4e} steps={ns}")

    save_dir = f'results/ns_{decomp_label}_G{G}'
    os.makedirs(save_dir, exist_ok=True)

    # Wandb
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96',
                   name=f'NS_{decomp_label}_G{G}')
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine ───────────────────────────────────────────
    nets = []
    gen = torch.zeros(args.n_eval, G*G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; sigma2 = b['sigma2']; scale = b['scale']
        n_steps = steps_list[bi]

        net = make_unet(R, b['in_ch'], b['out_ch'], args.ch).float().to(device)
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, {n_steps} steps\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            idx = torch.randint(N_train, (args.batch,), device=device)
            batch = train_w[idx]  # (B, G*G) in pixel or wavelet domain
            xF = batch[:, F]
            xC = batch[:, C] if len(C) > 0 else None

            z0 = scale * torch.randn(args.batch, nF, device=device)
            t = torch.rand(args.batch, device=device) * 0.998 + 0.001
            a = (1-t).unsqueeze(1); bv = t.unsqueeze(1)
            zt = a * z0 + bv * xF
            target = xF - z0

            # Embed
            if args.decomp == 'mask' or K == 0:
                inp = embed_mask(zt, xC, F, C, G, R, device)
                pred = net(inp, t, classes=None)
                pred_F = extract_mask(pred, F, G, R, device) if len(C) > 0 else pred.view(args.batch, -1)
            else:
                inp = embed_haar(zt, xC, C, W_dev, G, R, device) if len(C) > 0 else zt.view(args.batch, 1, R, R)
                pred = net(inp, t, classes=None)
                pred_F = pred.view(args.batch, -1)[:, :nF]

            loss = (pred_F - target).pow(2).mean() / sigma2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e4)
            opt.step(); sch.step()

            if step % max(1, n_steps//20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} lr={lr:.2e} [{(timer()-t0)/60:.1f}m]")
                if not args.no_wandb:
                    wandb.log({f'loss/{b["name"]}': loss.item()})

            # Eval
            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                xC_eval = gen[:, C] if len(C) > 0 else None
                if args.decomp == 'mask' or K == 0:
                    ef = lambda z, xC_=xC_eval, F_=F, C_=C, R_=R: embed_mask(z, xC_, F_, C_, G, R_, device)
                    xf = (lambda p, F_=F, R_=R: extract_mask(p, F_, G, R_, device)) if len(C) > 0 else (lambda p: p.view(args.n_eval, -1))
                else:
                    ef = (lambda z, xC_=xC_eval, C_=C, R_=R: embed_haar(z, xC_, C_, W_dev, G, R_, device)) if len(C) > 0 else (lambda z, R_=R: z.view(args.n_eval, 1, R_, R_))
                    xf = lambda p, nF_=nF: p.view(args.n_eval, -1)[:, :nF_]

                gen_F = generate_band(net, ef, xf, sigma2, args.n_eval, nF, device)
                gen_temp = gen.clone(); gen_temp[:, F] = gen_F

                # Spectrum at current resolution
                if args.decomp == 'haar' and K > 0:
                    gpix = (W_dev.T @ gen_temp.T).T
                else:
                    gpix = gen_temp
                stride_e = G // R
                gen_R = gpix.view(args.n_eval, G, G)[:, ::stride_e, ::stride_e].cpu()
                tru_R = test_data[:args.n_eval, ::stride_e, ::stride_e]
                kv_R, _, eg_R = get_energy_spectrum(gen_R)
                _, _, et_R = get_energy_spectrum(tru_R)
                kmax = min(R//2-1, len(eg_R))
                if kmax > 0:
                    rel = np.abs(eg_R[:kmax]-et_R[:kmax])/(np.abs(et_R[:kmax])+1e-30)
                    print(f"  [eval R={R}] energy mean_rel≤{kmax}={rel.mean():.4f} max={rel.max():.4f}")
                    if not args.no_wandb:
                        wandb.log({f'eval/energy_rel_R{R}': rel.mean()})

        # Save and store
        gen[:, F] = gen_F
        nets.append(net)
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))
        print(f"  Saved to {bd}")

    # ── Final eval at multiple RK4 steps ───────────────────────────────
    print(f"\n{'='*60}\nFINAL — {decomp_label}\n{'='*60}")
    for n_rk4 in [2, 4, 8, 16]:
        gen2 = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF2 = len(b['F']); R2 = b['R']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            if args.decomp == 'mask' or K == 0:
                ef2 = lambda z, xC_=xC2, F_=F2, C_=C2, R_=R2: embed_mask(z, xC_, F_, C_, G, R_, device)
                xf2 = (lambda p, F_=F2, R_=R2: extract_mask(p, F_, G, R_, device)) if len(C2) > 0 else (lambda p: p.view(args.n_eval, -1))
            else:
                ef2 = (lambda z, xC_=xC2, C_=C2, R_=R2: embed_haar(z, xC_, C_, W_dev, G, R_, device)) if len(C2) > 0 else (lambda z, R_=R2: z.view(args.n_eval, 1, R_, R_))
                xf2 = lambda p, nF_=nF2: p.view(args.n_eval, -1)[:, :nF_]
            gen2[:, F2] = generate_band(nets[bi], ef2, xf2, b['sigma2'], args.n_eval, nF2, device, n_rk4)

        if args.decomp == 'haar' and K > 0:
            gpix2 = (W_dev.T @ gen2.T).T
        else:
            gpix2 = gen2
        gn2 = gpix2.view(args.n_eval, G, G).cpu()
        kv, _, ener_gen = get_energy_spectrum(gn2)
        kmax = min(60, len(kv))
        rel = np.abs(ener_gen[:kmax]-energy_truth[:kmax])/(np.abs(energy_truth[:kmax])+1e-30)
        nb1 = np.sum(rel < 1)
        print(f"  RK4={n_rk4:2d}: energy mean_rel≤{kmax}={rel.mean():.4f} max={rel.max():.4f} bins<1={nb1}/{kmax}")

    # Spectrum plot
    gen_final = gen2  # last RK4 count
    gn_final = gpix2.view(args.n_eval, G, G).cpu()
    _, enstr_gen, ener_gen = get_energy_spectrum(gn_final)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].loglog(kvals, enstr_truth, 'k-', lw=2, label='truth')
    axes[0].loglog(kvals, enstr_gen, 'r--', label='generated')
    axes[0].set_title('Enstrophy'); axes[0].legend(); axes[0].set_xlabel('k')
    axes[1].loglog(kvals, energy_truth, 'k-', lw=2, label='truth')
    axes[1].loglog(kvals, ener_gen, 'r--', label='generated')
    axes[1].set_title('Energy'); axes[1].legend(); axes[1].set_xlabel('k')
    # Sample
    truth_img = test_data[0].numpy()
    gen_img = gn_final[0].numpy()
    vmax = max(abs(truth_img.min()), abs(truth_img.max()))
    combined = np.concatenate([truth_img, gen_img], axis=1)
    axes[2].imshow(combined, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Truth | Generated'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectrum.png'), dpi=150)
    if not args.no_wandb:
        wandb.log({'final_spectrum': wandb.Image(fig)})
    plt.close()

    print(f"\nTotal time: {(timer()-t0)/60:.1f} min")
    if not args.no_wandb:
        wandb.finish()
