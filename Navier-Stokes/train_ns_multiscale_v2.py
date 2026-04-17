"""
Multiscale per-band flow matching for NS with:
  - 1-channel mask (F+C in single image, no redundant coarse channel)
  - t-dependent input/output normalization (EDM-style preconditioning)
  - Raw variant: z0 ~ N(0, σ²I), target = x_F - z0

Normalization:
  c_in(t)  = 1/√((1-t)²σ² + t²·v_F)   — makes network input O(1) at all t
  c_out    = 1/√(v_F + σ²)             — makes target O(1)
  network sees: zt·c_in(t) as F pixels, x_C raw as C pixels
  network predicts: (x_F - z0)·c_out
  velocity at inference: network_output / c_out

Usage:
  python train_ns_multiscale_v2.py --K 3 --gpu 0        # multiscale 4 bands
  python train_ns_multiscale_v2.py --K 0 --gpu 1        # standard 1-mask baseline
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
from time import time as timer
from matplotlib import pyplot as plt
import scipy.stats as stats
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
    if K == 0:
        return [dict(F=torch.arange(G*G), C=torch.empty(0, dtype=torch.long),
                     R=G, in_ch=1, out_ch=1, name='full')]
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
        for k in range(K): masks.append(all_masks[k])
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
        # 1-channel for ALL bands (F+C in single image)
        bands.append(dict(F=F, C=C, R=R, in_ch=1, out_ch=1, name=f's{k_phase}_R{R}'))
    return bands


# ─── 1-channel embedding ──────────────────────────────────────────────────
def embed_1ch(zt_F, x_C, F, C, G, R, device):
    """Single-channel R×R: F pixels from zt_F, C pixels from x_C."""
    B = zt_F.shape[0]; stride = G // R
    full = torch.zeros(B, G*G, device=device)
    full[:, F] = zt_F
    if len(C) > 0:
        full[:, C] = x_C
    return full.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()


def extract_F(pred, F, G, R, device):
    B = pred.shape[0]; stride = G // R
    if stride == 1:
        return pred.view(B, -1)[:, F]
    pf = torch.zeros(B, G, G, device=device)
    pf[:, ::stride, ::stride] = pred.view(B, R, R)
    return pf.view(B, -1)[:, F]


# ─── σ² and v_F estimation ────────────────────────────────────────────────
def estimate_band_stats(data_flat, F, C, device, n_max=10000, ridge=1e-6):
    """Returns σ² (conditional variance) and v_F (marginal variance)."""
    N = min(data_flat.shape[0], n_max)
    Y = data_flat[:N, F]
    v_F = Y.var(dim=0).mean().item()  # marginal variance
    if len(C) == 0:
        return v_F, v_F  # σ² = v_F when no conditioning
    X = data_flat[:N, C]
    Xa = torch.cat([torch.ones(N, 1, device=device), X], dim=1)
    beta = torch.linalg.solve(Xa.T @ Xa + ridge * torch.eye(Xa.shape[1], device=device), Xa.T @ Y)
    sigma2 = (Y - Xa @ beta).var(dim=0).mean().item()
    return sigma2, v_F


# ─── Preconditioning functions ─────────────────────────────────────────────
def c_in(t, sigma2, v_F):
    """Input normalization: 1/√(Var(zt))."""
    return 1.0 / torch.sqrt((1-t)**2 * sigma2 + t**2 * v_F).clamp(min=1e-8)


def c_out_val(sigma2, v_F):
    """Output normalization: 1/√(Var(target))."""
    return 1.0 / math.sqrt(v_F + sigma2)


# ─── RK4 ──────────────────────────────────────────────────────────────────
def generate_band(net, F, C, G, R, sigma2, v_F, n_samples, device, x_C=None, n_rk4=4, no_tnorm=False):
    nF = len(F); scale = math.sqrt(sigma2)
    c_o = c_out_val(sigma2, v_F)
    zt = scale * torch.randn(n_samples, nF, device=device)  # z0 ~ N(0, σ²I)

    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    for i in range(len(nodes)-1):
        sv = float(nodes[i]); ds = float(nodes[i+1]-nodes[i])

        def vel(z, tv):
            if no_tnorm:
                inp = embed_1ch(z, x_C, F, C, G, R, device)
                tt = torch.full((n_samples,), tv, device=device)
                with torch.no_grad(): pred = net(inp, tt, classes=None)
                return extract_F(pred, F, G, R, device)  # raw velocity
            else:
                tv_t = torch.tensor(tv)
                cin = float(c_in(tv_t, sigma2, v_F))
                inp = embed_1ch(z * cin, x_C, F, C, G, R, device)
                tt = torch.full((n_samples,), tv, device=device)
                with torch.no_grad(): pred = net(inp, tt, classes=None)
                pred_F = extract_F(pred, F, G, R, device)
                return pred_F / c_o  # undo output normalization → raw velocity

        k1 = vel(zt, sv)
        k2 = vel(zt + .5*ds*k1, sv + .5*ds)
        k3 = vel(zt + .5*ds*k2, sv + .5*ds)
        k4 = vel(zt + ds*k3, sv + ds)
        zt = zt + (ds/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3)
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
    p.add_argument('--no_tnorm', action='store_true', help='Disable t-dependent normalization')
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
    train_dev = train_data.to(device).view(N_train, -1)

    kvals, enstr_truth, energy_truth = get_energy_spectrum(test_data[:args.n_eval])

    # Build bands
    bands = build_mask_bands(G, K)
    for b in bands:
        Ft = b['F'].to(device); Ct = b['C'].to(device) if len(b['C']) > 0 else b['C']
        b['sigma2'], b['v_F'] = estimate_band_stats(train_dev, Ft, Ct, device)
        b['c_out'] = c_out_val(b['sigma2'], b['v_F'])
    while len(steps_list) < len(bands):
        steps_list.append(steps_list[-1])

    label = f'mask_K{K}' if K > 0 else 'standard'
    print(f'\n{label} ({len(bands)} bands):')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:3d} |F|={len(b['F']):5d} σ²={b['sigma2']:.3e} v_F={b['v_F']:.3e} c_out={b['c_out']:.3e} steps={ns}")

    save_dir = f'results/ns_v2_{label}_G{G}'
    os.makedirs(save_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=f'NS_v2_{label}_G{G}')
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine ───────────────────────────────────────────
    nets = []
    gen = torch.zeros(args.n_eval, G*G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']
        sigma2 = b['sigma2']; v_F = b['v_F']; c_o = b['c_out']
        scale = math.sqrt(sigma2)
        n_steps = steps_list[bi]

        net = make_unet(R, 1, 1, args.ch).float().to(device)  # 1ch input, 1ch output
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, 1ch, {n_steps} steps\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            idx = torch.randint(N_train, (args.batch,), device=device)
            batch = train_dev[idx]
            xF = batch[:, F]
            xC = batch[:, C] if len(C) > 0 else None

            z0 = scale * torch.randn(args.batch, nF, device=device)
            t = torch.rand(args.batch, device=device) * 0.998 + 0.001
            a = (1-t).unsqueeze(1); bv = t.unsqueeze(1)
            zt = a * z0 + bv * xF
            target = xF - z0

            if args.no_tnorm:
                # No t-norm: raw zt, loss reweighted by 1/σ²
                inp = embed_1ch(zt, xC, F, C, G, R, device)
                pred = net(inp, t, classes=None)
                pred_F = extract_F(pred, F, G, R, device)
                loss = (pred_F - target).pow(2).mean() / sigma2
            else:
                # t-dependent normalization
                cin = c_in(t, sigma2, v_F).unsqueeze(1)  # (B, 1)
                zt_norm = zt * cin  # F pixels normalized, O(1)
                target_norm = target * c_o  # O(1)
                inp = embed_1ch(zt_norm, xC, F, C, G, R, device)
                pred = net(inp, t, classes=None)
                pred_F = extract_F(pred, F, G, R, device)
                loss = (pred_F - target_norm).pow(2).mean()
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
                gen_F = generate_band(net, F, C, G, R, sigma2, v_F,
                                       args.n_eval, device, xC_eval, no_tnorm=args.no_tnorm)
                gen_temp = gen.clone(); gen_temp[:, F] = gen_F

                stride_e = G // R
                gen_R = gen_temp.view(args.n_eval, G, G)[:, ::stride_e, ::stride_e].cpu()
                tru_R = test_data[:args.n_eval, ::stride_e, ::stride_e]
                kv_R, _, eg_R = get_energy_spectrum(gen_R)
                _, _, et_R = get_energy_spectrum(tru_R)
                kmax = min(R//2-1, len(eg_R))
                if kmax > 0:
                    rel = np.abs(eg_R[:kmax]-et_R[:kmax])/(np.abs(et_R[:kmax])+1e-30)
                    print(f"  [eval R={R}] energy mean_rel≤{kmax}={rel.mean():.4f} max={rel.max():.4f}")
                    if not args.no_wandb:
                        wandb.log({f'eval/energy_rel_R{R}': rel.mean()})

        gen[:, F] = gen_F
        nets.append(net)
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))
        print(f"  Saved to {bd}")

    # ── Final eval ─────────────────────────────────────────────────────
    print(f"\n{'='*60}\nFINAL — {label}\n{'='*60}")
    for n_rk4 in [2, 4, 8, 16]:
        gen2 = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            xC2 = gen2[:, C2] if len(C2) > 0 else None
            gen2[:, F2] = generate_band(nets[bi], F2, C2, G, b['R'],
                                         b['sigma2'], b['v_F'],
                                         args.n_eval, device, xC2, n_rk4, no_tnorm=args.no_tnorm)
        gn2 = gen2.view(args.n_eval, G, G).cpu()
        kv, _, ener_gen = get_energy_spectrum(gn2)
        kmax = min(60, len(kv))
        rel = np.abs(ener_gen[:kmax]-energy_truth[:kmax])/(np.abs(energy_truth[:kmax])+1e-30)
        nb1 = np.sum(rel < 1)
        print(f"  RK4={n_rk4:2d}: energy mean_rel≤{kmax}={rel.mean():.4f} max={rel.max():.4f} bins<1={nb1}/{kmax}")

    # Spectrum plot
    _, enstr_gen, ener_gen = get_energy_spectrum(gn2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(kvals, energy_truth, 'k-', lw=2, label='truth')
    axes[0].loglog(kvals, ener_gen, 'r--', label='generated')
    axes[0].set_title('Energy'); axes[0].legend(); axes[0].set_xlabel('k')
    axes[1].loglog(kvals, enstr_truth, 'k-', lw=2, label='truth')
    axes[1].loglog(kvals, enstr_gen, 'r--', label='generated')
    axes[1].set_title('Enstrophy'); axes[1].legend(); axes[1].set_xlabel('k')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectrum.png'), dpi=150)
    if not args.no_wandb:
        wandb.log({'final_spectrum': wandb.Image(fig)})
    plt.close()

    print(f"\nTotal time: {(timer()-t0)/60:.1f} min")
    if not args.no_wandb:
        wandb.finish()
