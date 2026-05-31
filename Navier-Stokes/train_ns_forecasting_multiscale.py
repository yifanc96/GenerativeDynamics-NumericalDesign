"""
Multiscale flow matching for NS forecasting (conditional generation).

Given x_lo (state at time t), generate x_hi (state at time t+τ).
Predicts innovation r = x_hi - x_lo via multiscale per-band flow matching.
Conditioning: x_lo as second channel at each band's resolution.

Usage:
  python train_ns_forecasting_multiscale.py --K 3 --gpu 0          # multiscale
  python train_ns_forecasting_multiscale.py --K 0 --gpu 1          # standard baseline
  python train_ns_forecasting_multiscale.py --K 3 --no_tnorm --gpu 2  # without t-norm
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


# ─── Data: forecasting pairs ──────────────────────────────────────────────
def load_forecasting_data(data_locs, hi_size, time_lag=2, train_test_split=0.9):
    """Load (x_lo, x_hi) pairs. x_lo = state at t, x_hi = state at t+time_lag."""
    avg_pixel_norm = 3.0679163932800293
    all_lo, all_hi = [], []
    for loc in data_locs:
        data_raw, _ = torch.load(loc, weights_only=False)
        Ntj, Nts, Nx, Ny = data_raw.shape
        print(f"  {loc}: {Ntj}x{Nts}x{Nx}x{Ny}")
        data_raw = data_raw / avg_pixel_norm
        if time_lag > 0:
            lo = data_raw[:, :-time_lag, :, :]  # (Ntj, Nts-lag, Nx, Ny)
            hi = data_raw[:, time_lag:, :, :]
        else:
            lo = hi = data_raw
        lo = lo.reshape(-1, Nx, Ny)
        hi = hi.reshape(-1, Nx, Ny)
        if hi_size != Nx:
            lo = nn.functional.interpolate(lo.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
            hi = nn.functional.interpolate(hi.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        all_lo.append(lo); all_hi.append(hi)
    lo = torch.cat(all_lo, dim=0).float()
    hi = torch.cat(all_hi, dim=0).float()
    N = lo.shape[0]; n_train = int(N * train_test_split)
    print(f"  Total pairs: {N}, train={n_train}, test={N-n_train}")
    innov = hi - lo
    print(f"  Innovation std={innov.std():.4f}, x_lo std={lo.std():.4f}")
    return lo[:n_train], hi[:n_train], lo[n_train:], hi[n_train:]


# ─── Spectrum ──────────────────────────────────────────────────────────────
def get_energy_spectrum(data):
    if isinstance(data, np.ndarray): data = torch.from_numpy(data)
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


# ─── UNet, mask bands, embedding (same as unconditional) ──────────────────
def make_unet(R, ic, oc, ch=32):
    dm = (1, 2) if R <= 16 else (1, 2, 2) if R <= 64 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)


def build_mask_bands(G, K):
    if K == 0:
        return [dict(F=torch.arange(G*G), C=torch.empty(0, dtype=torch.long),
                     R=G, in_ch=2, out_ch=1, name='full')]  # 2ch: innovation + x_lo
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
        # 2ch: [innovation_state, x_lo_at_R]
        bands.append(dict(F=F, C=C, R=R, in_ch=2, out_ch=1, name=f's{k_phase}_R{R}'))
    return bands


def embed_forecast(zt_F, x_C_innov, x_lo, F, C, G, R, device):
    """2-channel R×R: [innovation state (zt_F + coarse innovation), x_lo subsampled]."""
    B = zt_F.shape[0]; stride = G // R
    # Ch1: innovation state (F from zt, C from coarse innovation)
    full = torch.zeros(B, G*G, device=device)
    full[:, F] = zt_F
    if len(C) > 0 and x_C_innov is not None:
        full[:, C] = x_C_innov
    ch1 = full.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    # Ch2: x_lo subsampled to R×R
    ch2 = x_lo.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ch1, ch2], dim=1)


def extract_F(pred, F, G, R, device):
    B = pred.shape[0]; stride = G // R
    if stride == 1: return pred.view(B, -1)[:, F]
    pf = torch.zeros(B, G, G, device=device)
    pf[:, ::stride, ::stride] = pred.view(B, R, R)
    return pf.view(B, -1)[:, F]


# ─── Band stats on innovation ─────────────────────────────────────────────
def estimate_band_stats(innov_flat, F, C, device, n_max=10000, ridge=1e-6):
    N = min(innov_flat.shape[0], n_max)
    Y = innov_flat[:N, F]
    v_F = Y.var(dim=0).mean().item()
    if len(C) == 0:
        return v_F, v_F
    X = innov_flat[:N, C]
    Xa = torch.cat([torch.ones(N, 1, device=device), X], dim=1)
    beta = torch.linalg.solve(Xa.T @ Xa + ridge * torch.eye(Xa.shape[1], device=device), Xa.T @ Y)
    sigma2 = (Y - Xa @ beta).var(dim=0).mean().item()
    return sigma2, v_F


# ─── Preconditioning ──────────────────────────────────────────────────────
def c_in_fn(t, sigma2, v_F):
    return 1.0 / torch.sqrt((1-t)**2 * sigma2 + t**2 * v_F).clamp(min=1e-8)

def c_out_val(sigma2, v_F):
    return 1.0 / math.sqrt(v_F + sigma2)


# ─── Generate one band ────────────────────────────────────────────────────
def generate_band(net, F, C, G, R, sigma2, v_F, n_samples, device,
                  x_C_innov=None, x_lo_flat=None, n_rk4=4, no_tnorm=False):
    nF = len(F); scale = math.sqrt(sigma2)
    c_o = c_out_val(sigma2, v_F)
    zt = scale * torch.randn(n_samples, nF, device=device)

    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    for i in range(len(nodes)-1):
        sv = float(nodes[i]); ds = float(nodes[i+1]-nodes[i])
        def vel(z, tv):
            if no_tnorm:
                inp = embed_forecast(z, x_C_innov, x_lo_flat, F, C, G, R, device)
            else:
                cin = float(c_in_fn(torch.tensor(tv), sigma2, v_F))
                inp = embed_forecast(z * cin, x_C_innov, x_lo_flat, F, C, G, R, device)
            tt = torch.full((n_samples,), tv, device=device)
            with torch.no_grad(): pred = net(inp, tt, classes=None)
            pred_F = extract_F(pred, F, G, R, device)
            if no_tnorm:
                return pred_F
            else:
                return pred_F / c_o
        k1=vel(zt,sv); k2=vel(zt+.5*ds*k1,sv+.5*ds)
        k3=vel(zt+.5*ds*k2,sv+.5*ds); k4=vel(zt+ds*k3,sv+ds)
        zt = zt + (ds/6)*(k1+2*k2+2*k3+k4)
    return zt


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--time_lag', type=int, default=2)
    p.add_argument('--num_dataset', type=int, default=5)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--steps', type=str, default='30000,30000,30000,60000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--n_eval', type=int, default=200)
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--no_tnorm', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.hi_size; K = args.K
    steps_list = [int(x) for x in args.steps.split(',')]

    # Load forecasting pairs
    suffixes = ['', '02', '03', '04', '05']
    data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes[:args.num_dataset]]
    print(f'Loading forecasting data (lag={args.time_lag}):')
    lo_train, hi_train, lo_test, hi_test = load_forecasting_data(data_locs, G, args.time_lag)
    N_train = lo_train.shape[0]

    # Move to device
    lo_train_dev = lo_train.to(device).view(N_train, -1)
    hi_train_dev = hi_train.to(device).view(N_train, -1)
    innov_train_dev = hi_train_dev - lo_train_dev  # innovation

    # Truth spectra for x_hi
    kvals, enstr_truth, energy_truth = get_energy_spectrum(hi_test[:args.n_eval])

    # Build bands on innovation
    bands = build_mask_bands(G, K)
    for b in bands:
        Ft = b['F'].to(device); Ct = b['C'].to(device) if len(b['C']) > 0 else b['C']
        b['sigma2'], b['v_F'] = estimate_band_stats(innov_train_dev, Ft, Ct, device)
    while len(steps_list) < len(bands):
        steps_list.append(steps_list[-1])

    label = f'forecast_mask_K{K}_lag{args.time_lag}'
    tnorm_str = '' if not args.no_tnorm else '_notnorm'
    print(f'\n{label}{tnorm_str} ({len(bands)} bands):')
    for b, ns in zip(bands, steps_list):
        print(f"  {b['name']:12s} R={b['R']:3d} |F|={len(b['F']):5d} σ²={b['sigma2']:.3e} v_F={b['v_F']:.3e} steps={ns}")

    save_dir = f'results/ns_{label}{tnorm_str}_G{G}'
    os.makedirs(save_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=f'NS_{label}{tnorm_str}')
        wandb.config.update(vars(args))

    # ── Train coarse-to-fine ───────────────────────────────────────────
    nets = []
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']
        sigma2 = b['sigma2']; v_F = b['v_F']
        scale = math.sqrt(sigma2); c_o = c_out_val(sigma2, v_F)
        n_steps = steps_list[bi]

        net = make_unet(R, 2, 1, args.ch).float().to(device)  # 2ch: innovation + conditioning
        npar = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        print(f"\n{'='*60}\nBand {bi}: {b['name']} — R={R}, {npar:,} params, {n_steps} steps\n{'='*60}")

        for step in range(1, n_steps+1):
            net.train()
            idx = torch.randint(N_train, (args.batch,), device=device)
            innov_batch = innov_train_dev[idx]  # (B, G*G) innovation
            lo_batch = lo_train_dev[idx]         # (B, G*G) conditioning

            rF = innov_batch[:, F]               # innovation at F pixels
            rC = innov_batch[:, C] if len(C) > 0 else None  # coarse innovation

            z0 = scale * torch.randn(args.batch, nF, device=device)
            t = torch.rand(args.batch, device=device) * 0.998 + 0.001
            a = (1-t).unsqueeze(1); bv = t.unsqueeze(1)
            zt = a * z0 + bv * rF
            target = rF - z0

            if args.no_tnorm:
                inp = embed_forecast(zt, rC, lo_batch, F, C, G, R, device)
                pred = net(inp, t, classes=None)
                pred_F = extract_F(pred, F, G, R, device)
                loss = (pred_F - target).pow(2).mean() / sigma2
            else:
                cin = c_in_fn(t, sigma2, v_F).unsqueeze(1)
                inp = embed_forecast(zt * cin, rC, lo_batch, F, C, G, R, device)
                pred = net(inp, t, classes=None)
                pred_F = extract_F(pred, F, G, R, device)
                loss = (pred_F - target * c_o).pow(2).mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e4)
            opt.step(); sch.step()

            if step % max(1, n_steps//20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} lr={lr:.2e} [{(timer()-t0)/60:.1f}m]")
                if not args.no_wandb:
                    wandb.log({f'loss/{b["name"]}': loss.item()})

            # Eval: generate forecast for test samples
            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                lo_eval = lo_test[:args.n_eval].to(device).view(args.n_eval, -1)
                gen_innov = torch.zeros(args.n_eval, G*G, device=device)

                # Regenerate coarser bands
                for bj in range(bi):
                    Fj = bands[bj]['F'].to(device); Cj = bands[bj]['C'].to(device) if len(bands[bj]['C']) > 0 else bands[bj]['C']
                    xCj = gen_innov[:, Cj] if len(Cj) > 0 else None
                    gen_innov[:, Fj] = generate_band(
                        nets[bj], Fj, Cj, G, bands[bj]['R'], bands[bj]['sigma2'], bands[bj]['v_F'],
                        args.n_eval, device, xCj, lo_eval, n_rk4=4, no_tnorm=args.no_tnorm)

                # Generate current band
                xC_eval = gen_innov[:, C] if len(C) > 0 else None
                gen_innov[:, F] = generate_band(
                    net, F, C, G, R, sigma2, v_F,
                    args.n_eval, device, xC_eval, lo_eval, n_rk4=4, no_tnorm=args.no_tnorm)

                # x_hi = x_lo + innovation
                x_hi_gen = (lo_eval + gen_innov).view(args.n_eval, G, G).cpu()
                _, _, eg = get_energy_spectrum(x_hi_gen)
                kmax = min(60, len(eg))
                rel = np.abs(eg[:kmax]-energy_truth[:kmax])/(np.abs(energy_truth[:kmax])+1e-30)
                # Also compute forecast error (MSE of x_hi vs truth)
                hi_truth = hi_test[:args.n_eval]
                forecast_mse = (x_hi_gen - hi_truth).pow(2).mean().item()
                print(f"  [eval] energy mean_rel={rel.mean():.4f} max={rel.max():.4f} forecast_mse={forecast_mse:.4f}")
                if not args.no_wandb:
                    wandb.log({'eval/energy_rel': rel.mean(), 'eval/forecast_mse': forecast_mse})

        nets.append(net)
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))

    # ── Final eval at multiple RK4 ────────────────────────────────────
    print(f"\n{'='*60}\nFINAL\n{'='*60}")
    lo_eval = lo_test[:args.n_eval].to(device).view(args.n_eval, -1)
    hi_truth = hi_test[:args.n_eval]

    for n_rk4 in [2, 4, 8, 16]:
        gen_innov = torch.zeros(args.n_eval, G*G, device=device)
        for bi, b in enumerate(bands):
            F2 = b['F'].to(device); C2 = b['C'].to(device) if len(b['C']) > 0 else b['C']
            xC2 = gen_innov[:, C2] if len(C2) > 0 else None
            gen_innov[:, F2] = generate_band(
                nets[bi], F2, C2, G, b['R'], b['sigma2'], b['v_F'],
                args.n_eval, device, xC2, lo_eval, n_rk4, no_tnorm=args.no_tnorm)

        x_hi_gen = (lo_eval + gen_innov).view(args.n_eval, G, G).cpu()
        _, _, eg = get_energy_spectrum(x_hi_gen)
        kmax = min(60, len(eg))
        rel = np.abs(eg[:kmax]-energy_truth[:kmax])/(np.abs(energy_truth[:kmax])+1e-30)
        forecast_mse = (x_hi_gen - hi_truth).pow(2).mean().item()
        nb1 = np.sum(rel < 1)
        print(f"  RK4={n_rk4:2d}: energy mean_rel={rel.mean():.4f} max={rel.max():.4f} mse={forecast_mse:.4f} bins<1={nb1}/{kmax}")

    # Spectrum plot
    _, enstr_gen, ener_gen = get_energy_spectrum(x_hi_gen)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].loglog(kvals, energy_truth, 'k-', lw=2, label='truth')
    axes[0].loglog(kvals, ener_gen, 'r--', label='forecast')
    axes[0].set_title('Energy'); axes[0].legend(); axes[0].set_xlabel('k')
    axes[1].loglog(kvals, enstr_truth, 'k-', lw=2, label='truth')
    axes[1].loglog(kvals, enstr_gen, 'r--', label='forecast')
    axes[1].set_title('Enstrophy'); axes[1].legend(); axes[1].set_xlabel('k')
    # Sample: truth vs forecast
    vmax = max(abs(hi_truth[0].min()), abs(hi_truth[0].max()))
    combined = torch.cat([hi_truth[0], x_hi_gen[0]], dim=1).numpy()
    axes[2].imshow(combined, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Truth | Forecast'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forecast_spectrum.png'), dpi=150)
    if not args.no_wandb:
        wandb.log({'forecast_spectrum': wandb.Image(fig)})
    plt.close()

    print(f"\nTotal time: {(timer()-t0)/60:.1f} min")
    if not args.no_wandb:
        wandb.finish()
