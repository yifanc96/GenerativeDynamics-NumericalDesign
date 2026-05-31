"""
Multiscale mask flow matching for 2D Navier-Stokes.

Per-band independent training, coarse-to-fine.
128×128 resolution, K=3 (4 scales: 16→32→64→128).
Loads all 5 NS datasets.

Key design (from Gaussian field experiments):
  - z0 ~ N(0, σ²I), zt = (1-t)*z0 + t*x_F, target = x_F - z0
  - Normalize F pixels by 1/σ so network sees O(1); keep coarse context raw
  - No centering (network learns conditional mean from context)
  - Scale-appropriate UNet per band
  - Train and eval use identical embedding
"""
import os, sys, math, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from time import time as timer
import scipy.stats as stats
import wandb

from unet import Unet


# ─── Data loading (from existing NS code) ──────────────────────────────────
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
    data = torch.cat(all_data, dim=0).float()  # (N, H, W)
    N = data.shape[0]
    n_train = int(N * train_test_split)
    print(f"  Total: {N} samples, train={n_train}, test={N-n_train}, std={data.std():.4f}")
    return data[:n_train], data[n_train:]


# ─── Energy spectrum (from existing NS code) ───────────────────────────────
def get_energy_spectrum(data):
    """data: (N, H, W) tensor or numpy."""
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


# ─── Hierarchical masks ───────────────────────────────────────────────────
def build_masks(G, K):
    """Build K+1 mask bands. Returns list of (F, C, R) tuples."""
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

    # Aggregate into K+1 bands
    masks = []
    if K + 1 < n:
        for k in range(K):
            masks.append(all_masks[k])
        masks.append(sum(all_masks[K:]).clamp(max=1.0))
    else:
        masks = all_masks[:K+1]

    # Build F, C indices for each phase
    bands = []
    for k_phase in range(K+1):
        scale_idx = K - k_phase
        F = torch.nonzero(masks[scale_idx].flatten()).flatten()
        C_list = [torch.nonzero(masks[K-j].flatten()).flatten() for j in range(k_phase)]
        C = torch.cat(C_list) if C_list else torch.empty(0, dtype=torch.long)
        R = G // (2**(K - k_phase))
        bands.append(dict(F=F, C=C, R=R, name=f's{k_phase}_R{R}'))
    return bands


# ─── Estimate σ² per band via ridge regression ────────────────────────────
def estimate_sigma2_and_regression(train_data, F, C, n_samples=10000, ridge=1e-6):
    """Returns σ², M_op, intercept. train_data: (N, H, W)."""
    idx = torch.randperm(train_data.shape[0])[:n_samples]
    data_flat = train_data[idx].view(n_samples, -1)
    Y = data_flat[:, F]
    if len(C) == 0:
        return Y.var(dim=0).mean().item(), None, None
    X = data_flat[:, C]
    X_aug = torch.cat([torch.ones(n_samples, 1), X], dim=1)
    XtX = X_aug.T @ X_aug + ridge * torch.eye(X_aug.shape[1])
    beta = torch.linalg.solve(XtX, X_aug.T @ Y)  # (|C|+1, |F|)
    sigma2 = (Y - X_aug @ beta).var(dim=0).mean().item()
    M_op = beta[1:].T       # (|F|, |C|)
    intercept = beta[0]      # (|F|,)
    return sigma2, M_op, intercept


# ─── Embedding (SAME for train and eval) ──────────────────────────────────
def embed(zt_F, x_C, F, C, G, R, device):
    """2-channel R×R image: [state (F+C), context (C only)]."""
    B = zt_F.shape[0]
    if len(C) == 0:
        return zt_F.view(B, 1, R, R)
    stride = G // R
    full = torch.zeros(B, G*G, device=device)
    full[:, F] = zt_F
    full[:, C] = x_C
    ch1 = full.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    ctx = torch.zeros(B, G*G, device=device)
    ctx[:, C] = x_C
    ch2 = ctx.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ch1, ch2], dim=1)


def extract(pred, F, G, R, device):
    """Extract F pixel predictions from R×R output."""
    B = pred.shape[0]; stride = G // R
    if stride == 1:
        return pred.view(B, -1)[:, F]
    pred_full = torch.zeros(B, G, G, device=device)
    pred_full[:, ::stride, ::stride] = pred.view(B, R, R)
    return pred_full.view(B, -1)[:, F]


def make_unet(R, in_ch, out_ch, base_ch=32):
    dm = (1, 2) if R <= 16 else (1, 2, 2) if R <= 64 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=in_ch, out_channels=out_ch,
                dim=base_ch, dim_mults=dm, resnet_block_groups=min(8, base_ch),
                learned_sinusoidal_cond=True, random_fourier_features=False,
                learned_sinusoidal_dim=max(base_ch, 16),
                attn_dim_head=max(base_ch, 16), attn_heads=4, use_classes=False)


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=3, help='Number of doubling steps (K+1 bands)')
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--num_dataset', type=int, default=5)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--steps', type=str, default='20000,20000,20000,40000')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=500)
    p.add_argument('--save_dir', type=str, default='results/ns_multiscale_mask')
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.hi_size; K = args.K
    steps_per_band = [int(x) for x in args.steps.split(',')]

    # Load data
    suffixes = ['', '02', '03', '04', '05']
    data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes[:args.num_dataset]]
    print(f'Loading {args.num_dataset} NS datasets at {G}×{G}:')
    train_data, test_data = load_ns_data(data_locs, G)
    N_train = train_data.shape[0]

    # Truth spectrum from test data
    kvals, enstr_truth, energy_truth = get_energy_spectrum(test_data[:args.n_eval])

    # Build bands
    bands = build_masks(G, K)
    for b in bands:
        s2, M_op, intercept = estimate_sigma2_and_regression(train_data, b['F'], b['C'])
        b['sigma2'] = s2; b['M_op'] = M_op; b['intercept'] = intercept
        print(f"  {b['name']:10s}: R={b['R']:3d} |F|={len(b['F']):5d} |C|={len(b['C']):5d} σ²={b['sigma2']:.4e}")

    while len(steps_per_band) < len(bands):
        steps_per_band.append(steps_per_band[-1])

    save_dir = os.path.join(args.save_dir, f'K{K}_G{G}')
    os.makedirs(save_dir, exist_ok=True)

    # Wandb
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96',
                   name=f'NS_mask_K{K}_G{G}')
        wandb.config.update(vars(args))

    # Move train data to GPU for fast sampling
    train_data_dev = train_data.to(device)

    # ── Train coarse-to-fine ────────────────────────────────────────────
    gen = torch.zeros(args.n_eval, G*G, device=device)
    t0 = timer()

    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']
        sigma2 = b['sigma2']; scale = math.sqrt(sigma2)
        n_steps = steps_per_band[bi]
        in_ch = 1 if len(C) == 0 else 2

        net = make_unet(R, in_ch, 1, args.channels).float().to(device)
        n_params = sum(pp.numel() for pp in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=args.lr*0.01)

        # Move regression operator to device
        M_op_dev = b['M_op'].float().to(device) if b['M_op'] is not None else None
        intercept_dev = b['intercept'].float().to(device) if b['intercept'] is not None else None

        print(f"\n{'='*60}")
        print(f"Band {bi}: {b['name']} — R={R}, {n_params:,} params, {n_steps} steps")
        print(f"{'='*60}")

        for step in range(1, n_steps + 1):
            net.train()
            # Sample batch from training data
            idx = torch.randint(N_train, (args.batch_size,), device=device)
            batch = train_data_dev[idx].view(args.batch_size, -1)
            x_F = batch[:, F]
            x_C = batch[:, C] if len(C) > 0 else None

            # Center by conditional mean
            if M_op_dev is not None and x_C is not None:
                mu = intercept_dev.unsqueeze(0) + x_C @ M_op_dev.T
            else:
                mu = torch.zeros(args.batch_size, nF, device=device)
            x_centered = x_F - mu

            # Flow matching on centered+normalized data
            z0_std = torch.randn(args.batch_size, nF, device=device)
            t = torch.rand(args.batch_size, device=device) * 0.998 + 0.001
            a = (1-t).unsqueeze(1); bv = t.unsqueeze(1)
            zt_norm = a * z0_std + bv * (x_centered / scale)  # O(1)
            target_norm = (x_centered / scale) - z0_std         # O(1)

            # Network input: F at O(1), C at raw
            inp = embed(zt_norm, x_C, F, C, G, R, device)
            pred = net(inp, t, classes=None)

            if len(C) > 0:
                pred_F = extract(pred, F, G, R, device)
            else:
                pred_F = pred.view(args.batch_size, -1)

            loss = (pred_F - target_norm).pow(2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e4)
            opt.step(); sch.step()

            if step % max(1, n_steps // 20) == 0 or step == 1:
                lr = sch.get_last_lr()[0]
                print(f"  step {step}/{n_steps}: loss={loss.item():.4f} lr={lr:.2e} [{(timer()-t0)/60:.1f}m]")
                if not args.no_wandb:
                    wandb.log({f'loss/{b["name"]}': loss.item()}, step=step + bi*100000)

            # ── Eval: generate all bands up to current, check spectrum ──
            if step % args.eval_every == 0 or step == n_steps:
                net.eval()
                x_C_eval = gen[:, C] if len(C) > 0 else None
                if M_op_dev is not None and x_C_eval is not None:
                    mu_eval = intercept_dev.unsqueeze(0) + x_C_eval @ M_op_dev.T
                else:
                    mu_eval = torch.zeros(args.n_eval, nF, device=device)

                zt_eval = torch.randn(args.n_eval, nF, device=device)
                nodes = torch.linspace(1e-3, 1-1e-3, 5)  # 4 RK4 steps
                for ii in range(len(nodes)-1):
                    sv = float(nodes[ii]); ds = float(nodes[ii+1]-nodes[ii])
                    def vel(z, tv):
                        inp_e = embed(z, x_C_eval, F, C, G, R, device)
                        t_e = torch.full((args.n_eval,), tv, device=device)
                        with torch.no_grad(): p = net(inp_e, t_e, classes=None)
                        if len(C) > 0:
                            return extract(p, F, G, R, device)
                        return p.view(args.n_eval, -1)
                    k1=vel(zt_eval,sv); k2=vel(zt_eval+.5*ds*k1,sv+.5*ds)
                    k3=vel(zt_eval+.5*ds*k2,sv+.5*ds); k4=vel(zt_eval+ds*k3,sv+ds)
                    zt_eval = zt_eval+(ds/6)*(k1+2*k2+2*k3+k4)

                gen_temp = gen.clone()
                gen_temp[:, F] = zt_eval * scale + mu_eval

                # Spectrum at current resolution: subsample both gen and truth to R×R
                stride_eval = G // R
                gen_R = gen_temp.view(args.n_eval, G, G)[:, ::stride_eval, ::stride_eval].cpu()
                truth_R = test_data[:args.n_eval, ::stride_eval, ::stride_eval]
                kv_R, enstr_gen, energy_gen = get_energy_spectrum(gen_R)
                kv_R, enstr_tru, energy_tru = get_energy_spectrum(truth_R)
                kmax = min(R // 2 - 1, len(kv_R))
                rel_en = np.abs(energy_gen[:kmax] - energy_tru[:kmax]) / (np.abs(energy_tru[:kmax]) + 1e-30)
                rel_ens = np.abs(enstr_gen[:kmax] - enstr_tru[:kmax]) / (np.abs(enstr_tru[:kmax]) + 1e-30)
                print(f"  [eval R={R}] energy mean_rel≤{kmax}={rel_en.mean():.4f}  "
                      f"enstrophy mean_rel≤{kmax}={rel_ens.mean():.4f}")
                if not args.no_wandb:
                    wandb.log({f'eval/energy_rel_R{R}': rel_en.mean(),
                               f'eval/enstrophy_rel_R{R}': rel_ens.mean()})
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].loglog(kv_R[:kmax], energy_tru[:kmax], 'k-', lw=2, label='truth')
                    axes[0].loglog(kv_R[:kmax], energy_gen[:kmax], 'r--', label='gen')
                    axes[0].set_title(f'Energy R={R}'); axes[0].legend()
                    axes[1].loglog(kv_R[:kmax], enstr_tru[:kmax], 'k-', lw=2, label='truth')
                    axes[1].loglog(kv_R[:kmax], enstr_gen[:kmax], 'r--', label='gen')
                    axes[1].set_title(f'Enstrophy R={R}'); axes[1].legend()
                    plt.tight_layout()
                    wandb.log({f'spectrum_R{R}': wandb.Image(fig)})
                    plt.close()

        # Save and update gen (un-normalize + un-center)
        x_C_final = gen[:, C] if len(C) > 0 else None
        if M_op_dev is not None and x_C_final is not None:
            mu_final = intercept_dev.unsqueeze(0) + x_C_final @ M_op_dev.T
        else:
            mu_final = torch.zeros(args.n_eval, nF, device=device)
        gen[:, F] = zt_eval * scale + mu_final
        bd = os.path.join(save_dir, b['name'])
        os.makedirs(bd, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(bd, 'model.pt'))
        print(f"  Saved to {bd}")

    # ── Final evaluation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    gen_np = gen.view(args.n_eval, G, G).cpu()
    kv, enstr_gen, energy_gen = get_energy_spectrum(gen_np)
    rel_en = np.abs(energy_gen - energy_truth) / (np.abs(energy_truth) + 1e-30)

    print(f"{'k':>4} {'E_truth':>10} {'E_gen':>10} {'ratio':>8} {'rel':>8}")
    for i in range(len(kv)):
        r = energy_gen[i] / (energy_truth[i] + 1e-30)
        marker = ' ***' if abs(r-1) > 1 else (' * ' if abs(r-1) > 0.1 else '   ')
        print(f"{kv[i]:>4.0f} {energy_truth[i]:>10.3e} {energy_gen[i]:>10.3e} {r:>8.4f} {abs(r-1):>8.4f}{marker}")
    print(f"\nmean_rel≤60={rel_en[:60].mean():.4f}  max_rel≤60={rel_en[:60].max():.4f}")
    print(f"Total time: {(timer()-t0)/60:.1f} min")

    if not args.no_wandb:
        wandb.finish()
