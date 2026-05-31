"""Quick sanity-check the generated Kolmogorov-flow data.

Subsamples heavily to stay fast. Produces figs/verify_data.png.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft


def radial_spectrum_vec(omega):
    """(B, H, W) -> (B, N_k). Vectorised via torch.bincount."""
    B, H, W = omega.shape
    w_hat = fft.fft2(omega) / (H * W)
    power = w_hat.real ** 2 + w_hat.imag ** 2
    kx = fft.fftfreq(H, d=1.0 / H).view(-1, 1).expand(H, W)
    ky = fft.fftfreq(W, d=1.0 / W).view(1, -1).expand(H, W)
    kmag = (kx * kx + ky * ky).sqrt()
    k_int = kmag.long().flatten()                        # (H*W,)
    n_k = H // 2 + 1
    k_int = k_int.clamp_max(n_k - 1)
    # E(k) = |omega_hat|^2 / (2 k^2) summed in ring, but accumulate raw power first
    spec = torch.zeros(B, n_k, dtype=omega.dtype)
    for b in range(B):
        s = torch.bincount(k_int, weights=power[b].flatten(), minlength=n_k)
        spec[b] = s
    # divide by 2 k^2 (avoid k=0)
    kvals = torch.arange(n_k, dtype=omega.dtype).clamp_min(1.0)
    spec = spec / (2.0 * kvals.unsqueeze(0) ** 2)
    return torch.arange(n_k, dtype=omega.dtype), spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='figs/verify_data.png')
    p.add_argument('--n_grid', type=int, default=4)
    p.add_argument('--n_samples_spec', type=int, default=200,
                   help='snapshots to average the spectrum over')
    args = p.parse_args()

    ck = torch.load(args.data, weights_only=False, map_location='cpu', mmap=True)
    omega = ck['omega']   # (N_traj, N_snap, H, W)
    sim_args = ck['args']
    N, T, H, W = omega.shape
    print(f"[verify] data shape {tuple(omega.shape)}")
    print(f"[verify] global vorticity: mean={omega[0].mean():.3f} (traj 0)")

    # 1. Sample snapshots (fast: index a few)
    torch.manual_seed(0)
    n = args.n_grid
    snap_grid = torch.zeros(n, n, H, W)
    for i in range(n):
        for j in range(n):
            tj = torch.randint(0, N, (1,)).item()
            ts = torch.randint(0, T, (1,)).item()
            snap_grid[i, j] = omega[tj, ts]
    tiled = snap_grid.permute(0, 2, 1, 3).reshape(n * H, n * W)
    vlim = float(snap_grid.abs().quantile(0.995))

    # 2. Enstrophy vs time (only 20 trajectories, cheap)
    traj_idx = torch.randperm(N)[:20]
    omega_sub = omega[traj_idx].contiguous()             # (20, T, H, W)
    ens_t = 0.5 * (omega_sub ** 2).mean(dim=(-1, -2))    # (20, T)

    # 3. Energy spectrum: sample ~n_samples_spec random (traj, snap) pairs, vectorise
    idx_tj = torch.randint(0, N, (args.n_samples_spec,))
    idx_ts = torch.randint(0, T, (args.n_samples_spec,))
    omega_spec = torch.stack([omega[ti, si] for ti, si in zip(idx_tj, idx_ts)], dim=0)
    k_bins, spec = radial_spectrum_vec(omega_spec)
    spec_mean = spec.mean(dim=0).numpy()

    # 4. PDF of pixel vorticity (flatten a subset)
    sample = omega_sub.reshape(-1)
    idx_pdf = torch.randperm(sample.numel())[:100_000]
    sample = sample[idx_pdf].numpy()

    # 5. Autocorrelation at the centre pixel (cheap: 20 traj)
    pix = omega_sub[:, :, H // 2, W // 2].float()       # (20, T)
    pix = pix - pix.mean(dim=1, keepdim=True)
    var = pix.pow(2).mean()
    max_lag = min(T // 2, 60)
    ac = torch.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            ac[lag] = 1.0; continue
        ac[lag] = ((pix[:, :-lag] * pix[:, lag:]).mean() / var).item()

    # 6. Trajectory snapshots at lags
    lags = [1, 10, 40]
    n_show = 4
    tj = torch.randperm(N)[:n_show]
    ts_start = min(10, T - max(lags) - 1)
    tiles_lag = torch.zeros(n_show * H, (1 + len(lags)) * W)
    for i in range(n_show):
        tiles_lag[i*H:(i+1)*H, :W] = omega[tj[i], ts_start]
        for j, lg in enumerate(lags):
            tiles_lag[i*H:(i+1)*H, (j+1)*W:(j+2)*W] = omega[tj[i], ts_start + lg]

    # Plot
    import seaborn as sns
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tiled, cmap=sns.cm.icefire, vmin=-vlim, vmax=vlim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Vorticity snapshots')

    ax = fig.add_subplot(gs[0, 1])
    for k in range(ens_t.shape[0]):
        ax.plot(range(T), ens_t[k].numpy(), alpha=0.4)
    ax.plot(range(T), ens_t.mean(dim=0).numpy(), 'k-', lw=2, label='mean')
    ax.set_xlabel('snapshot index'); ax.set_ylabel('enstrophy')
    ax.set_title(f'Enstrophy vs time (mean={ens_t.mean():.2f})')
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(k_bins[1:].numpy(), spec_mean[1:], 'b-', label='data')
    ref_idx = 2
    ax.loglog(k_bins[ref_idx:].numpy(),
              spec_mean[ref_idx] * (k_bins[ref_idx:].numpy() / k_bins[ref_idx].item()) ** (-5/3),
              'r--', label='$k^{-5/3}$', alpha=0.6)
    ax.loglog(k_bins[ref_idx:].numpy(),
              spec_mean[ref_idx] * (k_bins[ref_idx:].numpy() / k_bins[ref_idx].item()) ** (-3),
              'g:', label='$k^{-3}$', alpha=0.6)
    ax.set_xlabel('$k$'); ax.set_ylabel('$E(k)$')
    ax.set_title('Time-averaged energy spectrum')
    ax.grid(which='both', alpha=0.3); ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(sample, bins=100, density=True)
    ax.set_xlabel('$\\omega$'); ax.set_ylabel('density')
    ax.set_title('Vorticity PDF')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(range(len(ac)), ac.numpy(), 'b-')
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel('lag (snapshots)'); ax.set_ylabel('autocorrelation')
    ax.set_title(f'Centre-pixel AC (Δt_snap={sim_args.get("snap_dt")})')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1:, 2])
    ax.imshow(tiles_lag.numpy(), cmap=sns.cm.icefire, vmin=-vlim, vmax=vlim)
    ax.set_xticks([W * (i + 0.5) for i in range(1 + len(lags))])
    ax.set_xticklabels(['$t$'] + [f'$t+{lg}$' for lg in lags])
    ax.set_yticks([])
    ax.set_title('Trajectory at forecast lags (4 samples)')

    ax = fig.add_subplot(gs[2, 0:2])
    ax.axis('off')
    decorr = 0
    for i in range(1, len(ac)):
        if ac[i].item() <= 0.1:
            decorr = i; break
    txt = (f"Stats summary (N_traj={N}, N_snap={T}, resolution={H}x{W})\n"
           f"  viscosity ν = {sim_args.get('viscosity')}\n"
           f"  drag = {sim_args.get('drag')}\n"
           f"  forcing k = {sim_args.get('forcing_k')} (sin({sim_args.get('forcing_k')}·y))\n"
           f"  snap Δt = {sim_args.get('snap_dt')}\n"
           f"  solver Δt = {sim_args.get('dt_solver')}\n"
           f"  vorticity: std≈{omega_sub.std():.2f} max|ω|≈{omega_sub.abs().max():.1f}\n"
           f"  mean enstrophy = {ens_t.mean():.3f}\n"
           f"  e-folding time (AC<=0.1) ≈ {decorr} snaps = {decorr * sim_args.get('snap_dt'):.2f} time units")
    ax.text(0.0, 1.0, txt, family='monospace', fontsize=11, va='top')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"[saved] {args.out}")


if __name__ == '__main__':
    main()
