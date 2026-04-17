"""
Auto-tuned (sigma, r) for Lip transfer on CelebA-HQ.

Algorithm:
  sigma = c * sqrt(max_k(S_data(k) / S_noise(k)))   [c is margin factor]
  r     = S_data(k_Nyquist) / (sigma^2 * S_noise(k_Nyquist))

Then evaluate using the noise-scaled affine Lip transfer with the noise=1 checkpoint.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from eval_lip_celeba import (load_celeba_data, get_spectrum_per_channel, Velocity,
                              compute_image_metrics, rk4_standard_noisescaled, rk4_lip_noisescaled,
                              compute_fid)


def auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=1.5):
    """
    Auto-select sigma and r such that:
      - sigma^2 * S_noise(k) >= S_data(k) for all k (with margin)
      - r = S_data(k_max) / (sigma^2 * S_noise(k_max))
    Returns sigma, r, and diagnostics.
    """
    # ratio per k for sigma=1
    eps = 1e-30
    ratio_per_k = S_data / (S_noise_unit + eps)
    k_max_arg = np.argmax(ratio_per_k)
    sigma_min = float(np.sqrt(ratio_per_k.max()))
    sigma = margin * sigma_min
    # r at the highest k
    r = float(S_data[-1] / (sigma**2 * S_noise_unit[-1] + eps))
    return sigma, r, dict(sigma_min=sigma_min, k_argmax=kvals[k_max_arg],
                           ratio_max=ratio_per_k.max(), ratio_at_kmax=ratio_per_k[-1])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ckpt', type=str, default='results/celeba_gauss_200k/model_final.pt')
    p.add_argument('--margins', type=float, nargs='+', default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--steps', type=int, nargs='+', default=[10, 20, 50])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    print(f"{'='*70}")
    print(f"  CelebA-HQ Auto-Lip: sweep margin c, derive sigma & r from spectra")
    print(f"  Margins: {args.margins}, Steps: {args.steps}")
    print(f"{'='*70}")

    # Data
    _, test_data = load_celeba_data(128)
    num_eval = min(args.num_eval, test_data.shape[0])
    truth = test_data[:num_eval]
    print(f"  Test data: {truth.shape}, range [{truth.min():.2f}, {truth.max():.2f}]")

    # Spectra
    kvals, S_data = get_spectrum_per_channel(truth)
    torch.manual_seed(123)
    noise_unit = torch.randn(num_eval, 3, 128, 128)
    _, S_noise_unit = get_spectrum_per_channel(noise_unit)

    print(f"\n  Data spectrum: range [{S_data.min():.4e}, {S_data.max():.4e}]")
    print(f"  Noise (sigma=1) spectrum: range [{S_noise_unit.min():.4e}, {S_noise_unit.max():.4e}]")
    print(f"  S_data/S_noise per k: min={(S_data/S_noise_unit).min():.4e}, max={(S_data/S_noise_unit).max():.4e}")
    k_argmax = kvals[np.argmax(S_data/S_noise_unit)]
    print(f"  Worst-case k (where data/noise is largest): k={k_argmax:.1f}")

    # Compute auto (sigma, r) for each margin
    print(f"\n  {'margin':>8} {'sigma':>10} {'r':>15} {'sigma_min':>10}")
    for m in args.margins:
        s, r, diag = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=m)
        print(f"  {m:>8.2f} {s:>10.4f} {r:>15.6e} {diag['sigma_min']:>10.4f}")

    # Model
    model = Velocity(C=3, dim=64, dim_mults=(1, 2, 4, 4)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()
    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} params")

    truth_m, _, truth_spec, _ = compute_image_metrics(truth, truth)

    # Build methods: Standard σ=1 (baseline, matches training), Lip with auto (σ, r) at each margin
    methods = []
    for ns in args.steps:
        methods.append(('Std σ=1 (baseline)', ns, 'std', 1.0, None))
    for margin in args.margins:
        sigma, r, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=margin)
        for ns in args.steps:
            methods.append((f'Lip c={margin} σ={sigma:.2f} r={r:.1e}', ns, 'lip', sigma, r))

    all_results = {(m[0], m[1]): [] for m in methods}
    all_specs = {(m[0], m[1]): [] for m in methods}
    all_fids = {(m[0], m[1]): [] for m in methods}
    batch_size = 50

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx * 1000
        torch.manual_seed(seed)
        z0_unit = torch.randn(num_eval, 3, 128, 128)

        for label, nsteps, kind, sigma, r in methods:
            print(f"  seed={seed}, {label} RK4-{nsteps}...", flush=True)
            chunks = []
            for start in range(0, num_eval, batch_size):
                end = min(start + batch_size, num_eval)
                z0_b = z0_unit[start:end].to(device)
                if kind == 'std':
                    chunk = rk4_standard_noisescaled(model, z0_b, nsteps, sigma)
                else:
                    chunk = rk4_lip_noisescaled(model, z0_b, nsteps, r, sigma)
                chunks.append(chunk.cpu()); torch.cuda.empty_cache()
            gen = torch.cat(chunks, dim=0)
            m, _, _, spec_gen = compute_image_metrics(truth, gen)
            all_results[(label, nsteps)].append(m)
            all_specs[(label, nsteps)].append(spec_gen)
            try:
                fid = compute_fid(truth, gen, device_str=str(device))
                all_fids[(label, nsteps)].append(fid)
                print(f"    FID = {fid:.2f}")
            except Exception as e:
                print(f"    FID failed: {type(e).__name__}")
                all_fids[(label, nsteps)].append(float('nan'))

    # ─── Print results ────────────────────────────────────────────────

    def agg(vals):
        a = np.array(vals); return a.mean(), a.std()

    print(f"\n{'='*70}")
    print(f"  RESULTS (mean ± std over {args.num_seeds} seeds)")
    print(f"  Truth: pixel_std={truth_m['pixel_std']:.4f}, grad_kurt={truth_m['grad_kurt']:.4f}")
    print(f"{'='*70}")
    print(f"\n--- Spectrum error by band (mean ± std, low: k<8, mid: 8≤k<24, high: k≥24) ---")
    print(f"  {'Method':<35} {'steps':>5} {'low':>16} {'mid':>16} {'high':>16} {'mean':>16}")
    print(f"  {'-'*108}")
    for label, nsteps, _, _, _ in methods:
        key = (label, nsteps)
        sa = [m['spec_err_mean'] for m in all_results[key]]
        sl = [m.get('spec_err_low', 0) for m in all_results[key]]
        sm = [m.get('spec_err_mid', 0) for m in all_results[key]]
        sh = [m.get('spec_err_high', 0) for m in all_results[key]]
        ml, sdl = agg(sl); mm, sdm = agg(sm); mh, sdh = agg(sh); ma, sda = agg(sa)
        print(f"  {label:<35} {nsteps:>5} {ml:.4f}±{sdl:.4f} {mm:.4f}±{sdm:.4f} {mh:.4f}±{sdh:.4f} {ma:.4f}±{sda:.4f}")

    print(f"\n--- FID (lower is better) ---")
    print(f"  {'Method':<35} {'steps':>5} {'FID':>16}")
    print(f"  {'-'*60}")
    for label, nsteps, _, _, _ in methods:
        key = (label, nsteps)
        fids = all_fids[key]
        if any(not np.isnan(f) for f in fids):
            mu, sd = agg(fids)
            print(f"  {label:<35} {nsteps:>5} {mu:>8.2f} ± {sd:>5.2f}")
        else:
            print(f"  {label:<35} {nsteps:>5} {'nan':>16}")

    print(f"\n--- Image quality metrics ---")
    print(f"  {'Method':<35} {'steps':>5} {'pixel_std':>10} {'var_ratio':>10} {'grad_kurt':>10}")
    print(f"  {'-'*78}")
    for label, nsteps, _, _, _ in methods:
        key = (label, nsteps)
        ps = [m['pixel_std'] for m in all_results[key]]
        vr = [m['total_var_ratio'] for m in all_results[key]]
        gk = [m['grad_kurt'] for m in all_results[key]]
        print(f"  {label:<35} {nsteps:>5} {agg(ps)[0]:>10.4f} {agg(vr)[0]:>10.4f} {agg(gk)[0]:>10.4f}")

    # ─── Plot ────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Spectra (data, noise unit, noise scaled to each sigma)
    ax = axes[0, 0]
    ax.loglog(kvals, S_data, 'k-', lw=2.5, label='Data')
    ax.loglog(kvals, S_noise_unit, 'gray', lw=1.5, ls=':', label='Noise σ=1')
    for margin in args.margins:
        sigma, _, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=margin)
        ax.loglog(kvals, sigma**2 * S_noise_unit, '--', lw=1.0, label=f'Noise σ={sigma:.1f} (c={margin})')
    ax.set_xlabel('k'); ax.set_ylabel('Power spectrum')
    ax.set_title('Data vs noise spectra (noise must dominate)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Spectrum error vs k at max steps
    ref = max(args.steps)
    std_baseline = 'Std σ=1 (baseline)'
    ax = axes[0, 1]
    palette = ['blue', 'green', 'purple', 'orange']
    if (std_baseline, ref) in all_specs:
        spec = np.mean(all_specs[(std_baseline, ref)], axis=0)
        err = np.abs(spec - truth_spec) / (np.abs(truth_spec) + 1e-20)
        ax.semilogy(kvals, err, 'r--', lw=1.5, label='Std σ=1')
    for i, margin in enumerate(args.margins):
        sigma, r, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=margin)
        lip_label = f'Lip c={margin} σ={sigma:.2f} r={r:.1e}'
        c = palette[i % len(palette)]
        if (lip_label, ref) in all_specs:
            spec = np.mean(all_specs[(lip_label, ref)], axis=0)
            err = np.abs(spec - truth_spec) / (np.abs(truth_spec) + 1e-20)
            ax.semilogy(kvals, err, '-', color=c, lw=1.5, label=f'Lip c={margin}')
    ax.axhline(y=0.1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('k'); ax.set_ylabel('Rel error')
    ax.set_title(f'Per-k Spectrum Error ({ref} steps)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Spec error vs steps
    ax = axes[1, 0]
    if all((std_baseline, n) in all_results for n in args.steps):
        errs_std = [agg([m['spec_err_mean'] for m in all_results[(std_baseline, n)]])[0] for n in args.steps]
        ax.semilogy(args.steps, errs_std, 'r--s', lw=2, markersize=6, label='Std σ=1')
    for i, margin in enumerate(args.margins):
        sigma, r, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=margin)
        lip_label = f'Lip c={margin} σ={sigma:.2f} r={r:.1e}'
        if all((lip_label, n) in all_results for n in args.steps):
            errs_lip = [agg([m['spec_err_mean'] for m in all_results[(lip_label, n)]])[0] for n in args.steps]
            c = palette[i % len(palette)]
            ax.semilogy(args.steps, errs_lip, '-o', color=c, lw=2, markersize=6, label=f'Lip c={margin}')
    ax.set_xlabel('RK4 steps'); ax.set_ylabel('Mean spectrum error')
    ax.set_title('Spectrum error vs steps')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(args.steps)

    # Sample images
    ax = axes[1, 1]
    # Pick best Lip method and show grid
    best_margin = args.margins[len(args.margins) // 2]
    sigma, r, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=best_margin)
    best_label = f'Lip c={best_margin} σ={sigma:.2f} r={r:.1e}'
    if (best_label, ref) in all_specs:
        # Just show the spectra comparison
        ax.loglog(kvals, truth_spec, 'k-', lw=2.5, label='Truth')
        for i, margin in enumerate(args.margins):
            sigma, r, _ = auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=margin)
            lip_label = f'Lip c={margin} σ={sigma:.2f} r={r:.1e}'
            if (lip_label, ref) in all_specs:
                spec = np.mean(all_specs[(lip_label, ref)], axis=0)
                ax.loglog(kvals, spec, '-', color=palette[i % len(palette)], lw=1.5,
                         label=f'Lip c={margin}')
        ax.set_xlabel('k'); ax.set_ylabel('Power')
        ax.set_title(f'Generated spectra ({ref} steps)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('CelebA-HQ Auto-tuned Lip Transfer', fontsize=14)
    plt.tight_layout()
    plt.savefig('celeba_auto_lip.png', dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: celeba_auto_lip.png")
