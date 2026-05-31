"""
CelebA-HQ: Test noise-scaling + Lip transfer using existing noise=1 checkpoint.

The trained checkpoint was with noise_strength=1 (z0 ~ N(0,I)).
We transfer to effective noise_strength=sigma using the affine drift transformation.
Then optionally apply Lip schedule on top.

Affine transfer (combines noise scaling and Lip schedule):
  effective interpolant: z_t = sigma*alpha_lip(t)*z0 + beta_lip(t)*z1
  orig_t = beta / (sigma*alpha + beta)
  orig_x = z_t / (sigma*alpha + beta)
  b(z,t) = orig_x*(sigma*alpha_dot + beta_dot)
         + v_nn(orig_x, orig_t)*(-sigma*alpha_dot*orig_t + beta_dot*(1-orig_t))
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
                              compute_image_metrics, rk4_standard_noisescaled, rk4_lip_noisescaled)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ckpt', type=str, default='results/celeba_gauss_200k/model_final.pt')
    p.add_argument('--sigma', type=float, default=10.0, help='Effective noise strength')
    p.add_argument('--lip_rs', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5])
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--steps', type=int, nargs='+', default=[10, 20, 50])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    print(f"{'='*70}")
    print(f"  CelebA-HQ: noise-scaled Lip transfer")
    print(f"  Existing checkpoint trained with noise=1, evaluated with sigma={args.sigma}")
    print(f"  Lip ratios r = {args.lip_rs}")
    print(f"  Steps: {args.steps}")
    print(f"{'='*70}")

    _, test_data = load_celeba_data(128)
    num_eval = min(args.num_eval, test_data.shape[0])
    truth = test_data[:num_eval]
    print(f"  Test data: {truth.shape}, range [{truth.min():.2f}, {truth.max():.2f}]")

    model = Velocity(C=3, dim=64, dim_mults=(1, 2, 4, 4)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Methods: standard with noise scaling + each Lip r with noise scaling
    methods = []
    for ns in args.steps:
        methods.append((f'Std σ={args.sigma}', ns, 'std', None))
    for lr in args.lip_rs:
        for ns in args.steps:
            methods.append((f'Lip r={lr:.0e} σ={args.sigma}', ns, 'lip', lr))

    truth_m, kvals, truth_spec, _ = compute_image_metrics(truth, truth)

    all_results = {key: [] for key in [(m[0], m[1]) for m in methods]}
    all_specs = {key: [] for key in [(m[0], m[1]) for m in methods]}
    batch_size = 50

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx * 1000
        torch.manual_seed(seed)
        z0_unit = torch.randn(num_eval, 3, 128, 128)  # CPU, std=1

        for method_label, nsteps, kind, lr in methods:
            print(f"  seed={seed}, {method_label} RK4-{nsteps}...", flush=True)
            gen_chunks = []
            for start in range(0, num_eval, batch_size):
                end = min(start + batch_size, num_eval)
                z0_batch = z0_unit[start:end].to(device)
                if kind == 'std':
                    chunk = rk4_standard_noisescaled(model, z0_batch, nsteps, args.sigma)
                else:
                    chunk = rk4_lip_noisescaled(model, z0_batch, nsteps, lr, args.sigma)
                gen_chunks.append(chunk.cpu())
                torch.cuda.empty_cache()
            gen_cpu = torch.cat(gen_chunks, dim=0)
            m, _, _, spec_gen = compute_image_metrics(truth, gen_cpu)
            all_results[(method_label, nsteps)].append(m)
            all_specs[(method_label, nsteps)].append(spec_gen)

    # ─── Print results ────────────────────────────────────────────────

    def agg(vals):
        arr = np.array(vals)
        return arr.mean(), arr.std()

    print(f"\n{'='*70}")
    print(f"  RESULTS (mean ± std over {args.num_seeds} seeds)")
    print(f"  Truth: pixel_std={truth_m['pixel_std']:.4f}, grad_kurt={truth_m['grad_kurt']:.4f}")
    print(f"{'='*70}")

    print(f"\n--- Spectrum relative error by band ---")
    print(f"  {'Method':<25} {'steps':>5} {'low':>8} {'mid':>8} {'high':>8} {'mean':>15}")
    print(f"  {'-'*70}")
    for method_label, nsteps, _, _ in methods:
        key = (method_label, nsteps)
        vals_low = [m.get('spec_err_low', 0) for m in all_results[key]]
        vals_mid = [m.get('spec_err_mid', 0) for m in all_results[key]]
        vals_high = [m.get('spec_err_high', 0) for m in all_results[key]]
        vals_mean = [m['spec_err_mean'] for m in all_results[key]]
        ml, _ = agg(vals_low); mm, _ = agg(vals_mid)
        mh, _ = agg(vals_high); ma, sa = agg(vals_mean)
        print(f"  {method_label:<25} {nsteps:>5} {ml:>8.4f} {mm:>8.4f} {mh:>8.4f} {ma:.4f}±{sa:.4f}")

    print(f"\n--- Image quality ---")
    print(f"  {'Method':<25} {'steps':>5} {'pixel_std':>10} {'grad_kurt':>10} {'var_ratio':>10}")
    print(f"  {'-'*65}")
    for method_label, nsteps, _, _ in methods:
        key = (method_label, nsteps)
        ps = [m['pixel_std'] for m in all_results[key]]
        gk = [m['grad_kurt'] for m in all_results[key]]
        vr = [m['total_var_ratio'] for m in all_results[key]]
        print(f"  {method_label:<25} {nsteps:>5} {agg(ps)[0]:>10.4f} {agg(gk)[0]:>10.4f} {agg(vr)[0]:>10.4f}")

    # ─── Plot ────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    all_colors = {}
    palette = ['red', 'blue', 'green', 'purple', 'orange']
    method_names_unique = []
    for m in methods:
        if m[0] not in method_names_unique:
            method_names_unique.append(m[0])
    for i, name in enumerate(method_names_unique):
        all_colors[name] = palette[i % len(palette)]

    ref = max(args.steps)
    ax = axes[0]
    ax.loglog(kvals, truth_spec, 'k-', lw=2.5, label='Truth')
    for method_label, nsteps, _, _ in methods:
        if nsteps == ref:
            specs = all_specs[(method_label, nsteps)]
            mean_spec = np.mean(specs, axis=0)
            c = all_colors[method_label]
            ls = '--' if 'Std' in method_label else '-'
            ax.loglog(kvals, mean_spec, ls, color=c, lw=1.5, label=method_label)
    ax.set_xlabel('k'); ax.set_ylabel('Power spectrum')
    ax.set_title(f'CelebA Power Spectrum (RK4 {ref}, σ={args.sigma})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method_label, nsteps, _, _ in methods:
        if nsteps == ref:
            specs = all_specs[(method_label, nsteps)]
            mean_spec = np.mean(specs, axis=0)
            rel_err = np.abs(mean_spec - truth_spec) / (np.abs(truth_spec) + 1e-20)
            c = all_colors[method_label]
            ls = '--' if 'Std' in method_label else '-'
            ax.semilogy(kvals, rel_err, ls, color=c, lw=1.5, label=method_label)
    ax.axhline(y=0.1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('k'); ax.set_ylabel('Relative error')
    ax.set_title(f'Per-k Spectrum Error (RK4 {ref})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'celeba_noisescaled_sigma{args.sigma:.0f}.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {out}")
