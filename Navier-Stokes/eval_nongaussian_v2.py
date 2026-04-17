"""
Non-Gaussian evaluation: metrics that go BEYOND second-order (spectrum).
All metrics here capture genuinely non-Gaussian structure.
Runs over multiple random seeds and reports mean ± std.

Metrics:
  1. Flatness S_4(r)/S_2(r)^2 — intermittency (=3 for Gaussian)
  2. Gradient kurtosis — 4th-order moment at smallest scale (=3 for Gaussian)
  3. Per-sample kurtosis distribution — excess kurtosis beyond Gaussian
  4. Tail probabilities P(|omega|>tau) — heavy-tail structure
  5. KS distance — full distributional comparison
  6. Corr(omega^2, omega^2) — 4th-order spatial correlation (=0 for Gaussian)
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
from unet import Unet


# ─── Infrastructure (data, model, samplers) ───────────────────────────

def load_ns_data(data_locs, hi_size, train_test_split):
    if isinstance(data_locs, str):
        data_locs = [data_locs]
    avg_pixel_norm = 3.0679163932800293
    all_data = []
    for loc in data_locs:
        data_raw, _ = torch.load(loc, weights_only=False)
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, data_raw.shape[2], data_raw.shape[3])
        if hi_size != data.shape[1]:
            data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        all_data.append(data)
    data = torch.cat(all_data, dim=0)[:, None, :, :]
    num_train = int(data.shape[0] * train_test_split)
    return data[:num_train], data[num_train:]

class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1, dim=32,
                        dim_mults=(1,2,2,2), resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4, use_classes=False)
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)

@torch.no_grad()
def rk4_standard(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0; ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = model(zt, ta); k2 = model(zt+.5*dt*k1, ta+.5*dt)
        k3 = model(zt+.5*dt*k2, ta+.5*dt); k4 = model(zt+dt*k3, ta+dt)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt

@torch.no_grad()
def rk4_lip(model, z0, steps, ratio, t_min=1e-3, t_max=1-1e-3):
    r = ratio; log_r = math.log(r)
    def alpha(t): return torch.sqrt((r - r**t)/(r - 1)) * torch.ones_like(t)
    def alpha_dot(t): return -0.5/alpha(t) * (r**t) * log_r/(r - 1)
    def beta(t): return torch.sqrt((r**t - 1)/(r - 1)) * torch.ones_like(t)
    def beta_dot(t): return 0.5/beta(t) * (r**t) * log_r/(r - 1)
    def drift(zt, ta):
        bt = (alpha_dot(ta)/alpha(ta))[:, None, None, None] * zt
        coef = (beta_dot(ta) - alpha_dot(ta)*beta(ta)/alpha(ta))[:, None, None, None]
        orig_t = 1/(1 + alpha(ta)/beta(ta))
        orig_x = orig_t[:, None, None, None]/(beta(ta)[:, None, None, None]) * zt
        orig_bt = model(orig_x, orig_t)
        bt += coef * ((1 - orig_t[:, None, None, None]) * orig_bt + orig_x)
        return bt
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0; ones = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tgrid) - 1):
        t_i = tgrid[i]; dt = tgrid[i+1] - tgrid[i]; ta = t_i * ones
        k1 = drift(zt, ta); k2 = drift(zt+.5*dt*k1, (t_i+.5*dt)*ones)
        k3 = drift(zt+.5*dt*k2, (t_i+.5*dt)*ones); k4 = drift(zt+dt*k3, (t_i+dt)*ones)
        zt = zt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return zt


# ─── Non-Gaussian metrics (all genuinely beyond 2nd order) ───────────

def compute_metrics(samples):
    """
    Compute all non-Gaussian metrics on a batch of samples.
    samples: (N, H, W) torch tensor
    Returns dict of scalar metrics.
    """
    x = samples.numpy()
    N, H, W = x.shape

    # 1. Gradient kurtosis (4th order at pixel scale)
    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = x[:, 1:, :] - x[:, :-1, :]
    grad = np.concatenate([dx.reshape(-1), dy.reshape(-1)])
    grad_kurt = np.mean(grad**4) / (np.mean(grad**2)**2 + 1e-20)

    # 2. Per-sample kurtosis: mean and std across samples
    flat = x.reshape(N, -1)
    mu = flat.mean(axis=1, keepdims=True)
    centered = flat - mu
    var = np.mean(centered**2, axis=1)
    sample_kurt = np.mean(centered**4, axis=1) / (var**2 + 1e-20)
    kurt_mean = sample_kurt.mean()
    kurt_std = sample_kurt.std()

    # 3. Flatness at r=1,2,4 (S4/S2^2, genuinely 4th order)
    flatness = {}
    for r in [1, 2, 4]:
        diffs = []
        if r < H: diffs.append((x[:, r:, :] - x[:, :-r, :]).reshape(-1))
        if r < W: diffs.append((x[:, :, r:] - x[:, :, :-r]).reshape(-1))
        d = np.concatenate(diffs)
        s2 = np.mean(d**2)
        s4 = np.mean(d**4)
        flatness[r] = s4 / (s2**2 + 1e-20)

    # 4. Tail probabilities (sensitive to non-Gaussian tails)
    flat_all = x.reshape(-1)
    tails = {}
    for tau in [2.0, 2.5, 3.0, 3.5]:
        tails[tau] = np.mean(np.abs(flat_all) > tau)

    # 5. Corr(omega^2, omega^2) excess at r=1,4,8 (4th order spatial)
    x2 = x**2
    x2_mean_sq = np.mean(x2)**2
    sq_corr = {}
    for r in [1, 4, 8]:
        if r < W:
            c = np.mean(x2[:, :, r:] * x2[:, :, :-r])
            sq_corr[r] = c / x2_mean_sq - 1  # excess over Gaussian (=0)

    return dict(
        grad_kurt=grad_kurt,
        kurt_mean=kurt_mean, kurt_std=kurt_std,
        flatness=flatness,
        tails=tails,
        sq_corr=sq_corr,
    )


def compute_ks(truth_samples, gen_samples, n_subsample=100000):
    """KS test on pixel distributions."""
    rng = np.random.RandomState(0)
    tf = truth_samples.numpy().reshape(-1)
    gf = gen_samples.numpy().reshape(-1)
    ts = tf[rng.choice(len(tf), min(n_subsample, len(tf)), replace=False)]
    gs = gf[rng.choice(len(gf), min(n_subsample, len(gf)), replace=False)]
    ks_stat, _ = stats.ks_2samp(ts, gs)
    return ks_stat


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--hi_size', type=int, default=128)
    p.add_argument('--lip_r', type=float, default=1e-5)
    p.add_argument('--noise_strength', type=float, default=10.0)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--num_eval', type=int, default=500)
    p.add_argument('--num_seeds', type=int, default=5)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    hi = args.hi_size; ns = args.noise_strength

    print(f"{'='*70}")
    print(f"  Non-Gaussian eval (genuinely beyond 2nd order)")
    print(f"  {hi}x{hi}, noise={ns}, lip_r={args.lip_r:.0e}, seeds={args.num_seeds}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"{'='*70}")

    _, test_data = load_ns_data('../NSdata/data_file.pt', hi, 0.9)
    num_eval = min(args.num_eval, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)

    model = Velocity().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()

    # Truth metrics (single computation)
    truth_m = compute_metrics(truth_sq)

    # Methods x steps
    methods = [('Standard', 10), ('Standard', 20), ('Lip', 10), ('Lip', 20)]

    # Collect metrics across seeds
    all_seed_metrics = {key: [] for key in methods}
    all_seed_ks = {key: [] for key in methods}

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx * 1000
        torch.manual_seed(seed)
        z0 = ns * torch.randn(num_eval, 1, hi, hi, device=device)

        for method, nsteps in methods:
            print(f"  seed={seed}, {method} RK4-{nsteps}...", flush=True)
            if method == 'Standard':
                gen = rk4_standard(model, z0.clone(), nsteps).squeeze(1).cpu()
            else:
                gen = rk4_lip(model, z0.clone(), nsteps, args.lip_r).squeeze(1).cpu()

            m = compute_metrics(gen)
            ks = compute_ks(truth_sq, gen)
            all_seed_metrics[(method, nsteps)].append(m)
            all_seed_ks[(method, nsteps)].append(ks)

    # ─── Aggregate across seeds ───────────────────────────────────────

    def agg(values):
        """Return mean ± std string."""
        arr = np.array(values)
        return arr.mean(), arr.std()

    print(f"\n{'='*70}")
    print(f"  RESULTS (mean ± std over {args.num_seeds} seeds)")
    print(f"  Truth values shown for reference (Gaussian value in parentheses)")
    print(f"{'='*70}")

    # Gradient kurtosis
    print(f"\n--- Gradient kurtosis (Gaussian=3) ---")
    print(f"  Truth: {truth_m['grad_kurt']:.4f}")
    for key in methods:
        vals = [m['grad_kurt'] for m in all_seed_metrics[key]]
        mu, sd = agg(vals)
        err = abs(mu - truth_m['grad_kurt']) / truth_m['grad_kurt']
        label = f"{key[0]} {key[1]}s"
        print(f"  {label:<16} {mu:.4f} ± {sd:.4f}  (rel err {err:.4f})")

    # Per-sample kurtosis
    print(f"\n--- Per-sample kurtosis mean (Gaussian=3) ---")
    print(f"  Truth: {truth_m['kurt_mean']:.4f} ± {truth_m['kurt_std']:.4f}")
    for key in methods:
        vals_mu = [m['kurt_mean'] for m in all_seed_metrics[key]]
        vals_sd = [m['kurt_std'] for m in all_seed_metrics[key]]
        mu, sd = agg(vals_mu)
        label = f"{key[0]} {key[1]}s"
        print(f"  {label:<16} {mu:.4f} ± {sd:.4f}")

    # Flatness at different r
    for r in [1, 2, 4]:
        print(f"\n--- Flatness S4/S2^2 at r={r} (Gaussian=3) ---")
        print(f"  Truth: {truth_m['flatness'][r]:.4f}")
        for key in methods:
            vals = [m['flatness'][r] for m in all_seed_metrics[key]]
            mu, sd = agg(vals)
            err = abs(mu - truth_m['flatness'][r]) / truth_m['flatness'][r]
            label = f"{key[0]} {key[1]}s"
            print(f"  {label:<16} {mu:.4f} ± {sd:.4f}  (rel err {err:.4f})")

    # Tail probabilities
    for tau in [2.5, 3.0, 3.5]:
        print(f"\n--- P(|omega| > {tau}) ---")
        print(f"  Truth: {truth_m['tails'][tau]:.6f}")
        for key in methods:
            vals = [m['tails'][tau] for m in all_seed_metrics[key]]
            mu, sd = agg(vals)
            err = abs(mu - truth_m['tails'][tau]) / (truth_m['tails'][tau] + 1e-20)
            label = f"{key[0]} {key[1]}s"
            print(f"  {label:<16} {mu:.6f} ± {sd:.6f}  (rel err {err:.4f})")

    # KS distance
    print(f"\n--- KS distance (lower = better) ---")
    for key in methods:
        mu, sd = agg(all_seed_ks[key])
        label = f"{key[0]} {key[1]}s"
        print(f"  {label:<16} {mu:.6f} ± {sd:.6f}")

    # Squared-vorticity correlation
    for r in [1, 4, 8]:
        print(f"\n--- Corr(omega^2, omega^2) excess at r={r} (Gaussian=0) ---")
        print(f"  Truth: {truth_m['sq_corr'][r]:.4f}")
        for key in methods:
            vals = [m['sq_corr'][r] for m in all_seed_metrics[key]]
            mu, sd = agg(vals)
            err = abs(mu - truth_m['sq_corr'][r]) / (abs(truth_m['sq_corr'][r]) + 1e-20)
            label = f"{key[0]} {key[1]}s"
            print(f"  {label:<16} {mu:.4f} ± {sd:.4f}  (rel err {err:.4f})")
