"""Path-KL Girsanov estimator + marginal metrics for the §3.4.1 study.

Path-KL:
    KL(P^{g}_theta || P^{g, *}) = 0.5 * E_t E_{X_t ~ p_t} [ g(t)^{-2} || b^g_theta - b^g_* ||^2 ].

Both drifts are composed via eq. (3.9); the only difference is the baseline drift
(theta network vs analytic) underneath. X_t is drawn from the interpolant
I_t = t x_star + (1-t) sqrt(t) z with x_star ~ target.

Terminal marginal metrics: W1, W2, MMD, moments, histogram KL.
"""
import math
import torch
import numpy as np


def wasserstein_1d(a, b, p=2):
    assert a.numel() == b.numel()
    a_sorted, _ = a.reshape(-1).sort()
    b_sorted, _ = b.reshape(-1).sort()
    diff = (a_sorted - b_sorted).abs()
    if p == 1:
        return diff.mean().item()
    return (diff.pow(p).mean().pow(1.0 / p)).item()


def sliced_wasserstein(a, b, n_proj=200, p=2, seed=0):
    """Sliced p-Wasserstein between two same-size 2D+ empirical distributions.
    For each of n_proj random unit directions, project both samples, compute
    the 1D W_p, average (p-th root at the end)."""
    assert a.shape == b.shape
    D = a.shape[1]
    gen = torch.Generator(device=a.device).manual_seed(seed)
    dirs = torch.randn(n_proj, D, generator=gen, device=a.device, dtype=a.dtype)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    # (n_proj, n) projections
    pa = a @ dirs.T       # (n, n_proj)
    pb = b @ dirs.T
    pa_s, _ = pa.sort(dim=0)
    pb_s, _ = pb.sort(dim=0)
    diff = (pa_s - pb_s).abs()
    if p == 1:
        return diff.mean().item()
    return (diff.pow(p).mean().pow(1.0 / p)).item()


def mmd_rbf(a, b, bandwidths=None, max_n=5000, use_median_heuristic=True):
    """Biased (V-statistic) MMD^2 with a mixture of RBF kernels:
        MMD^2_b = (1/n^2) sum k(x_i, x_j) + (1/m^2) sum k(y_i, y_j)
                 - (2/(nm)) sum k(x_i, y_j)

    The V-statistic is ALWAYS non-negative (unlike the unbiased U-statistic),
    with an O(1/n) positive bias that vanishes at the sample sizes used here.
    Using the biased estimator avoids misleading negative values while giving
    interpretable magnitudes.

    By default bandwidths are chosen by the median heuristic over the combined
    sample (robust, scale-adapting). Passing an explicit tuple overrides that.
    """
    a = a.reshape(a.shape[0], -1)[:max_n]
    b = b.reshape(b.shape[0], -1)[:max_n]
    n, m = a.shape[0], b.shape[0]
    def _pd(x, y): return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    d_aa, d_bb, d_ab = _pd(a, a), _pd(b, b), _pd(a, b)

    if bandwidths is None and use_median_heuristic:
        # Median of the pooled pairwise distances (a classical choice).
        combined = torch.cat([d_aa.flatten(), d_bb.flatten(), d_ab.flatten()])
        med = combined.median().clamp_min(1e-12).sqrt().item()
        bandwidths = (0.5 * med, med, 2.0 * med)
    elif bandwidths is None:
        bandwidths = (0.3, 1.0, 3.0)

    mmd2 = 0.0
    for h in bandwidths:
        k_aa = torch.exp(-d_aa / (2.0 * h * h)).mean()  # V-statistic (includes diag)
        k_bb = torch.exp(-d_bb / (2.0 * h * h)).mean()
        k_ab = torch.exp(-d_ab / (2.0 * h * h)).mean()
        mmd2 = mmd2 + (k_aa + k_bb - 2.0 * k_ab).item()
    return max(mmd2 / len(bandwidths), 0.0)


def moment_errors(a, b):
    """Absolute differences in first four moments. Works for 1D or 2D samples.
    For 2D, per-dimension moments are aggregated as L2-norm of the per-dim
    difference vector (so the output is a scalar per moment, comparable across
    1D and 2D targets). Definitions below match the README moment section.

    Returns dict with keys 'mean', 'std', 'skew', 'kurt':
      mean: | E_a[X] - E_b[X] |      (|.| is L2 norm if vector-valued)
      std:  | std_a(X) - std_b(X) |  (per-dim std, then L2)
      skew: | skew_a - skew_b |      (Fisher-Pearson standardized third moment)
      kurt: | kurt_a - kurt_b |      (excess kurtosis, i.e. fourth moment - 3)
    """
    def _moments(x):
        x = x.reshape(x.shape[0], -1).double()
        mean = x.mean(dim=0)                                          # (D,)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-12)           # (D,)
        z = (x - mean) / std                                          # standardized
        skew = (z ** 3).mean(dim=0)                                   # per-dim
        kurt = (z ** 4).mean(dim=0) - 3.0                             # excess
        return mean, std, skew, kurt

    ma, sa, ka, ku_a = _moments(a)
    mb, sb, kb, ku_b = _moments(b)
    def _l2(v): return float(v.norm().item())
    return {
        'mean': _l2(ma - mb),
        'std':  _l2(sa - sb),
        'skew': _l2(ka - kb),
        'kurt': _l2(ku_a - ku_b),
    }


def kl_analytic_1d(samples, density_fn, x_lo=None, x_hi=None, n_grid=400, kde_h=None):
    """Estimate KL(samples_empirical || density_fn) via a smoothed-histogram / KDE.
    density_fn(x_grid) -> analytic target density on grid.
    """
    s = samples.reshape(-1).float()
    n = s.numel()
    if x_lo is None: x_lo = s.min().item() - 1.0
    if x_hi is None: x_hi = s.max().item() + 1.0
    grid = torch.linspace(x_lo, x_hi, n_grid, device=s.device)
    dx = (x_hi - x_lo) / (n_grid - 1)
    # Gaussian KDE
    if kde_h is None:
        # Silverman's rule-of-thumb for 1D
        kde_h = 1.06 * s.std().clamp_min(1e-3).item() * (n ** (-1.0 / 5.0))
    diffs = (grid.unsqueeze(1) - s.unsqueeze(0)) / kde_h  # (n_grid, n)
    phi = torch.exp(-0.5 * diffs.pow(2)) / (kde_h * math.sqrt(2.0 * math.pi))
    p_hat = phi.mean(dim=1).clamp_min(1e-12)  # (n_grid,)
    p_true = torch.as_tensor(density_fn(grid.cpu().numpy()), device=s.device,
                             dtype=torch.float32).clamp_min(1e-12)
    # Normalize both on this grid to remove boundary effects
    p_hat = p_hat / (p_hat.sum() * dx)
    p_true = p_true / (p_true.sum() * dx)
    kl = (p_hat * (p_hat.log() - p_true.log())).sum().item() * dx
    return max(kl, 0.0)


def path_kl_girsanov(bg_theta, bg_star, g_fn, target, n_mc=40000,
                     t_min=1e-3, t_max=1.0 - 1e-3, device='cpu', dtype=torch.float32):
    """bg_theta, bg_star: callables (x, t, *cond) -> tensor; composed drifts for same g."""
    t = torch.rand(n_mc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    z = torch.randn(n_mc, target.dim, device=device, dtype=dtype)
    if target.conditional:
        x1, y = target.sample_x1(n_mc)
        cond = (y,)
    else:
        x1 = target.sample_x1(n_mc)
        cond = ()
    # interpolant: x_t = t x_1 + (1-t) sqrt(t) z
    xt = t * x1 + (1.0 - t) * t.clamp_min(0.0).sqrt() * z
    bb = bg_theta(xt, t, *cond)
    bs = bg_star(xt, t, *cond)
    err2 = (bb - bs).pow(2).sum(dim=-1, keepdim=True)
    gt = g_fn(t)
    integrand = err2 / (gt.pow(2).clamp_min(1e-12))
    return 0.5 * (t_max - t_min) * integrand.mean().item()
