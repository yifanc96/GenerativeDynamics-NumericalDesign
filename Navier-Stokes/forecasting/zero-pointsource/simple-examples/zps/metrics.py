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
    a = a.reshape(-1, 1)[:max_n]
    b = b.reshape(-1, 1)[:max_n]
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
    """Return dict of absolute errors in mean, std, skew, excess kurtosis (1D)."""
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    out = {}
    out['mean'] = float(abs(a.mean() - b.mean()).item())
    out['std'] = float(abs(a.std(unbiased=False) - b.std(unbiased=False)).item())
    for k, name in [(3, 'skew'), (4, 'kurt')]:
        am, bm = a.mean(), b.mean()
        asig = a.std(unbiased=False).clamp_min(1e-8)
        bsig = b.std(unbiased=False).clamp_min(1e-8)
        mk_a = (((a - am) / asig) ** k).mean().item()
        mk_b = (((b - bm) / bsig) ** k).mean().item()
        if name == 'kurt':
            mk_a -= 3.0; mk_b -= 3.0  # excess kurtosis
        out[name] = float(abs(mk_a - mk_b))
    return out


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
