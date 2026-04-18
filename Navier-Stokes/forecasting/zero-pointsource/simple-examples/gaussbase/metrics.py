"""Statistical metrics: W1/W2 (1D closed form), MMD-RBF, moment errors, KDE-KL,
and the Girsanov path-KL estimator.
"""
import math
import torch


def wasserstein_1d(a, b, p=2):
    """Exact p-Wasserstein distance between two 1D empirical distributions of
    equal sample count via sorted-diff."""
    assert a.numel() == b.numel(), "need equal sizes; use interp or subsample first"
    a_sorted, _ = a.reshape(-1).sort()
    b_sorted, _ = b.reshape(-1).sort()
    diff = (a_sorted - b_sorted).abs()
    if p == 1:
        return diff.mean().item()
    return (diff.pow(p).mean().pow(1.0 / p)).item()


def wasserstein_1d_unequal(a, b, n_q=4096, p=2):
    """Approximate p-W via quantile matching; robust to unequal sample sizes."""
    q = torch.linspace(0.0, 1.0, n_q + 2, device=a.device)[1:-1]
    qa = torch.quantile(a.reshape(-1).float(), q)
    qb = torch.quantile(b.reshape(-1).float(), q)
    diff = (qa - qb).abs()
    if p == 1:
        return diff.mean().item()
    return (diff.pow(p).mean().pow(1.0 / p)).item()


def mmd_rbf(a, b, bandwidths=(0.1, 0.5, 1.0, 2.0, 5.0)):
    """Unbiased MMD^2 with a mixture of RBF kernels. Both (n, D)."""
    def _rbf(xx, h):
        # xx: pairwise squared distances
        return torch.exp(-xx / (2.0 * h * h))

    def _pd(x, y):
        return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)

    d_aa = _pd(a, a)
    d_bb = _pd(b, b)
    d_ab = _pd(a, b)
    n, m = a.shape[0], b.shape[0]
    mmd2 = 0.0
    for h in bandwidths:
        k_aa = _rbf(d_aa, h)
        k_bb = _rbf(d_bb, h)
        k_ab = _rbf(d_ab, h)
        # unbiased estimator (subtract diagonal for aa, bb)
        k_aa = (k_aa.sum() - k_aa.diag().sum()) / (n * (n - 1))
        k_bb = (k_bb.sum() - k_bb.diag().sum()) / (m * (m - 1))
        k_ab = k_ab.mean()
        mmd2 = mmd2 + (k_aa + k_bb - 2.0 * k_ab).item()
    return mmd2 / len(bandwidths)


def moment_errors(a, b, k=4):
    """Absolute differences of first k central moments (1D only)."""
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    out = {}
    out['mean'] = abs(a.mean().item() - b.mean().item())
    out['var'] = abs(a.var(unbiased=False).item() - b.var(unbiased=False).item())
    am, bm = a.mean(), b.mean()
    for j in range(3, k + 1):
        mj_a = ((a - am) ** j).mean().item()
        mj_b = ((b - bm) ** j).mean().item()
        out[f'cm{j}'] = abs(mj_a - mj_b)
    return out


def path_kl_girsanov(b_fn, b_star_fn, g_fn, interpolant, target, n_mc=20000,
                    t_min=0.0, t_max=1.0, device='cpu', dtype=torch.float32):
    """Monte-Carlo estimate of the Girsanov path-KL:
        KL(P || P*) = 0.5 E_t E_{X_t} [ g(t)^{-2} || b(X_t, t) - b*(X_t, t) ||^2 ].
    X_t is drawn from the interpolant I_t = beta_t X_1 + gamma_t Z, t ~ U(t_min, t_max).

    b_fn and b_star_fn share a signature (x, t, *cond); cond for conditional targets.
    """
    t = torch.rand(n_mc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    z = torch.randn(n_mc, target.dim, device=device, dtype=dtype)
    if target.conditional:
        x1, y = target.sample_x1(n_mc)
        cond = (y,)
    else:
        x1 = target.sample_x1(n_mc)
        cond = ()
    xt = interpolant.It(x1, z, t)
    bb = b_fn(xt, t, *cond)
    bs = b_star_fn(xt, t, *cond)
    err2 = (bb - bs).pow(2).sum(dim=-1, keepdim=True)  # (n_mc, 1)
    gt = g_fn(t)  # (n_mc, 1)
    integrand = err2 / (gt.pow(2).clamp_min(1e-12))
    kl = 0.5 * (t_max - t_min) * integrand.mean().item()
    return kl


def sample_target(target, n, device='cpu'):
    """Draw n samples from target for marginal comparison. For conditional targets
    we marginalize by also drawing the cond from its stationary distribution."""
    if target.conditional:
        x1, _ = target.sample_x1(n)
        return x1
    return target.sample_x1(n)
