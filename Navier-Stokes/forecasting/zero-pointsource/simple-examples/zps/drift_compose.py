"""Closed-form drift composition from eq. (3.9) of arXiv:2602.10989 for the
linear-linear schedule (beta_t = t, sigma_t = 1 - t):

    b^g_t(x) = b_t(x) + [ (g_t^2 - (1-t)^2) / (2 t (1-t)) ] * ( t b_t(x) - x ).

Equivalently
    b^g_t(x) = (1 + corr) b_t(x) - (corr / t) x    with    corr = (g_t^2 - sigma_t^2) / (2 sigma_t).

The composition is amplitude-sensitive, so g(t) is evaluated at its natural
amplitude (no shared epsilon normalization).

`b_fn` must have signature (x, t, *cond) -> tensor same shape as x.
"""
import torch


def compose_drift(b_fn, g_fn, eps=1e-8):
    """Return a callable b_g(x, t, *cond) implementing eq. (3.9)."""
    def b_g(x, t, *cond):
        b = b_fn(x, t, *cond)
        gt = g_fn(t)
        while gt.dim() < b.dim():
            gt = gt.unsqueeze(-1)
        sigma = 1.0 - t
        while sigma.dim() < b.dim():
            sigma = sigma.unsqueeze(-1)
        t_b = t
        while t_b.dim() < b.dim():
            t_b = t_b.unsqueeze(-1)
        denom = (2.0 * t_b * sigma).clamp_min(eps)
        alpha = (gt * gt - sigma * sigma) / denom       # coefficient of (t b - x)
        return b + alpha * (t_b * b - x)
    return b_g


def follmer_drift_sanity(b_fn):
    """Returns a callable implementing the closed-form Föllmer drift (eq. 3.31):
        b^{g_F}(x) = (1 + t) b_t(x) - x.
    Useful for numerical sanity-checking compose_drift against eq. (3.30).
    """
    def b_gF(x, t, *cond):
        b = b_fn(x, t, *cond)
        tb = t
        while tb.dim() < b.dim():
            tb = tb.unsqueeze(-1)
        return (1.0 + tb) * b - x
    return b_gF
