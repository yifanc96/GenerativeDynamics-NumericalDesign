"""Eq (3.3)/(3.9) of arXiv:2602.10989 for a general interpolant (beta, sigma).

    b^g_t(x) = b_t(x) + (g^2 - sigma^2) / (2 t sigma (beta_dot sigma - beta sigma_dot))
              * (beta b_t(x) - beta_dot x)

For the linear-linear schedule this reduces to:
    b^g_t = b_t + (g^2 - (1-t)^2) / (2 t (1-t)) * (t b_t - x)

For the square-linear (beta=t^2, sigma=1-t):
    denom = t(1-t) * (2t(1-t) + t^2) = t^2 (1-t)(2-t)
    b^g_t = b_t + (g^2 - (1-t)^2) / (2 t^2 (1-t)(2-t)) * (t^2 b_t - 2t x)
"""
import torch


def compose_drift(b_fn, g_fn, interpolant, eps=1e-8):
    """Return b_g(x, t, *cond) implementing eq (3.9) for the given interpolant."""
    def b_g(x, t, *cond):
        b = b_fn(x, t, *cond)
        gt = g_fn(t)
        while gt.dim() < b.dim():
            gt = gt.unsqueeze(-1)
        tb = t
        while tb.dim() < b.dim():
            tb = tb.unsqueeze(-1)
        beta = interpolant.beta(tb)
        beta_dot = interpolant.beta_dot(tb)
        sigma = interpolant.sigma(tb)
        sigma_dot = interpolant.sigma_dot(tb)
        # denominator: 2 t sigma (beta_dot sigma - beta sigma_dot)
        mixing = beta_dot * sigma - beta * sigma_dot
        denom = (2.0 * tb * sigma * mixing).clamp_min(eps)
        alpha = (gt * gt - sigma * sigma) / denom
        return b + alpha * (beta * b - beta_dot * x)
    return b_g


def follmer_drift_sanity(b_fn, interpolant):
    """For linear-linear: b^{gF}(x) = (1+t) b_t - x  (eq 3.31)."""
    def b_gF(x, t, *cond):
        b = b_fn(x, t, *cond)
        tb = t
        while tb.dim() < b.dim():
            tb = tb.unsqueeze(-1)
        return (1.0 + tb) * b - x
    return b_gF
