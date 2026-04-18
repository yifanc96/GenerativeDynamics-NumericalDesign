"""Diffusion coefficient registry g(t) and drift composition.

Each schedule is a shape g_shape(t). We rescale so that int_0^1 g(u)^2 du = epsilon
(a shared noise-budget knob), giving g(t) = sqrt(epsilon / integral) * g_shape(t).

SDE drift is composed as b^(g)_t(x) = v_t(x) + (g(t)^2 / 2) * s_t(x), where
v and s are the interpolant velocity and score, respectively.
"""
import math
import torch

_SHAPES = {
    'follmer':    lambda t: (1.0 - t).clamp_min(0.0).sqrt(),     # predicted optimum
    'const':      lambda t: torch.ones_like(t),
    'lin_decay':  lambda t: (1.0 - t).clamp_min(0.0),
    'sqrt_t':     lambda t: t.clamp_min(0.0).sqrt(),
    'triangle':   lambda t: (t * (1.0 - t)).clamp_min(0.0).sqrt(),
    'ode':        lambda t: torch.zeros_like(t),
}


def list_schedules():
    return list(_SHAPES.keys())


def _integrate_g2(shape_fn, n=4096, device='cpu', dtype=torch.float64):
    ts = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    g = shape_fn(ts)
    # trapezoidal rule
    g2 = g * g
    return torch.trapz(g2, ts).item()


def make_g(name, epsilon=1.0, device='cpu', dtype=torch.float32):
    """Return a callable g(t) normalized so int g^2 = epsilon. For 'ode', g == 0."""
    shape = _SHAPES[name]
    if name == 'ode':
        return lambda t: torch.zeros_like(t)
    integral = _integrate_g2(shape, device=device)
    scale = math.sqrt(epsilon / integral)

    def g_fn(t):
        return scale * shape(t.to(dtype=dtype))
    g_fn.integral_g2 = epsilon
    g_fn.name = name
    return g_fn


def compose_drift(v_fn, s_fn, g_fn):
    """Return b^(g)(x, t) = v(x, t) + 0.5 * g(t)^2 * s(x, t). v_fn, s_fn take (x, t, *cond)."""
    def b_fn(x, t, *cond):
        v = v_fn(x, t, *cond)
        s = s_fn(x, t, *cond)
        gt = g_fn(t)
        # broadcast gt over x: t has shape (B,), x has shape (B, D) or (B,)
        while gt.dim() < v.dim():
            gt = gt.unsqueeze(-1)
        return v + 0.5 * gt * gt * s
    return b_fn
