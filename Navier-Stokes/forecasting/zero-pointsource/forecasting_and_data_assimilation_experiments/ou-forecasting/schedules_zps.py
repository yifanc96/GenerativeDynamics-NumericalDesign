"""Diffusion coefficient registry for the §3.4.1 study.

Each schedule is a callable g(t) -> tensor same shape as t. Schedules are NOT
renormalized by a shared epsilon budget (unlike the gaussbase folder): the
Föllmer choice g_F = sqrt(1 - t^2) has a specific amplitude set by the paper's
variational argument, and we want to compare schedules at their natural
amplitude. A single `scale` multiplier is optionally provided.

The drift-correction in eq. (3.9) is amplitude-sensitive: with scale=1, the
"baseline" schedule g = sigma_t = 1 - t recovers the paper's baseline
generative diffusion exactly.
"""
import torch


_SHAPES = {
    'follmer':   lambda t: (1.0 - t * t).clamp_min(0.0).sqrt(),   # sqrt(1 - t^2)
    'baseline':  lambda t: 1.0 - t,                                # sigma_t
    'const':     lambda t: torch.ones_like(t),
    'sqrt_t':    lambda t: t.clamp_min(0.0).sqrt(),
    'triangle':  lambda t: (t * (1.0 - t)).clamp_min(0.0).sqrt(),
    'zero':      lambda t: torch.zeros_like(t),                    # ODE baseline
}


def list_schedules():
    return list(_SHAPES.keys())


def make_g(name, scale=1.0):
    shape = _SHAPES[name]

    def g_fn(t):
        return scale * shape(t)
    g_fn.name = name
    g_fn.scale = scale
    return g_fn
