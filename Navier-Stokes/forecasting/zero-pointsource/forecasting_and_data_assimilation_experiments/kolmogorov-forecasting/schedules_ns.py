"""g(t) registry. `follmer` is interpolant-dependent (uses ip.follmer_g); the rest
are fixed shapes that work for any interpolant.
"""
import torch


_SHAPES = {
    'baseline':  lambda t: 1.0 - t,
    'triangle':  lambda t: (t * (1.0 - t)).clamp_min(0.0).sqrt(),
    'const':     lambda t: torch.ones_like(t),
    'sqrt_t':    lambda t: t.clamp_min(0.0).sqrt(),
    'zero':      lambda t: torch.zeros_like(t),
    # 'follmer' handled specially via the interpolant
}


def list_schedules():
    return ['follmer', 'baseline', 'triangle', 'const', 'sqrt_t', 'zero']


def make_g(name, scale=1.0, interpolant=None):
    if name == 'follmer':
        assert interpolant is not None, "follmer requires interpolant"
        shape = interpolant.follmer_g
    else:
        shape = _SHAPES[name]

    def g_fn(t):
        return scale * shape(t)
    g_fn.name = name
    g_fn.scale = scale
    return g_fn
