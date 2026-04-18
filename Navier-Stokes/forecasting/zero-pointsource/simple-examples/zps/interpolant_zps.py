"""Point-source stochastic interpolant, linear-linear schedule (arXiv:2602.10989 §3.4.1).

beta_t = t, sigma_t = 1 - t. In law at each t:
    x_t = beta_t x_star + sigma_t sqrt(t) z,   z ~ N(0, I).

I_0 = 0 (Dirac), I_1 = x_star. gamma_t = sigma_t sqrt(t) = (1 - t) sqrt(t).

Convention: t has shape (B, 1).
"""
import torch


def canon_t(t, like=None):
    if t.dim() == 0:
        t = t.reshape(1, 1)
    elif t.dim() == 1:
        t = t.unsqueeze(-1)
    if like is not None:
        t = t.to(dtype=like.dtype, device=like.device)
    return t


class ZPSInterpolant:
    """beta_t = t, sigma_t = 1 - t, gamma_t = (1 - t) sqrt(t)."""
    base = 'dirac'

    def beta(self, t):
        return t

    def beta_dot(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        return 1.0 - t

    def sigma_dot(self, t):
        return -torch.ones_like(t)

    def gamma(self, t):
        return (1.0 - t) * t.clamp_min(0.0).sqrt()

    def It(self, x1, z, t):
        """x_t = t x_1 + (1-t) sqrt(t) z."""
        return self.beta(t) * x1 + self.sigma(t) * t.clamp_min(0.0).sqrt() * z

    def Rb(self, x1, z, t):
        """DSM regression target for the baseline drift b_t:
            b_t(x) = E[ dot{beta}_t x_1 + dot{sigma}_t sqrt(t) z | x_t = x ]
                   = E[ x_1 - sqrt(t) z | x_t = x ].
        Target is bounded on [0, 1] (|sqrt(t)| <= 1).
        """
        return self.beta_dot(t) * x1 + self.sigma_dot(t) * t.clamp_min(0.0).sqrt() * z
