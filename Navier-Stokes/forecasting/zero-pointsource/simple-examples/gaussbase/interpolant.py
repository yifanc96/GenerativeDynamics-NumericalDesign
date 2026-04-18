"""Stochastic interpolant with **Gaussian base** X_0 ~ N(0, I):
    I_t = beta_t X_1 + gamma_t Z,  X_0 = Z ~ N(0, I).

Default: beta_t = t, gamma_t = 1 - t. Then I_0 = Z and I_1 = X_1.

All intermediate quantities are smooth across t in [0, 1], so v^*, s^* are
bounded and training a single (v_theta, s_theta) pair lets us sweep the
diffusion coefficient g(t) at sample time via b^(g) = v + (g^2/2) s.

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


class GaussianBaseInterpolant:
    """I_t = t X_1 + (1 - t) Z."""
    base = 'gaussian'

    def beta(self, t):
        return t

    def beta_dot(self, t):
        return torch.ones_like(t)

    def gamma(self, t):
        return 1.0 - t

    def gamma_dot(self, t):
        return -torch.ones_like(t)

    def It(self, x1, z, t):
        return self.beta(t) * x1 + self.gamma(t) * z

    def Rv(self, x1, z, t):
        """Velocity target: dot{I}_t. E[Rv | I_t=x] = v^*(x,t)."""
        return self.beta_dot(t) * x1 + self.gamma_dot(t) * z

    def Rs(self, x1, z, t, eps=1e-8):
        """Score target: -z / gamma_t. E[Rs | I_t=x] = s^*(x,t)."""
        g = self.gamma(t).clamp_min(eps)
        return -z / g
