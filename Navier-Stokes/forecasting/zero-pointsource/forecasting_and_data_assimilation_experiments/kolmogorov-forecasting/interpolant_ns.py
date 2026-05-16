"""Parametrised §3.4.1 / §3.4.x ZPS interpolants. I_s = beta_s x_1 + (1-s) sqrt(s) z.

Two variants shipped:
  linear-linear:  beta_s = s,     sigma_s = 1-s      (paper §3.4.1)
  square-linear:  beta_s = s^2,   sigma_s = 1-s      (ablation; matches the Albergo-style
                                                       beta = s^2 choice for forecasting)

Convention: t has shape (B, 1, ..., 1) after _widen; broadcasts with (B, 1, H, W) fields.
"""
import torch


def _widen(t, x):
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t


class LinearLinearInterpolant:
    """beta_s = s, sigma_s = 1-s. Paper §3.4.1."""
    name = 'linlin'

    def beta(self, t): return t
    def beta_dot(self, t): return torch.ones_like(t)
    def sigma(self, t): return 1.0 - t
    def sigma_dot(self, t): return -torch.ones_like(t)

    def gamma(self, t):
        return (1.0 - t) * t.clamp_min(0.0).sqrt()

    def It(self, x1, z, t):
        tt = _widen(t, x1)
        return tt * x1 + (1.0 - tt) * tt.clamp_min(0.0).sqrt() * z

    def Rb(self, x1, z, t):
        """Regression target for baseline drift b_t = E[beta_dot x_1 + sigma_dot sqrt(t) z | x_t=x]."""
        tt = _widen(t, x1)
        return x1 - tt.clamp_min(0.0).sqrt() * z

    def follmer_g(self, t):
        """g^F_t = sqrt(1 - t^2)  (eq 3.30)."""
        return (1.0 - t * t).clamp_min(0.0).sqrt()

    def tweedie_x1(self, x, s, b_hat):
        """E[x1 | x_s = x] = x + (1-s) * b_hat for beta=s, sigma=1-s."""
        ss = _widen(s, x)
        return x + (1.0 - ss) * b_hat

    def C_g(self, t, g_val):
        """"Natural" score-coefficient C_g(t) such that b^g = x/t + C_g * ∇log p_t.
        Derived by expanding b_t = x/t + (1-t) ∇log p_t then substituting into b^g formula.
        For linlin: C_g(t) = (g^2 + 1 - t^2)/2. C_g = g^2 iff Föllmer."""
        return 0.5 * (g_val * g_val + 1.0 - t * t)


class SquareLinearInterpolant:
    """beta_s = s^2, sigma_s = 1-s. 'Beta-squared' variant; matches Albergo-style forecasting choice."""
    name = 'sqlin'

    def beta(self, t): return t * t
    def beta_dot(self, t): return 2.0 * t
    def sigma(self, t): return 1.0 - t
    def sigma_dot(self, t): return -torch.ones_like(t)

    def gamma(self, t):
        return (1.0 - t) * t.clamp_min(0.0).sqrt()

    def It(self, x1, z, t):
        tt = _widen(t, x1)
        return tt * tt * x1 + (1.0 - tt) * tt.clamp_min(0.0).sqrt() * z

    def Rb(self, x1, z, t):
        """R_b = 2t x_1 - sqrt(t) z."""
        tt = _widen(t, x1)
        return 2.0 * tt * x1 - tt.clamp_min(0.0).sqrt() * z

    def follmer_g(self, t):
        """g^F_t = sqrt((1-t)(3-t))  derived from eq 3.10 for beta=t^2, sigma=1-t.
        At t=0: sqrt(3) ≈ 1.73; at t=1: 0."""
        return ((1.0 - t) * (3.0 - t)).clamp_min(0.0).sqrt()

    def tweedie_x1(self, x, s, b_hat):
        """E[x1 | x_s = x] = (x + (1-s) b_hat)/(s(2-s)) for beta=s^2, sigma=1-s.
        Derived from b_s = 2s x_1 - sqrt(s) z and x_s = s^2 x_1 + (1-s) sqrt(s) z."""
        ss = _widen(s, x)
        return (x + (1.0 - ss) * b_hat) / (ss * (2.0 - ss)).clamp_min(1e-6)

    def C_g(self, t, g_val):
        """C_g^sqlin(t) = ((1-t)(3-t) + g^2)/2 such that b^g = 2x/t + C_g * ∇log p_t.
        Key identity: C_g = g^2 iff g^2 = (1-t)(3-t), i.e. iff Föllmer (same property as linlin)."""
        return 0.5 * ((1.0 - t) * (3.0 - t) + g_val * g_val)


# Back-compat alias (matches the original ZPSInterpolant name)
class ZPSInterpolant(LinearLinearInterpolant):
    pass


INTERPOLANTS = {
    'linlin': LinearLinearInterpolant,
    'sqlin':  SquareLinearInterpolant,
}
