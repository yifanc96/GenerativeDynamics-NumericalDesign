"""Toy targets with analytic v_star, s_star under any interpolant I_t = beta_t X_1 + gamma_t Z.

All three targets are (conditional) mixtures of Gaussians, so the interpolant
marginal p_t is also a (conditional) Gaussian mixture with:
  - component mean:  m_k(t) = beta_t * mu_k
  - component var:   V_k(t) = beta_t^2 * sigma_k^2 + gamma_t^2
The score and ODE velocity follow by direct computation.

API:
  .sample_x1(n)                     -> (n, D) target samples (or (x1, cond) for conditional)
  .sample_cond(n)                   -> tuple of conditioning tensors (empty if unconditional)
  .v_star(x, t, ip, *cond)          -> (B, D) ODE drift
  .s_star(x, t, ip, *cond)          -> (B, D) score
"""
import math
import torch


def _gaussian_component_scoring(x, t, ip, mu, sig2):
    """Given a single Gaussian target N(mu, sig2), compute:
      m(t) = beta_t * mu,  V(t) = beta_t^2 sig2 + gamma_t^2
      v_component = beta_dot * mu + (dV/(2V)) * (x - m)
      s_component = -(x - m) / V
    Returns (V, v_component, s_component, m).
    """
    beta = ip.beta(t)
    beta_dot = ip.beta_dot(t)
    gamma = ip.gamma(t)
    gamma_dot = ip.gamma_dot(t)
    m = beta * mu
    V = beta ** 2 * sig2 + gamma ** 2
    dV = 2.0 * beta * beta_dot * sig2 + 2.0 * gamma * gamma_dot
    Vc = V.clamp_min(1e-10)
    v = beta_dot * mu + (dV / (2.0 * Vc)) * (x - m)
    s = -(x - m) / Vc
    return V, v, s, m


class Gaussian1D:
    """Target A: X_1 ~ N(mu, sigma^2), 1D."""
    name = 'gaussian1d'
    dim = 1
    conditional = False

    def __init__(self, mu=1.5, sigma=0.5, device='cpu'):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sig2 = self.sigma ** 2
        self.device = device

    def sample_x1(self, n):
        return self.mu + self.sigma * torch.randn(n, 1, device=self.device)

    def sample_cond(self, n):
        return ()

    def v_star(self, x, t, ip):
        _, v, _, _ = _gaussian_component_scoring(x, t, ip, self.mu, self.sig2)
        return v

    def s_star(self, x, t, ip):
        _, _, s, _ = _gaussian_component_scoring(x, t, ip, self.mu, self.sig2)
        return s


class Bimodal1D:
    """Target B: X_1 ~ 1/2 N(-m, tau^2) + 1/2 N(m, tau^2), 1D."""
    name = 'bimodal1d'
    dim = 1
    conditional = False

    def __init__(self, m=1.0, tau=0.3, device='cpu'):
        self.m = float(m)
        self.tau = float(tau)
        self.tau2 = self.tau ** 2
        self.device = device

    def sample_x1(self, n):
        z = torch.randn(n, 1, device=self.device) * self.tau
        signs = (torch.randint(0, 2, (n, 1), device=self.device) * 2 - 1).to(z.dtype)
        return z + self.m * signs

    def sample_cond(self, n):
        return ()

    def _per_component(self, x, t, ip):
        Vp, vp, sp, mp = _gaussian_component_scoring(x, t, ip, +self.m, self.tau2)
        Vn, vn, sn, mn = _gaussian_component_scoring(x, t, ip, -self.m, self.tau2)
        # responsibility (equal weights, equal V since both have same sig2)
        log_p = -0.5 * (x - mp) ** 2 / Vp.clamp_min(1e-10)
        log_n = -0.5 * (x - mn) ** 2 / Vn.clamp_min(1e-10)
        w_p = torch.sigmoid(log_p - log_n)  # (B, 1)
        return w_p, vp, vn, sp, sn

    def v_star(self, x, t, ip):
        w_p, vp, vn, _, _ = self._per_component(x, t, ip)
        return w_p * vp + (1.0 - w_p) * vn

    def s_star(self, x, t, ip):
        w_p, _, _, sp, sn = self._per_component(x, t, ip)
        return w_p * sp + (1.0 - w_p) * sn


class OUForecast:
    """Target C (conditional): X_1 = Y_{s+tau} - Y_s given Y_s=y under OU
    dY = -lam Y ds + sigma dW. Conditional on y:
      X_1 | y ~ N( (exp(-lam*tau) - 1) y,  sigma^2 (1 - exp(-2 lam tau)) / (2 lam) ).
    Stationary Y_s ~ N(0, sigma^2/(2 lam)).
    """
    name = 'ou_forecast'
    dim = 1
    conditional = True

    def __init__(self, lam=1.0, sigma=1.0, tau=0.5, device='cpu'):
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.device = device
        self.decay = math.exp(-self.lam * self.tau)
        self.cond_sig2 = (self.sigma ** 2) * (1.0 - math.exp(-2.0 * self.lam * self.tau)) / (2.0 * self.lam)
        self.stationary_sig2 = (self.sigma ** 2) / (2.0 * self.lam)

    def sample_cond(self, n):
        y = math.sqrt(self.stationary_sig2) * torch.randn(n, 1, device=self.device)
        return (y,)

    def sample_x1(self, n, y=None):
        if y is None:
            (y,) = self.sample_cond(n)
        mu = (self.decay - 1.0) * y
        return mu + math.sqrt(self.cond_sig2) * torch.randn_like(y), y

    def v_star(self, x, t, ip, y):
        mu = (self.decay - 1.0) * y
        _, v, _, _ = _gaussian_component_scoring(x, t, ip, mu, self.cond_sig2)
        return v

    def s_star(self, x, t, ip, y):
        mu = (self.decay - 1.0) * y
        _, _, s, _ = _gaussian_component_scoring(x, t, ip, mu, self.cond_sig2)
        return s


TARGETS = {
    'gaussian1d': Gaussian1D,
    'bimodal1d':  Bimodal1D,
    'ou_forecast': OUForecast,
}


def make_target(name, device='cpu', **kwargs):
    return TARGETS[name](device=device, **kwargs)
