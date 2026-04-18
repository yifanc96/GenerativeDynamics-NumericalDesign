"""Toy 1D targets with analytic **baseline drift** b_star(x, t) under the
§3.4.1 interpolant x_t = t x_star + (1 - t) sqrt(t) z, i.e.

    b_star(x, t) = E[ x_star - sqrt(t) z | x_t = x ].

For the Gaussian component N(mu, sig2),
    m(t)    = t mu,                     V(t) = t^2 sig2 + (1 - t)^2 t = t [t sig2 + (1-t)^2]
    E[x_star | x_t = x] = mu + (t sig2  / V) (x - m),
    E[z      | x_t = x] = ((1-t) sqrt(t) / V) (x - m),
so
    b_star_comp(x, t) = mu + (sig2 - 1 + t) / (t sig2 + (1-t)^2) * (x - m).

Mixture / conditional targets marginalize/weight this form.

API for each target:
    .sample_x1(n) -> (n, D) or (x1, y) for conditional
    .sample_cond(n) -> () or (y,)
    .b_star(x, t, *cond) -> (B, D)
"""
import math
import numpy as np
import torch


def _gaussian_comp_bstar(x, t, mu, sig2):
    """Analytic b_star for a single Gaussian component N(mu, sig2).
    Returns (b, m, V) where m = t*mu, V = t^2 sig2 + (1-t)^2 t (helpful for mixtures).
    """
    m = t * mu
    V = t * t * sig2 + (1.0 - t) ** 2 * t
    denom = (t * sig2 + (1.0 - t) ** 2)
    # b = mu + (sig2 - 1 + t) / denom * (x - m)
    b = mu + (sig2 - 1.0 + t) / denom.clamp_min(1e-10) * (x - m)
    return b, m, V


class Gaussian1D:
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

    def b_star(self, x, t):
        b, _, _ = _gaussian_comp_bstar(x, t, self.mu, self.sig2)
        return b

    def density(self, x):
        x = np.asarray(x)
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * math.sqrt(2.0 * math.pi))


class Bimodal1D:
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

    def b_star(self, x, t):
        bp, mp, V = _gaussian_comp_bstar(x, t,  self.m, self.tau2)
        bn, mn, _ = _gaussian_comp_bstar(x, t, -self.m, self.tau2)
        log_p = -0.5 * (x - mp) ** 2 / V.clamp_min(1e-10)
        log_n = -0.5 * (x - mn) ** 2 / V.clamp_min(1e-10)
        w_p = torch.sigmoid(log_p - log_n)
        return w_p * bp + (1.0 - w_p) * bn

    def density(self, x):
        x = np.asarray(x)
        tau = self.tau
        c = 1.0 / (tau * math.sqrt(2.0 * math.pi))
        p_plus = c * np.exp(-0.5 * ((x - self.m) / tau) ** 2)
        p_minus = c * np.exp(-0.5 * ((x + self.m) / tau) ** 2)
        return 0.5 * (p_plus + p_minus)


class OUForecast:
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

    def b_star(self, x, t, y):
        mu = (self.decay - 1.0) * y
        b, _, _ = _gaussian_comp_bstar(x, t, mu, self.cond_sig2)
        return b

    def marginal_var(self):
        """Unconditional var of X_1 when y ~ stationary: (decay-1)^2 stat_sig2 + cond_sig2."""
        return (self.decay - 1.0) ** 2 * self.stationary_sig2 + self.cond_sig2

    def density(self, x):
        """Unconditional density of X_1 (marginalized over stationary Y_s): Gaussian N(0, marginal_var)."""
        x = np.asarray(x)
        v = self.marginal_var()
        return np.exp(-0.5 * x ** 2 / v) / math.sqrt(2.0 * math.pi * v)


TARGETS = {
    'gaussian1d': Gaussian1D,
    'bimodal1d':  Bimodal1D,
    'ou_forecast': OUForecast,
}


def make_target(name, device='cpu', **kwargs):
    return TARGETS[name](device=device, **kwargs)
