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
    cond_dim = 1

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


class GMM2DForecast:
    """Target D: 2D GMM jump-forecast, ZPS version of arXiv:2403.13724 §4.1.

    Simplification over the paper: *pure* Poisson-mod-5 rotation + isotropic
    Gaussian noise (no between-jump Langevin), so that the conditional
    X_1|y (where X_1 = x_{t+tau} - y is the difference = interpolant target,
    and y = x_t is the current state) is exactly a 5-mode GMM with

        X_1 | y  ~  sum_k pi_k  N( (R^k - I) y,  sigma_noise^2 I_2 ),
        pi_k = P(Poisson(lam_tau) mod 5 = k).

    The equilibrium x_0 distribution (for drawing y) is also a 5-mode GMM
    (equal weights, per-mode covariance C0 rotated).
    """
    name = 'gmm2d_forecast'
    dim = 2
    conditional = True
    cond_dim = 2

    def __init__(self, m0=5.0, cov_axis=(1.5, 0.1), sigma_noise=0.5,
                 lam_tau=1.0, device='cpu', n_series=60):
        self.m0 = float(m0)
        self.sigma_noise = float(sigma_noise)
        self.sig2 = self.sigma_noise ** 2
        self.lam_tau = float(lam_tau)
        self.device = device
        angle = 2.0 * math.pi / 5.0
        c, s = math.cos(angle), math.sin(angle)
        R = torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)
        # Precompute R^k for k = 0..4 -> (5, 2, 2)
        R_pow = [torch.eye(2, device=device, dtype=torch.float32)]
        for _ in range(4):
            R_pow.append(R @ R_pow[-1])
        self.R_pow = torch.stack(R_pow, dim=0)
        # (R^k - I), used as component mean multiplier on the conditioning y
        self.R_minus_I = self.R_pow - torch.eye(2, device=device).unsqueeze(0)  # (5, 2, 2)
        # Equilibrium GMM (for sampling x_0 = y)
        center0 = torch.tensor([m0, 0.0], device=device, dtype=torch.float32)
        self.mode_means = torch.einsum('kij,j->ki', self.R_pow, center0)          # (5, 2)
        cov0 = torch.diag(torch.tensor(cov_axis, device=device, dtype=torch.float32))
        mode_covs = torch.stack([self.R_pow[k] @ cov0 @ self.R_pow[k].T for k in range(5)], dim=0)
        self.mode_covs = mode_covs
        self.mode_L = torch.stack([torch.linalg.cholesky(C) for C in mode_covs], dim=0)
        # Poisson-mod-5 weights: pi[k] = sum_{j>=0} exp(-lam_tau) lam_tau^(5j+k) / (5j+k)!
        pi = [0.0] * 5
        for j in range(n_series):
            for k in range(5):
                n = 5 * j + k
                log_term = -self.lam_tau + n * math.log(max(self.lam_tau, 1e-12)) - math.lgamma(n + 1)
                pi[k] += math.exp(log_term) if log_term > -50 else 0.0
        total = sum(pi)
        self.pi = torch.tensor([p / total for p in pi], device=device, dtype=torch.float32)

    def sample_cond(self, n):
        """Draw y ~ rho_GMM (equal-weight 5-mode)."""
        comp = torch.randint(0, 5, (n,), device=self.device)
        mu = self.mode_means[comp]
        L = self.mode_L[comp]
        z = torch.randn(n, 2, device=self.device)
        y = mu + torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
        return (y,)

    def sample_x1(self, n, y=None):
        """Return (x1_interpolant_target, y) where x1_interp = x_{t+tau} - y."""
        if y is None:
            (y,) = self.sample_cond(n)
        N = torch.multinomial(self.pi, n, replacement=True)                      # (n,)
        # (R^N - I) y, per-sample rotation
        RmI = self.R_minus_I[N]                                                   # (n, 2, 2)
        mean = torch.einsum('nij,nj->ni', RmI, y)
        eps = self.sigma_noise * torch.randn(n, 2, device=self.device)
        return mean + eps, y

    def b_star(self, x, t, y):
        """b_star(x, t, y) = E[x1 - sqrt(t) z | x_t = x, y] for the 5-mode GMM with cov sig2 I.
        x: (B, 2), t: (B, 1), y: (B, 2). Interpolant target x1 has component means mu_k = (R^k-I) y.
        """
        # Per-component mu_k = (R^k - I) y  -> (5, B, 2)
        mus = torch.einsum('kij,bj->kbi', self.R_minus_I, y)
        mk = t.unsqueeze(0) * mus                                   # (5, B, 2)
        Vc = (t * self.sig2 + (1.0 - t) ** 2).clamp_min(1e-10)      # (B, 1)  (== V / t for the posterior)
        coeff = (self.sig2 - 1.0 + t) / Vc                          # (B, 1)
        drifts = mus + coeff.unsqueeze(0) * (x.unsqueeze(0) - mk)   # (5, B, 2)
        # Responsibilities: w_k ∝ pi_k * N(x; t mu_k, V I_2), V = t(t sig2 + (1-t)^2) = t*Vc
        V = (t * Vc).clamp_min(1e-10)                               # (B, 1)
        diff = x.unsqueeze(0) - mk                                  # (5, B, 2)
        log_comp = -0.5 * diff.pow(2).sum(dim=-1) / V.squeeze(-1).unsqueeze(0)    # (5, B)
        log_w = log_comp + torch.log(self.pi).unsqueeze(-1)                       # (5, B)
        log_norm = torch.logsumexp(log_w, dim=0, keepdim=True)
        w = torch.exp(log_w - log_norm).unsqueeze(-1)                             # (5, B, 1)
        return (w * drifts).sum(dim=0)                                            # (B, 2)


class Bimodal1DSharp(Bimodal1D):
    """Sharper bimodal (mode separation 2, std 0.1): harder mode-selection
    so path-level errors translate more tightly into marginal errors."""
    name = 'bimodal1d_sharp'

    def __init__(self, m=2.0, tau=0.1, device='cpu'):
        super().__init__(m=m, tau=tau, device=device)


TARGETS = {
    'gaussian1d': Gaussian1D,
    'bimodal1d':  Bimodal1D,
    'bimodal1d_sharp': Bimodal1DSharp,
    'ou_forecast': OUForecast,
    'gmm2d_forecast': GMM2DForecast,
}


def make_target(name, device='cpu', **kwargs):
    return TARGETS[name](device=device, **kwargs)
