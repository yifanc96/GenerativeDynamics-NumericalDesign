"""1D posterior sampling via twisted SMC.

Setup:
  prior            p(x_1) = target distribution
  observation      y = A x_1 + eta,  eta ~ N(0, sigma_y^2)    (A = identity by default)
  posterior        p(x_1 | y)  -- analytical for Gaussian / GMM targets
  Tweedie twist    phi_s(x) = p(y | xhat1(x, s))
  xhat1(x, s)      = x + (1-s) * hat_b_theta(x, s)

Two proposal modes:
  'uncond'   -- use learned SDE as proposal, reweight by Tweedie ratios only
  'guided'   -- add g^2 * grad_x log phi_s to drift; Girsanov correction in weights
"""
import math
import torch


def ess(log_w):
    lw = log_w - log_w.max()
    w = lw.exp(); w = w / w.sum()
    return 1.0 / (w * w).sum().item()


def systematic_resample(log_w, generator=None):
    lw = log_w - log_w.max()
    w = lw.exp(); w = w / w.sum()
    N = w.numel()
    u0 = torch.rand(1, generator=generator, device=w.device).item() / N
    cdf = torch.cumsum(w, dim=0)
    us = u0 + torch.arange(N, device=w.device).float() / N
    idx = torch.searchsorted(cdf, us).clamp_max(N - 1)
    return idx


def log_lik_gauss_id(x1_hat, y, sigma_y):
    """-1/(2 sigma_y^2) * ||x1_hat - y||^2 elementwise reduced over dim D."""
    d = (x1_hat - y)
    return -0.5 * (d * d).sum(dim=-1) / (sigma_y ** 2)


def _tweedie_x1(b_fn, x, s_scalar, cond):
    """x1_hat(x, s) = x + (1-s) * b_theta(x, s, *cond)."""
    s_ = s_scalar
    if s_.dim() == 0:
        s_ = s_.reshape(1, 1).expand(x.shape[0], 1)
    b_hat = b_fn(x, s_, *cond)
    return x + (1.0 - s_) * b_hat


def twisted_smc_1d(b_theta_fn, compose_b_g_fn, g_fn, y_obs, sigma_y,
                   n_particles, dim, cond=(),
                   n_steps=200, t_min=1e-3, t_max=1.0 - 1e-3,
                   resample_thresh=0.5, proposal='guided',
                   guidance_type='doob', guidance_eta=1.0,
                   device='cpu', dtype=torch.float32, generator=None,
                   return_trace=False):
    """
    b_theta_fn:    (x, s, *cond) -> hat_b_theta
    compose_b_g_fn:(x, s, *cond) -> b^g  (schedule-specific drift via eq 3.9)
    g_fn:          (s,) -> g(s)
    y_obs:         (1, D)  observation
    sigma_y:       scalar

    Returns (x_final, log_w, logZ, trace).
    """
    ts = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)
    N = n_particles
    # Replicate cond across particles
    cond_p = tuple(c.expand(N, *c.shape[1:]).contiguous() for c in cond)
    y_obs_p = y_obs.expand(N, *y_obs.shape[1:]).contiguous()

    # Init: x ~ N(0, gamma(t_min)^2) (interpolant marginal)
    s0 = ts[0]
    g0 = float((1.0 - s0) * s0.sqrt())
    x = g0 * torch.randn(N, dim, device=device, dtype=dtype, generator=generator)

    log_w = torch.zeros(N, device=device, dtype=dtype)
    logZ_accum = 0.0

    trace = {'ess': [], 's': [], 'logZ': []} if return_trace else None

    def compute_phi_and_grad(xx, s_scalar):
        xx_req = xx.detach().requires_grad_(True)
        x1_hat = _tweedie_x1(b_theta_fn, xx_req, s_scalar, cond_p)
        diff = x1_hat - y_obs_p
        log_phi_total = -0.5 * (diff * diff).sum() / (sigma_y ** 2)
        grad = torch.autograd.grad(log_phi_total, xx_req, retain_graph=False)[0]
        with torch.no_grad():
            log_phi = -0.5 * (diff * diff).sum(dim=-1) / (sigma_y ** 2)
        return log_phi.detach(), grad.detach()

    def compute_phi(xx, s_scalar):
        with torch.no_grad():
            x1_hat = _tweedie_x1(b_theta_fn, xx, s_scalar, cond_p)
            diff = x1_hat - y_obs_p
            return -0.5 * (diff * diff).sum(dim=-1) / (sigma_y ** 2)

    log_phi_prev = compute_phi(x, ts[0])

    for i in range(n_steps):
        s_cur = ts[i]; s_next = ts[i + 1]
        dt = (s_next - s_cur).item()
        if proposal == 'guided':
            log_phi_cur, grad_log_phi = compute_phi_and_grad(x, s_cur)
        else:
            log_phi_cur = compute_phi(x, s_cur)
            grad_log_phi = None

        with torch.no_grad():
            s_w = s_cur.reshape(1, 1).expand(N, 1)
            b_g = compose_b_g_fn(x, s_w, *cond_p)
            g_s = g_fn(s_w)
            while g_s.dim() < x.dim():
                g_s = g_s.unsqueeze(-1)
            g_scalar = float(g_s.flatten()[0].item())
            t_scalar = float(s_cur.item())
            # Guidance coefficient kappa
            if proposal == 'guided':
                if guidance_type == 'doob':
                    kappa = guidance_eta * (g_scalar ** 2)
                elif guidance_type == 'natural':
                    # C_g(t) = (g^2 + 1 - t^2) / 2
                    C_g = (g_scalar ** 2 + 1.0 - t_scalar ** 2) / 2.0
                    kappa = guidance_eta * C_g
                else:
                    raise ValueError(f'unknown guidance_type {guidance_type}')
                b_eff = b_g + kappa * grad_log_phi
            else:
                kappa = 0.0
                b_eff = b_g
            noise = torch.randn(*x.shape, device=device, dtype=dtype, generator=generator)
            x_new = x + b_eff * dt + g_s * noise * (dt ** 0.5)
            log_phi_next = compute_phi(x_new, s_next)

            # Weight update
            if proposal == 'guided':
                inner = (noise * grad_log_phi).sum(dim=-1)
                grad_sq = (grad_log_phi * grad_log_phi).sum(dim=-1)
                # Girsanov log(p/q) with Δb = kappa * ∇log phi,  Δb/g = (kappa/g) ∇log phi
                kog = kappa / max(g_scalar, 1e-8)
                log_p_over_q = - kog * (dt ** 0.5) * inner \
                               - 0.5 * (kog ** 2) * grad_sq * dt
            else:
                log_p_over_q = 0.0

            log_w = log_w + (log_phi_next - log_phi_cur) + log_p_over_q
            x = x_new

        cur_ess = ess(log_w)
        if return_trace:
            trace['ess'].append(cur_ess); trace['s'].append(float(s_next.item()))

        if cur_ess < resample_thresh * N:
            lw_max = log_w.max()
            total = (log_w - lw_max).exp().sum()
            logZ_accum = logZ_accum + float(lw_max.item()) + math.log(float(total.item() / N))
            idx = systematic_resample(log_w, generator=generator)
            x = x[idx]
            log_w = torch.zeros(N, device=device, dtype=dtype)
            if return_trace:
                trace['logZ'].append(logZ_accum)

    lw_max = log_w.max()
    total = (log_w - lw_max).exp().sum()
    logZ_final = logZ_accum + float(lw_max.item()) + math.log(float(total.item() / N))
    return x, log_w, logZ_final, trace
