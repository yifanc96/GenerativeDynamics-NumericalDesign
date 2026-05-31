"""Twisted SMC for posterior sampling with Tweedie twist and guided-SDE proposal.

Target (path measure):
  pi(x_{0:1}) ∝ p_{prior}(x_{0:1}) · p(y | x_1)

Intermediate targets use Tweedie twist:
  phi_s(x) = p(y | xhat1(x, s)),   xhat1(x, s) = x + (1-s) * b_theta(x, s, cond)

Proposal kernel (per EM step):
  dX = (b^g + g^2 ∇log phi_s) ds + g dW  (guided SDE)

Girsanov-corrected incremental log-weight:
  log w_step = log phi_{s+Δ} - log phi_s
              - g sqrt(Δ) <Z, ∇log phi_s>
              - (1/2) g^2 ‖∇log phi_s‖^2 Δ

with Z = noise used to propose that step.

Resample by multinomial with ESS < N/2 threshold.
"""
import math
import torch


def ess(log_w):
    """log_w: (N,) log weights. Returns effective sample size = 1 / sum(w_i^2) (after normalising)."""
    lw = log_w - log_w.max()
    w = lw.exp()
    w = w / w.sum()
    return 1.0 / (w * w).sum().item()


def systematic_resample(log_w, generator=None):
    """Return indices of N resampled particles."""
    lw = log_w - log_w.max()
    w = lw.exp()
    w = w / w.sum()
    N = w.numel()
    u0 = torch.rand(1, generator=generator, device=w.device).item() / N
    cdf = torch.cumsum(w, dim=0)
    us = u0 + torch.arange(N, device=w.device).float() / N
    idx = torch.searchsorted(cdf, us).clamp_max(N - 1)
    return idx


@torch.no_grad()
def em_init_ensemble(n_particles, dim_tuple, s0, device, ip, dtype=torch.float32, generator=None):
    """Draw initial particles from the interpolant marginal at s0: x ~ N(0, gamma(s0)^2 I)."""
    s0_t = torch.tensor(s0, device=device, dtype=dtype)
    g0 = float(ip.gamma(s0_t).item())
    x = g0 * torch.randn(n_particles, *dim_tuple, device=device, dtype=dtype, generator=generator)
    return x


def _tweedie_x1(b_theta_fn, x, s, cond, ip):
    """Returns (x1_hat, b_hat) via interpolant-specific Tweedie formula."""
    b_hat = b_theta_fn(x, s, *cond)
    x1_hat = ip.tweedie_x1(x, s, b_hat)
    return x1_hat, b_hat


def twisted_smc(b_theta_fn, compose_b_g_fn, g_fn, obs_A, y_obs, sigma_y,
                n_particles, dim_tuple, cond, ip, n_steps=100, t_min=1e-2, t_max=1.0 - 1e-2,
                resample_thresh=0.5, device='cpu', dtype=torch.float32,
                return_trace=False, generator=None,
                proposal='guided', guidance_type='doob', guidance_eta=1.0):
    """
    b_theta_fn:    (x, s, cond) -> hat_b   (raw learned baseline drift)
    compose_b_g_fn:(x, s, cond) -> b^g     (eq 3.9 schedule-specific drift)
    g_fn:          (s,) -> g value
    obs_A:         callable A(x1)
    y_obs:         (1, h, w)  observation tensor (pre-broadcast, same for all particles)
    sigma_y:       scalar
    cond:          tuple of conditioning tensors, each (1, ...) broadcasted across particles
    """
    ts = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)
    N = n_particles
    # Replicate cond across particles
    cond_p = tuple(c.expand(N, *c.shape[1:]).contiguous() for c in cond)
    # Pre-broadcast obs
    y_obs_p = y_obs.expand(N, *y_obs.shape[1:]).contiguous()

    x = em_init_ensemble(N, dim_tuple, ts[0].item(), device, ip, dtype, generator)

    log_w = torch.zeros(N, device=device, dtype=dtype)

    trace = {'ess': [], 'logZ_increment': [], 'step_phi': [], 's': []} if return_trace else None

    def compute_log_phi_and_grad(xx, s_scalar):
        """log phi_s(x) = -‖A xhat1 - y‖^2/(2 sigma_y^2).
        Returns (log_phi : (N,),  grad_log_phi : x.shape),  with graph through the network.
        """
        xx_req = xx.detach().requires_grad_(True)
        s = s_scalar.expand(N, 1)
        x1_hat, _ = _tweedie_x1(b_theta_fn, xx_req, s, cond_p, ip)
        y_pred = obs_A(x1_hat)
        diff = y_pred - y_obs_p
        log_phi_sum = -0.5 * (diff * diff).sum() / (sigma_y ** 2)
        grad = torch.autograd.grad(log_phi_sum, xx_req, retain_graph=False)[0]
        with torch.no_grad():
            log_phi = -0.5 * (diff * diff).sum(dim=(-1, -2, -3)) / (sigma_y ** 2)
        return log_phi.detach(), grad.detach()

    # Evaluate phi_0 at the initial particles
    s0 = ts[0]
    log_phi_prev, _ = compute_log_phi_and_grad(x, s0)

    logZ_accum = 0.0

    for i in range(n_steps):
        s_cur = ts[i]
        s_next = ts[i + 1]
        dt = (s_next - s_cur).item()
        # Compute grad log phi_s only if using guided proposal
        if proposal == 'guided':
            log_phi_cur, grad_log_phi = compute_log_phi_and_grad(x, s_cur)
        else:
            with torch.no_grad():
                s_wide_cur = s_cur.expand(N, 1)
                b_hat_cur = b_theta_fn(x, s_wide_cur, *cond_p)
                x1_hat_cur = ip.tweedie_x1(x, s_wide_cur, b_hat_cur)
                y_pred_cur = obs_A(x1_hat_cur)
                diff_cur = y_pred_cur - y_obs_p
                log_phi_cur = -0.5 * (diff_cur * diff_cur).sum(dim=(-1, -2, -3)) / (sigma_y ** 2)
                grad_log_phi = None
        with torch.no_grad():
            # Schedule-specific drift b^g (forward only, no grad)
            s_wide = s_cur.expand(N, 1)
            b_g = compose_b_g_fn(x, s_wide, *cond_p)
            g_s = g_fn(s_wide)
            while g_s.dim() < x.dim():
                g_s = g_s.unsqueeze(-1)
            g_scalar = float(g_s.flatten()[0].item())
            t_scalar = float(s_cur.item())
            # Determine kappa (guidance coefficient)
            if proposal == 'guided':
                if guidance_type == 'doob':
                    kappa = guidance_eta * (g_scalar ** 2)
                elif guidance_type == 'natural':
                    t_tensor = torch.tensor(t_scalar, device=device, dtype=dtype)
                    g_tensor = torch.tensor(g_scalar, device=device, dtype=dtype)
                    C_g_val = float(ip.C_g(t_tensor, g_tensor).item())
                    kappa = guidance_eta * C_g_val
                else:
                    raise ValueError(f'unknown guidance_type {guidance_type}')
                b_guided = b_g + kappa * grad_log_phi
            else:
                kappa = 0.0
                b_guided = b_g
            noise = torch.randn(*x.shape, device=device, dtype=dtype, generator=generator)
            x_new = x + b_guided * dt + g_s * noise * (dt ** 0.5)
        # Evaluate phi at new particles (no grad needed for the log weight)
        with torch.no_grad():
            s_next_wide = s_next.expand(N, 1)
            b_hat_next = b_theta_fn(x_new, s_next_wide, *cond_p)
            x1_hat_next = ip.tweedie_x1(x_new, s_next_wide, b_hat_next)
            y_pred_next = obs_A(x1_hat_next)
            diff_next = y_pred_next - y_obs_p
            log_phi_next = -0.5 * (diff_next * diff_next).sum(dim=(-1, -2, -3)) / (sigma_y ** 2)
            # Girsanov weight correction: log(p/q)_step = -(kappa/g) sqrt(dt) Z·∇log φ - (kappa²/2g²)||∇log φ||² dt
            if proposal == 'guided':
                inner = (noise * grad_log_phi).sum(dim=(-1, -2, -3))       # (N,)
                grad_sq = (grad_log_phi * grad_log_phi).sum(dim=(-1, -2, -3))
                kog = kappa / max(g_scalar, 1e-8)
                log_p_over_q = - kog * (dt ** 0.5) * inner \
                               - 0.5 * (kog ** 2) * grad_sq * dt
            else:
                log_p_over_q = 0.0
            log_w = log_w + (log_phi_next - log_phi_cur) + log_p_over_q
            x = x_new
        # ESS / resample
        cur_ess = ess(log_w)
        if return_trace:
            trace['ess'].append(cur_ess)
            trace['s'].append(s_next.item())
            trace['step_phi'].append(log_phi_next.mean().item())
        if cur_ess < resample_thresh * N:
            # log-Z increment from this resampling
            lw_max = log_w.max()
            total = (log_w - lw_max).exp().sum()
            logZ_accum = logZ_accum + lw_max.item() + math.log(total.item() / N)
            idx = systematic_resample(log_w, generator=generator)
            x = x[idx]
            log_w = torch.zeros(N, device=device, dtype=dtype)
            if return_trace:
                trace['logZ_increment'].append(logZ_accum)

    # Final log-Z
    lw_max = log_w.max()
    total = (log_w - lw_max).exp().sum()
    logZ_final = logZ_accum + lw_max.item() + math.log(total.item() / N)
    return x, log_w, logZ_final, trace
