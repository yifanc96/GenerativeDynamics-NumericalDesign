"""EM sampler for the point-source SDE dX = b^g(X, t) dt + g(t) dW, X_0 = 0.

Because gamma(0) = 0 (delta initial), we start at t = t_min > 0 with
x ~ N(0, gamma(t_min)^2 I), i.e. drawn from the interpolant marginal at t_min.
For g = 0 (ODE), the noise step is skipped.
"""
import torch


def gamma_t(t):
    """gamma_t for the §3.4.1 interpolant, = (1 - t) sqrt(t). t: tensor (..., 1)."""
    return (1.0 - t) * t.clamp_min(0.0).sqrt()


@torch.no_grad()
def em_sample(b_fn, g_fn, n, dim, n_steps=200, t_min=1e-3, t_max=1.0 - 1e-3,
              cond=(), device='cpu', dtype=torch.float32, return_last_mean=True):
    ts = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)
    # initial condition: sample from interpolant marginal at t_min
    t0 = ts[0].reshape(1, 1)
    gamma0 = gamma_t(t0).item()
    if gamma0 > 0:
        x = gamma0 * torch.randn(n, dim, device=device, dtype=dtype)
    else:
        x = torch.zeros(n, dim, device=device, dtype=dtype)
    x_mean = x
    for i in range(n_steps):
        t_scalar = ts[i]
        t = t_scalar.expand(n, 1)
        dt = (ts[i + 1] - ts[i]).item()
        drift = b_fn(x, t, *cond)
        gt = g_fn(t)
        x_mean = x + drift * dt
        if gt.abs().max().item() > 0.0:
            noise = torch.randn_like(x) * (dt ** 0.5)
            x = x_mean + gt * noise
        else:
            x = x_mean
    return x_mean if return_last_mean else x
