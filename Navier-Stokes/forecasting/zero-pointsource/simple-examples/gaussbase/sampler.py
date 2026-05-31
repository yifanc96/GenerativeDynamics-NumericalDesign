"""Euler-Maruyama sampler for X_0 = 0 to X_1 ~ target.

The drift `b_fn(x, t, *cond)` is a pluggable callable: learned (b_theta),
exact (b_star), or exact + injected noise.
"""
import torch


@torch.no_grad()
def em_sample(b_fn, g_fn, n, dim, n_steps=100, t_min=0.0, t_max=1.0,
              cond=(), device='cpu', dtype=torch.float32, return_last_mean=True,
              init='gaussian'):
    """Draw n samples by integrating dX = b dt + g dW on [t_min, t_max].

    Args:
        b_fn: drift callable, accepts (x, t, *cond) -> tensor same shape as x.
        g_fn: diffusion-coefficient callable, accepts (t,) -> tensor shape (B,1) or scalar.
        init: 'gaussian' for X_0 ~ N(0, I) (Gaussian-base SI) or 'delta' for X_0 = 0.
    Returns:
        (n, dim) tensor.
    """
    ts = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)
    if init == 'gaussian':
        x = torch.randn(n, dim, device=device, dtype=dtype)
    else:
        x = torch.zeros(n, dim, device=device, dtype=dtype)
    x_mean = x
    for i in range(n_steps):
        t_scalar = ts[i]
        t = t_scalar.expand(n, 1)
        dt = (ts[i + 1] - ts[i]).item()
        drift = b_fn(x, t, *cond)
        gt = g_fn(t)  # shape (n, 1) or (1,)
        x_mean = x + drift * dt
        if gt.abs().max().item() > 0.0:
            noise = torch.randn_like(x) * torch.sqrt(torch.tensor(dt, device=device, dtype=dtype))
            x = x_mean + gt * noise
        else:
            x = x_mean
    return x_mean if return_last_mean else x


@torch.no_grad()
def em_sample_trajectories(b_fn, g_fn, n, dim, n_steps=100, t_min=0.0, t_max=1.0,
                           cond=(), device='cpu', dtype=torch.float32):
    """Return full trajectory of shape (n_steps+1, n, dim) for diagnostics."""
    ts = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)
    x = torch.zeros(n, dim, device=device, dtype=dtype)
    traj = torch.zeros(n_steps + 1, n, dim, device=device, dtype=dtype)
    traj[0] = x
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
        traj[i + 1] = x
    return traj, ts
