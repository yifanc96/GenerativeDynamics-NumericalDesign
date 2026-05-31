"""EM sampler for (B, 1, H, W) fields, conditioned on x0_up."""
import torch


def gamma_t(t):
    return (1.0 - t) * t.clamp_min(0.0).sqrt()


@torch.no_grad()
def em_sample(b_fn, g_fn, shape, n_steps=100, t_min=1e-3, t_max=1.0 - 1e-3,
              cond=(), device='cpu', dtype=torch.float32, return_last_mean=True):
    """shape = (n, C, H, W). b_fn(x, t, *cond) -> same shape."""
    ts = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)
    t0 = ts[0].reshape(1, 1)
    g0 = gamma_t(t0).item()
    if g0 > 0:
        x = g0 * torch.randn(*shape, device=device, dtype=dtype)
    else:
        x = torch.zeros(*shape, device=device, dtype=dtype)
    x_mean = x
    for i in range(n_steps):
        t_scalar = ts[i]
        t = t_scalar.expand(shape[0], 1)
        dt = (ts[i + 1] - ts[i]).item()
        drift = b_fn(x, t, *cond)
        gt = g_fn(t)
        while gt.dim() < x.dim():
            gt = gt.unsqueeze(-1)
        x_mean = x + drift * dt
        if gt.abs().max().item() > 0.0:
            noise = torch.randn_like(x) * (dt ** 0.5)
            x = x_mean + gt * noise
        else:
            x = x_mean
    return x_mean if return_last_mean else x
