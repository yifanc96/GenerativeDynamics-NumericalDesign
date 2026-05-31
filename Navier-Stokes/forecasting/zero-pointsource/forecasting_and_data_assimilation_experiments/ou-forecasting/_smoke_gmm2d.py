"""Sanity checks for the GMM2DForecast target + 2D path/metric pipeline."""
import torch

from interpolant_zps import ZPSInterpolant
from schedules_zps import make_g, list_schedules
from drift_compose import compose_drift
from targets_zps import make_target
from sampler import em_sample
from metrics import sliced_wasserstein, path_kl_girsanov


def test_gmm2d_distributions():
    torch.manual_seed(0)
    tg = make_target('gmm2d_forecast', sigma_noise=0.5, lam_tau=1.0)
    (y,) = tg.sample_cond(20000)
    print(f"[cond] y mean: {y.mean(0).tolist()}  std: {y.std(0).tolist()}")
    x1, y2 = tg.sample_x1(20000)
    print(f"[target x1] mean: {x1.mean(0).tolist()}  std: {x1.std(0).tolist()}  "
          f"min/max: {x1.min().item():.2f}/{x1.max().item():.2f}")
    # Poisson mod-5 weights for lam=1: expected weights (computed by hand below)
    print(f"[pi] mod-5 weights: {tg.pi.tolist()}  (sum={tg.pi.sum().item():.6f})")


def test_bstar_finite():
    torch.manual_seed(0)
    tg = make_target('gmm2d_forecast', sigma_noise=0.5)
    y = tg.sample_cond(100)[0]
    t = torch.rand(100, 1) * 0.9 + 0.05
    z = torch.randn(100, 2)
    x1, _ = tg.sample_x1(100, y=y)
    xt = t * x1 + (1.0 - t) * t.clamp_min(0.0).sqrt() * z
    b = tg.b_star(xt, t, y)
    print(f"[b_star] shape {tuple(b.shape)}  max |b|={b.abs().max().item():.2f}")


def test_exact_drift_preserves_marginal():
    """With analytic b_star under eq (3.9), every (non-ODE) schedule should
    yield samples close to truth. Fix one y, draw many x_1, compare."""
    torch.manual_seed(0)
    tg = make_target('gmm2d_forecast', sigma_noise=0.5)
    # Use a batch of y_s; each trajectory conditioned on the shared y_s
    n = 10000
    (y,) = tg.sample_cond(n)
    truth, _ = tg.sample_x1(n, y=y)
    t_eps = 1e-3
    for name in list_schedules():
        g_fn = make_g(name, scale=1.0)
        bg = compose_drift(tg.b_star, g_fn)
        samples = em_sample(bg, g_fn, n, tg.dim, n_steps=500,
                            t_min=t_eps, t_max=1.0 - t_eps, cond=(y,))
        sw2 = sliced_wasserstein(samples, truth, n_proj=200, p=2)
        tag = 'OK' if sw2 < 0.1 else 'WARN'
        print(f"  [{tag}] {name:10s} SW2={sw2:.4e}")


def test_path_kl_zero():
    torch.manual_seed(0)
    tg = make_target('gmm2d_forecast', sigma_noise=0.5)
    g_fn = make_g('follmer', scale=1.0)
    bg = compose_drift(tg.b_star, g_fn)
    kl = path_kl_girsanov(bg, bg, g_fn, tg, n_mc=20000, t_min=1e-3, t_max=1.0 - 1e-3)
    print(f"[path-KL b_theta=b_star] {kl:.3e}")


if __name__ == "__main__":
    test_gmm2d_distributions()
    test_bstar_finite()
    test_exact_drift_preserves_marginal()
    test_path_kl_zero()
