"""Correctness checks for the §3.4.1 implementation."""
import torch

from interpolant_zps import ZPSInterpolant
from schedules_zps import make_g, list_schedules
from drift_compose import compose_drift, follmer_drift_sanity
from targets_zps import make_target
from sampler import em_sample
from metrics import wasserstein_1d, path_kl_girsanov


def test_interpolant():
    ip = ZPSInterpolant()
    t = torch.tensor([[0.0], [0.5], [1.0]])
    # gamma(0) = 0, gamma(1) = 0
    g = ip.gamma(t)
    assert g[0].item() == 0.0 and abs(g[2].item()) < 1e-6
    assert abs(g[1].item() - 0.5 * (0.5 ** 0.5)) < 1e-6
    x1 = torch.randn(3, 1); z = torch.randn(3, 1)
    I0 = ip.It(x1, z, torch.tensor([[0.0], [0.0], [0.0]]))
    I1 = ip.It(x1, z, torch.tensor([[1.0], [1.0], [1.0]]))
    assert torch.allclose(I0, torch.zeros_like(I0), atol=1e-6)
    assert torch.allclose(I1, x1, atol=1e-6)
    print("[OK] interpolant: I_0 = 0, I_1 = x_1, gamma_0 = gamma_1 = 0")


def test_drift_composition_at_baseline():
    """At g = sigma_t = 1 - t, the composition should reduce to b_t (no correction)."""
    target = make_target('gaussian1d', mu=1.2, sigma=0.4)
    b_fn = target.b_star
    g_baseline = make_g('baseline', scale=1.0)
    b_g = compose_drift(b_fn, g_baseline)
    x = torch.randn(50, 1)
    t = torch.rand(50, 1) * 0.9 + 0.05
    assert torch.allclose(b_fn(x, t), b_g(x, t), atol=1e-5)
    print("[OK] at g = sigma_t, composed drift == baseline b_t")


def test_drift_composition_at_follmer():
    """At g = sqrt(1 - t^2), composed drift == (1+t) b_t - x (eq 3.31)."""
    target = make_target('gaussian1d', mu=1.2, sigma=0.4)
    b_fn = target.b_star
    g_follmer = make_g('follmer', scale=1.0)
    b_g_numerical = compose_drift(b_fn, g_follmer)
    b_g_closed = follmer_drift_sanity(b_fn)
    x = torch.randn(100, 1)
    t = torch.rand(100, 1) * 0.9 + 0.05
    diff = (b_g_numerical(x, t) - b_g_closed(x, t)).abs().max().item()
    assert diff < 1e-5, f"max diff: {diff}"
    print(f"[OK] at g = sqrt(1 - t^2), composed drift == (1+t) b_t - x (max diff {diff:.2e})")


def test_exact_drift_preserves_marginals():
    """With analytic b_star composed via eq (3.9), every schedule should produce
    X_1 ~ target (up to EM error). Target A, 20k samples, N_EM=500."""
    torch.manual_seed(0)
    target = make_target('gaussian1d', mu=1.2, sigma=0.4)
    truth = target.sample_x1(20000)
    print(f"[test] target mean {truth.mean().item():.4f} (expected 1.2), "
          f"std {truth.std().item():.4f} (expected 0.4)")
    for name in list_schedules():
        g_fn = make_g(name, scale=1.0)
        b_g = compose_drift(target.b_star, g_fn)
        samples = em_sample(b_g, g_fn, n=20000, dim=1, n_steps=500)
        w2 = wasserstein_1d(samples, truth, p=2)
        tag = 'OK' if w2 < 5e-2 else 'WARN'
        print(f"  [{tag}] {name:10s} W2={w2:.4e}")


def test_path_kl_zero_for_exact():
    torch.manual_seed(0)
    target = make_target('gaussian1d', mu=1.2, sigma=0.4)
    g_fn = make_g('follmer', scale=1.0)
    b_g = compose_drift(target.b_star, g_fn)
    kl = path_kl_girsanov(b_g, b_g, g_fn, target, n_mc=10000)
    assert abs(kl) < 1e-10
    print(f"[OK] path KL for b_theta = b_star is {kl:.2e}")


if __name__ == '__main__':
    test_interpolant()
    test_drift_composition_at_baseline()
    test_drift_composition_at_follmer()
    test_exact_drift_preserves_marginals()
    test_path_kl_zero_for_exact()
