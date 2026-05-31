"""
Empirical Lipschitz constant of the drift over time, for linear vs designed schedule.

For each time t, we estimate ||∇b_t||_2 (spectral norm) via power iteration on
Jacobian-vector products (JVPs) using torch.func.jvp.

Trajectories are obtained by integrating the ODE with each schedule, so the
states at which ||∇b_t|| is evaluated are on-trajectory.

Output:
  images/NS-lipschitz-vs-t.pdf  — ||∇b_t|| vs t, both schedules, both resolutions
  prints summary
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.func import jvp

sys.path.insert(0, os.path.dirname(__file__))
from unet import Unet


def load_ns_data(loc, hi, split):
    avg = 3.0679163932800293
    d, _ = torch.load(loc, weights_only=False)
    d = d / avg
    d = d.reshape(-1, d.shape[2], d.shape[3])
    if hi != d.shape[1]:
        d = nn.functional.interpolate(d.unsqueeze(1), size=(hi, hi), mode='bilinear').squeeze(1)
    d = d[:, None, :, :]
    n = int(d.shape[0] * split)
    return d[:n], d[n:]


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1, dim=32,
                        dim_mults=(1, 2, 2, 2), resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4,
                        use_classes=False)
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


def make_designed_drift(model, ratio):
    r = ratio; lr = math.log(r)
    def alpha(t): return torch.sqrt((r - r**t)/(r - 1)) * torch.ones_like(t)
    def alpha_dot(t): return -0.5/alpha(t) * (r**t) * lr/(r - 1)
    def beta(t): return torch.sqrt((r**t - 1)/(r - 1)) * torch.ones_like(t)
    def beta_dot(t): return 0.5/beta(t) * (r**t) * lr/(r - 1)
    def b_des(zt, t_arr):
        bt = (alpha_dot(t_arr)/alpha(t_arr))[:, None, None, None] * zt
        coef = (beta_dot(t_arr) - alpha_dot(t_arr)*beta(t_arr)/alpha(t_arr))[:, None, None, None]
        orig_t = 1/(1 + alpha(t_arr)/beta(t_arr))
        orig_x = orig_t[:, None, None, None]/(beta(t_arr)[:, None, None, None]) * zt
        orig_bt = model(orig_x, orig_t)
        bt += coef * ((1 - orig_t[:, None, None, None]) * orig_bt + orig_x)
        return bt
    return b_des


@torch.no_grad()
def rk4_integrate(drift_fn, z0, steps, t_min=1e-3, t_max=1-1e-3):
    """Integrate dz/dt = drift_fn(z, t) and return states at all grid points."""
    tg = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0
    states = [zt.clone()]
    times = [tg[0].item()]
    o = torch.ones(zt.shape[0], device=zt.device)
    for i in range(len(tg) - 1):
        t_i = tg[i]; dt = tg[i+1] - tg[i]; ta = t_i * o
        k1 = drift_fn(zt, ta); k2 = drift_fn(zt + 0.5*dt*k1, (t_i + 0.5*dt) * o)
        k3 = drift_fn(zt + 0.5*dt*k2, (t_i + 0.5*dt) * o); k4 = drift_fn(zt + dt*k3, (t_i + dt) * o)
        zt = zt + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        states.append(zt.clone())
        times.append(tg[i+1].item())
    return states, times


def spectral_norm_jvp(drift_fn, z, t_scalar, num_iter=12, num_seeds=2):
    """
    Estimate ||∇_z b_t(z)||_2 (spectral norm) via power iteration on JVPs.
    Average over a few random initial vectors and take the max.
    z: (1, C, H, W)  -- single example at the desired state.
    t_scalar: float
    """
    device = z.device
    C, H, W = z.shape[1], z.shape[2], z.shape[3]
    t_arr = torch.tensor([t_scalar], device=device, dtype=z.dtype)

    def fn(zz):
        return drift_fn(zz, t_arr)

    best = 0.0
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        v = torch.randn_like(z)
        v = v / (v.flatten().norm() + 1e-30)
        sigma_est = 0.0
        for _ in range(num_iter):
            _, jv = jvp(fn, (z,), (v,))
            jv_norm = jv.flatten().norm().item() + 1e-30
            sigma_est = jv_norm
            v = jv / jv_norm
        if sigma_est > best:
            best = sigma_est
    return best


def evaluate_lipschitz(model, hi, sigma, lam, num_states=8, num_t=21, num_iter=12, num_seeds=2):
    """
    Returns dict: t_grid -> {'linear': mean_lip, 'designed': mean_lip}
    Estimated by averaging spectral norm over `num_states` independent trajectories at each t.
    """
    device = next(model.parameters()).device
    torch.manual_seed(42)
    z0 = sigma * torch.randn(num_states, 1, hi, hi, device=device)

    # Linear trajectory
    def b_lin(zt, ta): return model(zt, ta)
    states_lin, times_lin = rk4_integrate(b_lin, z0, num_t)

    # Designed trajectory
    b_des = make_designed_drift(model, lam)
    states_des, times_des = rk4_integrate(b_des, z0, num_t)

    lip_lin = []; lip_des = []
    for ti, t in enumerate(times_lin):
        per_state_lin = []
        per_state_des = []
        for s in range(num_states):
            zlin = states_lin[ti][s:s+1]
            zdes = states_des[ti][s:s+1]
            with torch.enable_grad():
                sn_lin = spectral_norm_jvp(b_lin, zlin, t, num_iter=num_iter, num_seeds=num_seeds)
                sn_des = spectral_norm_jvp(b_des, zdes, t, num_iter=num_iter, num_seeds=num_seeds)
            per_state_lin.append(sn_lin)
            per_state_des.append(sn_des)
        lip_lin.append(per_state_lin)
        lip_des.append(per_state_des)
        print(f"  t={t:.4f}  Lip(linear)={np.mean(per_state_lin):8.3f}  Lip(designed)={np.mean(per_state_des):8.3f}", flush=True)

    return np.array(times_lin), np.array(lip_lin), np.array(lip_des)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_states', type=int, default=8)
    p.add_argument('--num_t', type=int, default=21)
    p.add_argument('--num_iter', type=int, default=12)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    cases = [
        (64,  'results/ns_64_noise10/model_final.pt',  10.0, 3e-4),
        (128, 'results/ns_noise10_lipR1e5/model_final.pt', 10.0, 1e-5),
    ]

    results = {}
    for hi, ckpt, sigma, lam in cases:
        print(f"\n=== Resolution {hi}x{hi}, sigma={sigma}, lambda*={lam:.0e} ===")
        model = Velocity().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()
        t_grid, lip_lin, lip_des = evaluate_lipschitz(
            model, hi, sigma, lam,
            num_states=args.num_states, num_t=args.num_t, num_iter=args.num_iter,
        )
        results[hi] = (t_grid, lip_lin, lip_des)

    # Save raw data
    torch.save(results, 'results/lipschitz_check_data.pt')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, hi in zip(axes, [64, 128]):
        t_grid, lip_lin, lip_des = results[hi]
        m_lin = lip_lin.mean(axis=1); s_lin = lip_lin.std(axis=1)
        m_des = lip_des.mean(axis=1); s_des = lip_des.std(axis=1)
        ax.plot(t_grid, m_lin, 'r--', lw=1.8, label='Linear')
        ax.fill_between(t_grid, m_lin - s_lin, m_lin + s_lin, color='r', alpha=0.15)
        ax.plot(t_grid, m_des, 'b-', lw=1.8, label='Designed')
        ax.fill_between(t_grid, m_des - s_des, m_des + s_des, color='b', alpha=0.15)
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$', fontsize=12)
        ax.set_ylabel(r'$\|\nabla b_t\|_2$', fontsize=12)
        ax.set_title(f'${hi}\\times{hi}$', fontsize=13)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=11)

    plt.tight_layout()
    out = '../images/NS-lipschitz-vs-t.pdf'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # Print summary: integrated Lipschitz (avg-Lip indicator)
    print("\n=== Time-averaged Lipschitz (proxy for avg-Lip) ===")
    for hi in [64, 128]:
        t_grid, lip_lin, lip_des = results[hi]
        m_lin = lip_lin.mean(axis=1)
        m_des = lip_des.mean(axis=1)
        avg_lin = np.trapz(m_lin, t_grid)
        avg_des = np.trapz(m_des, t_grid)
        max_lin = m_lin.max()
        max_des = m_des.max()
        print(f"  {hi}x{hi}:  ∫ Lip dt:  Linear={avg_lin:7.2f}  Designed={avg_des:7.2f}  "
              f"(ratio L/D={avg_lin/avg_des:.2f})")
        print(f"           max Lip:    Linear={max_lin:7.2f}  Designed={max_des:7.2f}  "
              f"(ratio L/D={max_lin/max_des:.2f})")
