"""
Per-frequency-band Lipschitz of the drift over time.

For a perturbation v restricted to wavenumber band B, we measure
  L_B(t, z) = || JVP(b_t)(z; v) ||_2 / ||v||_2.
This is a one-sided estimate (not the spectral norm); but isolates how strongly
the drift responds to perturbations in band B. We average over a few random
v drawn uniformly from the band.

Output: ||∇b_t · v_B||/||v_B|| as function of t, for each band B and each schedule.
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


def make_band_perturbation(shape, k_lo, k_hi, device, seed=0):
    """
    Build a real-valued perturbation v whose Fourier energy is supported in
    wavenumbers k in [k_lo, k_hi). Random phases per mode.
    shape: (1, 1, H, W)
    """
    H, W = shape[-2], shape[-1]
    rng = np.random.RandomState(seed)
    # Build complex Fourier coefficients
    fhat = np.zeros((H, W), dtype=np.complex128)
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq_x = np.fft.fftfreq(W) * W
    ky, kx = np.meshgrid(kfreq_y, kfreq_x, indexing='ij')
    knrm = np.sqrt(kx**2 + ky**2)
    mask = (knrm >= k_lo) & (knrm < k_hi)
    # Random phases
    phases = rng.uniform(0, 2*np.pi, (H, W))
    fhat[mask] = np.exp(1j * phases[mask])
    # Enforce conjugate symmetry for real output
    # (use FFT of a real-valued image: take ifft of fhat then take real part,
    #  this is fine for the L2 norm of v in spatial domain)
    v = np.fft.ifft2(fhat).real
    v = v / (np.linalg.norm(v) + 1e-30)
    v_t = torch.from_numpy(v).to(device).to(torch.float32)
    v_t = v_t.unsqueeze(0).unsqueeze(0).expand(shape).contiguous()
    return v_t


def evaluate_lipperband(model, hi, sigma, lam, num_t=11, num_states=4, num_seeds=3):
    """For each band, schedule, t: estimate ||JVP(b_t)(z; v_B)|| / ||v_B||."""
    device = next(model.parameters()).device
    torch.manual_seed(42)
    z0 = sigma * torch.randn(num_states, 1, hi, hi, device=device)

    def b_lin(zt, ta): return model(zt, ta)
    b_des = make_designed_drift(model, lam)

    states_lin, times = rk4_integrate(b_lin, z0, num_t)
    states_des, _    = rk4_integrate(b_des, z0, num_t)

    # bands: low [1,8), mid [8,24), high [24, hi/2]
    bands = {'low': (1, 8), 'mid': (8, 24), 'high': (24, hi // 2)}

    out = {sched: {b: np.zeros(num_t) for b in bands} for sched in ['linear', 'designed']}

    for ti, tval in enumerate(times):
        t_arr_single = torch.tensor([tval], device=device, dtype=torch.float32)

        for s in range(num_states):
            z_lin = states_lin[ti][s:s+1]
            z_des = states_des[ti][s:s+1]
            for bname, (k_lo, k_hi) in bands.items():
                v_lin = torch.stack([make_band_perturbation(z_lin.shape, k_lo, k_hi, device, seed=g)[0]
                                     for g in range(num_seeds)], dim=0)
                v_des = torch.stack([make_band_perturbation(z_des.shape, k_lo, k_hi, device, seed=g)[0]
                                     for g in range(num_seeds)], dim=0)
                # Average JVP norm over seeds
                resp_lin = 0.0; resp_des = 0.0
                for g in range(num_seeds):
                    fn_lin = lambda zz: b_lin(zz, t_arr_single)
                    fn_des = lambda zz: b_des(zz, t_arr_single)
                    _, jv_lin = jvp(fn_lin, (z_lin,), (v_lin[g:g+1],))
                    _, jv_des = jvp(fn_des, (z_des,), (v_des[g:g+1],))
                    resp_lin += jv_lin.flatten().norm().item() / max(v_lin[g].flatten().norm().item(), 1e-30)
                    resp_des += jv_des.flatten().norm().item() / max(v_des[g].flatten().norm().item(), 1e-30)
                resp_lin /= num_seeds; resp_des /= num_seeds
                out['linear'][bname][ti]   += resp_lin / num_states
                out['designed'][bname][ti] += resp_des / num_states
        print(f"  t={tval:.4f}  Linear (low/mid/high): "
              f"{out['linear']['low'][ti]:6.2f} {out['linear']['mid'][ti]:6.2f} {out['linear']['high'][ti]:6.2f} | "
              f"Designed: {out['designed']['low'][ti]:6.2f} {out['designed']['mid'][ti]:6.2f} {out['designed']['high'][ti]:6.2f}",
              flush=True)
    return np.array(times), out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_states', type=int, default=4)
    p.add_argument('--num_t', type=int, default=11)
    p.add_argument('--num_seeds', type=int, default=3)
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
        t_grid, out = evaluate_lipperband(
            model, hi, sigma, lam,
            num_t=args.num_t, num_states=args.num_states, num_seeds=args.num_seeds,
        )
        results[hi] = (t_grid, out)

    torch.save(results, 'results/lipschitz_perband_data.pt')

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
    band_titles = {'low': r'Low band ($k<8$)',
                   'mid': r'Mid band ($8 \leq k < 24$)',
                   'high': r'High band ($k \geq 24$)'}
    band_keys = ['low', 'mid', 'high']
    for row, hi in enumerate([64, 128]):
        t_grid, out = results[hi]
        for col, bn in enumerate(band_keys):
            ax = axes[row, col]
            ax.plot(t_grid, out['linear'][bn],   'r--', lw=1.8, label='Linear')
            ax.plot(t_grid, out['designed'][bn], 'b-',  lw=1.8, label='Designed')
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)
            if row == 0:
                ax.set_title(band_titles[bn], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"${hi}\\times{hi}$\n" + r"$\|\nabla b_t \cdot v_B\| / \|v_B\|$", fontsize=11)
            if row == 1:
                ax.set_xlabel(r'$t$', fontsize=12)
            if row == 0 and col == 2:
                ax.legend(fontsize=10)
    plt.tight_layout()
    out_path = '../images/NS-lipschitz-perband.pdf'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
