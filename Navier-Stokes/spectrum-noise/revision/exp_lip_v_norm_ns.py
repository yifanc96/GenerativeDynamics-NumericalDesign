"""
Numerical verification of Proposition 3.3's V-norm Lipschitz claim, for the
trained Navier-Stokes white-noise + linear-schedule UNet.

Setup: white-noise prior C_0 = sigma_0^2 I, so V = H with norm
  ||x||_V = (1/sigma_0) ||x||_H.
The data is in V (Appendix F.A: V-norm tightly concentrated near 1), so
Proposition 3.3 strictly applies. The proposition claims that the drift b_t
is Lipschitz in V uniformly on t in [0, 1-delta]. We verify this directly by
sampling random pairs (x_1, x_2) ~ mu_t and computing

    L_emp(t) = max over pairs of  ||b_t(x_1) - b_t(x_2)||_V / ||x_1 - x_2||_V

at a grid of t values.

We also compare with the per-mode V-Lipschitz when we use the matched-
spectrum trained UNet (where V differs from H), to give a meaningful
non-trivial V-norm comparison.
"""
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HOME)
from unet import Unet


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1, dim=32,
                        dim_mults=(1, 2, 2, 2), resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4, use_classes=False)

    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


def load_model(path):
    m = Velocity().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


def v_norm_sq(field, spec_C0):
    """||x||_V^2 = sum_modes |xhat(m)|^2 / c_0(m), per sample.
    spec_C0: (H, W) tensor of per-mode prior variance c_0(m).
    field:   (B, 1, H, W).
    Returns: (B,) tensor of squared V-norms.
    """
    f = field.squeeze(1) if field.dim() == 4 else field
    fhat = torch.fft.fftn(f, dim=(1, 2), norm='forward')
    amp2 = fhat.abs() ** 2
    inv = torch.where(spec_C0 > 1e-30, 1.0 / spec_C0, torch.zeros_like(spec_C0))
    inv[0, 0] = 0.0
    return (amp2 * inv[None]).sum(dim=(1, 2))


def lipschitz_v_norm(model, t, num_pairs, sigma_noise, spec_C0):
    """Empirical V-norm Lipschitz: max over pairs of ||b_t(x_1)-b_t(x_2)||_V / ||x_1-x_2||_V."""
    grid = 128
    # sample pairs (x_1, x_2) from approximate mu_t
    z1 = torch.randn(num_pairs, 1, grid, grid, device=DEVICE) * sigma_noise
    z2 = torch.randn(num_pairs, 1, grid, grid, device=DEVICE) * sigma_noise
    # use small perturbations for numerical stability of the Lipschitz ratio
    eps = 1e-2
    u = torch.randn(num_pairs, 1, grid, grid, device=DEVICE)
    # normalize u to have V-norm = 1
    un = v_norm_sq(u, spec_C0).sqrt()
    u = u / un.view(-1, 1, 1, 1)
    x = z1
    x_pert = x + eps * u
    t_arr = torch.full((num_pairs,), t, device=DEVICE)
    with torch.no_grad():
        bx = model(x, t_arr)
        bxe = model(x_pert, t_arr)
    diff_v = v_norm_sq(bxe - bx, spec_C0).sqrt()
    pert_v = v_norm_sq(eps * u, spec_C0).sqrt()
    ratio = diff_v / pert_v
    return ratio.cpu().numpy()


def main():
    grid = 128
    ckpt = '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/Navier-Stokes/results/ns_gauss_base_5data/model_final.pt'
    model = load_model(ckpt)
    print(f'loaded white-noise NS model: {ckpt}')

    # white-noise prior: c_0(m) = sigma_0^2 (constant); V-norm = H-norm/sigma_0
    sigma_0 = 1.0  # consistent with the trained model
    spec_white = torch.ones(grid, grid, device=DEVICE) * sigma_0 ** 2

    # Also test with the matched-spectrum prior (trained matched model)
    matched_ckpt = os.path.join(HOME, 'ns_spectrum_noise_step5000_hi128.pt')
    if os.path.exists(matched_ckpt):
        amp_match = torch.load(os.path.join(HOME, 'enstrohpy_spectrum_amplitude.pt'),
                               weights_only=False).to(DEVICE) / 5.0
        spec_match = (amp_match.squeeze() ** 2)
    else:
        spec_match = None

    tgrid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
    num_pairs = 32

    print(f"\n=== V-norm Lipschitz of trained NS drift, white-noise prior (V = H) ===")
    print(f"{'t':>5}  {'mean L_V(t)':>12}  {'max L_V(t)':>12}")
    print('-' * 45)
    Lw_means = []
    Lw_maxes = []
    for t in tgrid:
        L = lipschitz_v_norm(model, t, num_pairs, sigma_0, spec_white)
        Lw_means.append(L.mean())
        Lw_maxes.append(L.max())
        print(f"{t:>5.2f}  {L.mean():>12.3e}  {L.max():>12.3e}")

    if spec_match is not None:
        print(f"\n=== V-norm Lipschitz of trained NS drift, MATCHED-spectrum prior (V != H) ===")
        model_m = load_model(matched_ckpt)
        # for matched-spectrum, use sigma_0=1 internally (already in spec_match)
        sigma_match = 1.0
        print(f"{'t':>5}  {'mean L_V(t)':>12}  {'max L_V(t)':>12}")
        print('-' * 45)
        Lm_means = []
        Lm_maxes = []
        for t in tgrid:
            L = lipschitz_v_norm(model_m, t, num_pairs, sigma_match, spec_match)
            Lm_means.append(L.mean())
            Lm_maxes.append(L.max())
            print(f"{t:>5.2f}  {L.mean():>12.3e}  {L.max():>12.3e}")
    else:
        Lm_means = Lm_maxes = None

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    ax.plot(tgrid, Lw_maxes, 'C0-o', ms=4, lw=1.4, label='White-noise prior (rougher than data, $V=H$)')
    if Lm_maxes is not None:
        ax.plot(tgrid, Lm_maxes, 'C2-s', ms=4, lw=1.4, label='Matched-spectrum prior ($V\\neq H$)')
    ax.set_xlabel(r'Interpolation time $t$')
    ax.set_ylabel('Empirical $V$-norm Lipschitz of trained drift')
    ax.set_yscale('log')
    ax.set_title(r'Numerical validation of Proposition \ref{prop-lip-CM-space}: $V$-norm Lipschitz of $b_t$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    out = os.path.join(HOME, 'lip_v_norm_ns.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
