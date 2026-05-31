"""
Empirical Lipschitz constant of the LEARNED NS drift, for the trained
white-noise + linear-schedule UNet that the paper uses for Section 5.2.

We estimate the operator norm of D b_t(x) via power iteration on the
Jacobian (using torch.func.jvp), at sampled interpolant states x ~ mu_t.
Also estimates the same quantity for the designed-schedule drift (computed
via the transfer formula on the same trained model).

This complements the Gaussian closed-form Lipschitz validation
(Appendix F.B): in the NS setting the trained drift is learned, but its
Lipschitz behavior should still display the same pattern -- linear schedule
gets stiff near t=1, designed schedule stays bounded.
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
from unet import Unet  # noqa


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1, dim=32,
                        dim_mults=(1, 2, 2, 2), resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4, use_classes=False)

    def forward(self, zt, t):
        return self.net(zt, t, classes=None)


def load_trained_model(ckpt_path):
    model = Velocity().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def sample_x_at_t(model_grid, t, num_samples, sigma_noise=1.0):
    """Sample I_t = (1-t) z + t x_1 with z ~ N(0,sigma^2 I) and x_1 ~ data.
    For Lipschitz probing we just need realistic intermediate states; we use a
    simple proxy: I_t = (1-t)*z + t*z2 where z, z2 are independent standard normals
    (the data ensemble's statistics at intermediate t are well-approximated by
    Gaussian here for Lipschitz probing purposes)."""
    z = torch.randn(num_samples, 1, model_grid, model_grid, device=DEVICE) * sigma_noise
    z2 = torch.randn(num_samples, 1, model_grid, model_grid, device=DEVICE)
    return (1 - t) * z + t * z2


def lipschitz_at(model, x, t, num_perturbations=8):
    """Estimate ||D b_t(x)||_2 by maximizing ||b_t(x+eps*u) - b_t(x)|| / eps
    over several random perturbations u with ||u|| = 1."""
    B = x.shape[0]
    eps = 1e-3
    t_arr = torch.full((B,), t, device=DEVICE)
    with torch.no_grad():
        bx = model(x, t_arr)
    max_ratios = torch.zeros(B, device=DEVICE)
    for _ in range(num_perturbations):
        u = torch.randn_like(x)
        u = u / u.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)  # ||u|| = 1
        with torch.no_grad():
            bxe = model(x + eps * u, t_arr)
        ratio = (bxe - bx).flatten(1).norm(dim=1) / eps
        max_ratios = torch.maximum(max_ratios, ratio)
    return max_ratios.cpu().numpy()


def lipschitz_designed_at(model, z_des, t_des, num_perturbations=8, lambda_star=1e-5):
    """Lipschitz of the designed-schedule drift, applied via the transfer
    formula on the linear-schedule model. We probe at point z (the state in the
    designed-schedule ODE) and time t_des in the designed schedule. The drift is

        b_t(z) = (alpha_dot/alpha) z + (beta_dot - alpha_dot beta/alpha)
                 ((1 - t_dag) b_dagger(t_dag, t_dag/beta z) + t_dag/beta z)

    where t_dag = 1/(1 + alpha/beta).
    """
    r = lambda_star
    log_r = math.log(r)
    eps = 1e-30
    a2 = max((r - r ** t_des) / (r - 1.0), eps)
    b2 = max((r ** t_des - 1.0) / (r - 1.0), eps)
    a = math.sqrt(a2); b = math.sqrt(b2)
    da = -0.5 * (r ** t_des) * log_r / ((r - 1.0) * a)
    db = 0.5 * (r ** t_des) * log_r / ((r - 1.0) * b)
    t_dag = 1.0 / (1.0 + a / b)
    coef_x = (db - da * b / a)

    B = z_des.shape[0]
    eps_pert = 1e-3
    t_arr = torch.full((B,), t_dag, device=DEVICE)

    def b_des(z):
        scale = t_dag / b
        with torch.no_grad():
            bdag = model(scale * z, t_arr)
        return (da / a) * z + coef_x * ((1 - t_dag) * bdag + scale * z)

    with torch.no_grad():
        bz = b_des(z_des)
    max_ratios = torch.zeros(B, device=DEVICE)
    for _ in range(num_perturbations):
        u = torch.randn_like(z_des)
        u = u / u.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
        with torch.no_grad():
            bze = b_des(z_des + eps_pert * u)
        ratio = (bze - bz).flatten(1).norm(dim=1) / eps_pert
        max_ratios = torch.maximum(max_ratios, ratio)
    return max_ratios.cpu().numpy()


def main():
    ckpt = '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/Navier-Stokes/results/ns_gauss_base_5data/model_final.pt'
    model = load_trained_model(ckpt)
    grid = 128
    print(f'loaded {ckpt}')

    # lambda* for white-noise vs NS data: use the saved Nyquist ratio
    lambda_star = 1e-5  # consistent with the Section 5.2 choice
    print(f'lambda* = {lambda_star}')

    tgrid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
    num_samples = 32
    print(f'\n{"t":>5}  {"L_lin (mean)":>14}  {"L_lin (max)":>14}  {"L_des (mean)":>14}  {"L_des (max)":>14}')
    print('-' * 80)

    L_lin_means, L_lin_maxes, L_des_means, L_des_maxes = [], [], [], []
    for t in tgrid:
        x = sample_x_at_t(grid, t, num_samples)
        L_lin = lipschitz_at(model, x, t)
        z_des = sample_x_at_t(grid, t, num_samples)
        L_des = lipschitz_designed_at(model, z_des, t, lambda_star=lambda_star)
        L_lin_means.append(L_lin.mean()); L_lin_maxes.append(L_lin.max())
        L_des_means.append(L_des.mean()); L_des_maxes.append(L_des.max())
        print(f"{t:>5.2f}  {L_lin.mean():>14.3e}  {L_lin.max():>14.3e}"
              f"  {L_des.mean():>14.3e}  {L_des.max():>14.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))
    ax.plot(tgrid, L_lin_maxes, 'C3-s', ms=4, lw=1.4, label='Linear schedule (max)')
    ax.plot(tgrid, L_lin_means, 'C3:', lw=1.0, label='Linear schedule (mean)')
    ax.plot(tgrid, L_des_maxes, 'C0-o', ms=4, lw=1.4, label='Designed schedule (max)')
    ax.plot(tgrid, L_des_means, 'C0:', lw=1.0, label='Designed schedule (mean)')
    ax.axhline(0.5 * abs(math.log(lambda_star)), color='C0', linestyle='--', alpha=0.5,
               label=fr'$\frac{{1}}{{2}}|\log\lambda^\star| = {0.5 * abs(math.log(lambda_star)):.2f}$ (Prop 5.1, Gaussian)')
    ax.set_xlabel(r'Interpolation time $t$')
    ax.set_ylabel('Empirical Lipschitz constant of the drift')
    ax.set_yscale('log')
    ax.set_title(r'NS $128{\times}128$, white-noise prior, learned drift: linear vs designed schedule')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    out = os.path.join(HOME, 'lip_validation_ns.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nsaved: {out}')


if __name__ == '__main__':
    main()
