"""
Reproduce Figure 2 of the paper (Gaussian energy spectra at 32, 64, 128 with
white noise vs spectrum noise) using the closed-form Gaussian drift.
"""
import math
import os
import numpy as np
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))
T_MIN, T_MAX = 1e-4, 1.0 - 1e-4

def sample_matern(num_samples, grid, sig, ls, s, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    f = torch.fft.fftfreq(grid, device=DEVICE) * 2 * math.pi * grid
    fx, fy = torch.meshgrid(f, f, indexing='ij')
    lap = fx**2 + fy**2
    spec = sig * (lap + ls**2) ** (-s); spec[0,0] = 0
    re = torch.randn(num_samples, grid, grid, generator=g, device=DEVICE)
    im = torch.randn(num_samples, grid, grid, generator=g, device=DEVICE)
    return torch.fft.ifft2(torch.sqrt(spec)[None] * (re + 1j*im), norm='forward').real

def spec_density(grid, sig, ls, s):
    f = torch.fft.fftfreq(grid, device=DEVICE) * 2 * math.pi * grid
    fx, fy = torch.meshgrid(f, f, indexing='ij')
    return sig * (fx**2 + fy**2 + ls**2) ** (-s)

def integrate_rk4_linear(z0, sd0, sd1, steps, t_min=T_MIN, t_max=T_MAX):
    tg = torch.linspace(t_min, t_max, steps + 1, device=DEVICE)
    z = z0.clone()
    for i in range(steps):
        ti = tg[i].item(); dt = (tg[i+1]-tg[i]).item()
        def b(z_, t_):
            a, da, bb, db = 1-t_, -1.0, t_, 1.0
            zfft = torch.fft.fft2(z_, norm='forward')
            r = (a*da*sd0 + bb*db*sd1) / (a**2*sd0 + bb**2*sd1 + 1e-40)
            return torch.fft.ifft2(r[None]*zfft, norm='forward').real
        k1=b(z,ti); k2=b(z+0.5*dt*k1, ti+0.5*dt); k3=b(z+0.5*dt*k2, ti+0.5*dt); k4=b(z+dt*k3, ti+dt)
        z = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return z

def radial_spectrum(field):
    fhat = torch.fft.fftn(field.cpu(), dim=(1,2), norm='forward')
    amp2 = (fhat.abs()**2).mean(0).numpy()
    npix = amp2.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kxg, kyg = np.meshgrid(kfreq, kfreq, indexing='ij')
    knrm = np.sqrt(kxg**2 + kyg**2).flatten()
    af = amp2.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.0)
    kvals = 0.5*(kbins[1:]+kbins[:-1])
    A,_,_ = stats.binned_statistic(knrm, af, statistic='mean', bins=kbins)
    return kvals, A * np.pi*(kbins[1:]**2-kbins[:-1]**2)

# Recreate Figure 2: 3 panels at res 32, 64, 128
ls = 1.0; s1 = 3
sig1 = ((2*math.pi)**2 + ls**2)**s1

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, grid, white_steps in zip(axes, [32, 64, 128], [20, 40, 80]):
    sd0_white = spec_density(grid, 1.0, ls, 0)
    sd0_match = spec_density(grid, sig1, ls, s1)  # matched = same as data
    sd1 = spec_density(grid, sig1, ls, s1)
    z0_white = sample_matern(2000, grid, 1.0, ls, 0, seed=11)
    z0_match = sample_matern(2000, grid, sig1, ls, s1, seed=12)
    truth = sample_matern(2000, grid, sig1, ls, s1, seed=13)
    gen_white = integrate_rk4_linear(z0_white, sd0_white, sd1, white_steps)
    gen_match = integrate_rk4_linear(z0_match, sd0_match, sd1, 5)
    kvals, S_truth = radial_spectrum(truth)
    _, S_white = radial_spectrum(gen_white)
    _, S_match = radial_spectrum(gen_match)
    ax.plot(kvals, S_truth, 'k-', lw=1.5, label='Truth')
    ax.plot(kvals, S_match, 'b-o', ms=2, lw=1.0, label=f'Spectrum noise (5 RK4)')
    ax.plot(kvals, S_white, 'r--', lw=1.0, label=f'White noise ({white_steps} RK4)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'Energy spectrum')
    ax.set_title(f'Gaussian {grid}x{grid}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    rel_white = float(np.abs(S_white-S_truth)[1:].mean()/np.abs(S_truth)[1:].mean())
    rel_match = float(np.abs(S_match-S_truth)[1:].mean()/np.abs(S_truth)[1:].mean())
    print(f'res{grid}: white {white_steps} RK4 rel err = {rel_white:.3e} | spec 5 RK4 rel err = {rel_match:.3e}')
plt.tight_layout()
out = os.path.join(HOME, 'reproduced_paper_fig2_Gaussian.pdf')
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved:', out)
