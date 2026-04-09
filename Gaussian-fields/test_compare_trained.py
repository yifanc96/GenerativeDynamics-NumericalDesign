"""
Compare trained checkpoints at G=64 s=3 with the FIXED per-phase samplers:
  (a) standard 1-mask flow matching   (results/scaling_s3.0_1mask_long)
  (b) wavelet K=3 single-net          (results/s3_wav3_50k)
  (c) mask K=6 multi-net (just trained: results/mask_K6_multi_proper)

Sweep RK4 step counts and report mean |E_gen / E_truth - 1| over k=1..30
(skipping the last 2 Nyquist bins where the radial-binning convention bites).
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_multiscale_interpolation import (
    HierarchicalMasks, Sampler as MaskSampler,
    SingleVelocity, MultiVelocity,
    precompute_matern_amplitude, sample_matern_batch,
    estimate_conditional_variances, build_noise_std,
    get_fourier_spectrum,
)

torch.manual_seed(0)
np.random.seed(0)

G        = 64
s        = 3.0
ls       = 1.0
sigma_sq_o = 1.0
n_test   = 1000
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── truth ───────────────────────────────────────────────────────────────────
sigma_sq = sigma_sq_o * ((2 * math.pi) ** 2 + ls ** 2) ** s
amp = precompute_matern_amplitude(G, sigma_sq, ls, s)
truth = sample_matern_batch(amp, n_test, device='cpu')
print(f'[Truth] G={G} s={s} std={truth.std():.4f}')

kvals, E_truth = get_fourier_spectrum(truth.numpy())

def relerr(E_gen, hi=30, lo=0):
    return float(np.mean(np.abs(E_gen[lo:hi] - E_truth[lo:hi]) / (np.abs(E_truth[lo:hi]) + 1e-30)))

def chunked_run(sampler, model, z0, steps, chunk=200, **kwargs):
    outs = []
    for i in range(0, z0.shape[0], chunk):
        out = sampler.RK4(z0[i:i+chunk].clone(), model, steps=steps, **kwargs)
        outs.append(out.cpu())
    return torch.cat(outs, dim=0)


# ════════════════════════════════════════════════════════════════════════════
# (a) Standard 1-mask flow matching
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 72)
print('(a) Standard 1-mask flow matching  (results/scaling_s3.0_1mask_long)')
print('=' * 72)

hier1 = HierarchicalMasks(G, num_masks=1, device=device)
var_data = sample_matern_batch(amp, 2000, device='cpu')
cond_vars1 = estimate_conditional_variances(var_data, [m.cpu() for m in hier1.masks])
print(f'  cond_vars (K=1): {cond_vars1}')
noise_std_1 = build_noise_std(hier1.masks, cond_vars1, device=device)

net1 = SingleVelocity(hier1, channels=32, dim_mults=(1, 2, 2, 2)).to(device)
sd1 = torch.load('results/scaling_s3.0_1mask_long/model_final.pt', map_location=device)
net1.load_state_dict(sd1)
net1.eval()

sampler1 = MaskSampler(num_masks=1)

def make_z0_1(B):
    z = torch.randn(B, 1, G, G, device=device)
    return z * noise_std_1[None, None, :, :]

torch.manual_seed(42)
z0_1 = make_z0_1(n_test)

print(f'  {"steps":>6}  {"k≤10":>10}  {"k≤20":>10}  {"k≤30":>10}')
results_1mask = {}
for steps in [10, 20, 40, 80]:
    out = chunked_run(sampler1, net1, z0_1, steps)
    _, E = get_fourier_spectrum(out.squeeze(1).numpy())
    e10 = relerr(E, hi=10); e20 = relerr(E, hi=20); e30 = relerr(E, hi=30)
    results_1mask[steps] = (e10, e20, e30)
    print(f'  {steps:>6}  {e10:>10.4f}  {e20:>10.4f}  {e30:>10.4f}')

del net1
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# (c) mask K=6 multi-net (just trained)
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 72)
print('(c) mask K=6 multi-net  (results/mask_K6_multi_proper)')
print('=' * 72)

hier6 = HierarchicalMasks(G, num_masks=6, device=device)
cond_vars6 = estimate_conditional_variances(var_data, [m.cpu() for m in hier6.masks])
print(f'  cond_vars (K=6): {[f"{v:.2e}" for v in cond_vars6]}')
noise_std_6 = build_noise_std(hier6.masks, cond_vars6, device=device)

net6 = MultiVelocity(hier6, channels_per_net=24, dim_mults=(1, 2, 2)).to(device)
sd6 = torch.load('results/mask_K6_multi_proper/model_final.pt', map_location=device)
net6.load_state_dict(sd6)
net6.eval()

sampler6 = MaskSampler(num_masks=6)

def make_z0_6(B):
    z = torch.randn(B, 1, G, G, device=device)
    return z * noise_std_6[None, None, :, :]

torch.manual_seed(42)
z0_6 = make_z0_6(n_test)

print(f'  {"steps":>6}  {"k≤10":>10}  {"k≤20":>10}  {"k≤30":>10}')
results_mask_uni = {}
for steps in [12, 24, 48, 96]:
    out = chunked_run(sampler6, net6, z0_6, steps, grid_mode='uniform')
    _, E = get_fourier_spectrum(out.squeeze(1).numpy())
    e10 = relerr(E, hi=10); e20 = relerr(E, hi=20); e30 = relerr(E, hi=30)
    results_mask_uni[steps] = (e10, e20, e30)
    print(f'  {steps:>6}  {e10:>10.4f}  {e20:>10.4f}  {e30:>10.4f}')

del net6
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# (b) Wavelet K=3 single-net
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 72)
print('(b) Wavelet K=3 single-net  (results/s3_wav3_50k)')
print('=' * 72)

# Import wavelet machinery
from train_wavelet_multiscale_v2 import (
    HaarWavelet, WaveletInterpolant, WaveletVelocity, WaveletSampler,
)

haar3 = HaarWavelet(G, num_levels=3, device=device)
print(f'  num_phases = {haar3.num_phases}')

net_w = WaveletVelocity(haar3, channels=32, dim_mults=(1, 2, 2, 2)).to(device)
sd_w = torch.load('results/s3_wav3_50k/model_final.pt', map_location=device)
net_w.load_state_dict(sd_w)
net_w.eval()

sampler_w = WaveletSampler(haar3.num_phases)

# wavelet trainer's noise: per-band variance from estimate (separately computed in pixel space)
# We use the same construction as the wavelet trainer.
from train_wavelet_multiscale_v2 import Trainer as WTrainer

# Actually we need just the noise std building. The wavelet trainer puts iid
# noise per pixel, with std varying per pixel based on which active band the
# pixel ends up in via wavelet projection. Let's reuse the same logic — for
# the test we just need a reasonable z0. Easiest: sample same way as the
# trainer's `make_noise_iid` would for K=3.
# Look up trainer's exact noise computation:
import torch.nn.functional as F_nn

# Wavelet trainer's noise: per-sub-band variance estimated from data, then
# noise sampled in coefficient space and reconstructed to pixel space.
ll_var, detail_vars = haar3.estimate_sub_band_variances(var_data.to(device))
print(f'  ll_var={ll_var:.4f}, detail_vars[0]={detail_vars[0]}')

def make_z0_wav(B):
    pix, _, _ = haar3.make_noise(B, ll_var, detail_vars)
    return pix.unsqueeze(1)

torch.manual_seed(42)
z0_w = make_z0_wav(n_test)

# Try wavelet sampler with both grid modes if available
print(f'  {"steps":>6}  {"k≤10":>10}  {"k≤20":>10}  {"k≤30":>10}')
results_wav = {}
for steps in [12, 24, 48, 96]:
    outs = []
    for i in range(0, n_test, 200):
        out = sampler_w.RK4(z0_w[i:i+200].clone(), net_w, steps=steps)
        outs.append(out.cpu())
    out = torch.cat(outs, dim=0)
    _, E = get_fourier_spectrum(out.squeeze(1).numpy())
    e10 = relerr(E, hi=10); e20 = relerr(E, hi=20); e30 = relerr(E, hi=30)
    results_wav[steps] = (e10, e20, e30)
    print(f'  {steps:>6}  {e10:>10.4f}  {e20:>10.4f}  {e30:>10.4f}')

del net_w
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print()
print('=' * 72)
print('SUMMARY: mean |E_gen / E_truth - 1| over k=1..30  (lower = better)')
print('=' * 72)
print('  k≤10:')
print(f'  {"steps":>6}  {"1-mask FM":>11}  {"wavelet K=3":>13}  {"mask K=6":>11}')
all_steps = sorted(set(list(results_1mask.keys()) + list(results_mask_uni.keys()) + list(results_wav.keys())))
for st in all_steps:
    a = results_1mask.get(st)
    b = results_wav.get(st)
    c = results_mask_uni.get(st)
    fmt = lambda x: f'{x[0]:11.4f}' if x is not None else f'{"--":>11}'
    print(f'  {st:>6}  {fmt(a)}  {fmt(b):>13}  {fmt(c):>11}')

print('  k≤20:')
print(f'  {"steps":>6}  {"1-mask FM":>11}  {"wavelet K=3":>13}  {"mask K=6":>11}')
for st in all_steps:
    a = results_1mask.get(st)
    b = results_wav.get(st)
    c = results_mask_uni.get(st)
    fmt = lambda x: f'{x[1]:11.4f}' if x is not None else f'{"--":>11}'
    print(f'  {st:>6}  {fmt(a)}  {fmt(b):>13}  {fmt(c):>11}')

print('  k≤30:')
print(f'  {"steps":>6}  {"1-mask FM":>11}  {"wavelet K=3":>13}  {"mask K=6":>11}')
for st in all_steps:
    a = results_1mask.get(st)
    b = results_wav.get(st)
    c = results_mask_uni.get(st)
    fmt = lambda x: f'{x[2]:11.4f}' if x is not None else f'{"--":>11}'
    print(f'  {st:>6}  {fmt(a)}  {fmt(b):>13}  {fmt(c):>11}')
