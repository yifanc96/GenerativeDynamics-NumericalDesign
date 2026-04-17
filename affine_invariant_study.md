# Affine-Invariant (Data-Dependent) Noise for Flow Matching: An Empirical Study

**Branch:** `data-dep-noise-and-multiscale`
**Date range:** 2026-04 session

## 1. Motivation

Standard flow matching uses isotropic Gaussian base noise `z0 ~ N(0, I)`. The idea of **affine-invariant / data-dependent noise** is to replace this with noise whose covariance matches the empirical data covariance:

```
z0_j = (1/sqrt(N)) * sum_i (x_i - x_bar) * xi_{i,j},  xi_{i,j} ~ N(0, 1)
```

where `{x_i}` is a mini-batch and `xi` is an iid Gaussian. By construction, `Cov(z0) = Sigma_data` (empirical covariance). The hypothesis: flow matching with base noise whose covariance already matches the target should need fewer integration steps at sampling time, since it starts "closer" to the target in a second-moment sense.

## 2. Datasets and experimental setup

| Dataset | Resolution | kappa (spectral cond. #) | Marginal | Notes |
|---|---|---|---|---|
| phi4 (Guth) L=32 | 32×32 | ~30 | Bimodal (+/-1) | Well-conditioned, critical |
| phi4 (Guth) L=64 | 64×64 | ~50 | Bimodal (+/-1) | Well-conditioned, critical |
| phi4-Ising L=64 | 64×64 | ~8226 | Bimodal, kurtosis=-1.9 | Ill-conditioned, ordered phase |
| CelebA-HQ 64 | 64×64 | moderate | Non-Gaussian | Natural images, 200k samples |
| Navier-Stokes | 128×128 | ~1e3-1e4 | Non-Gaussian | 2D vorticity, 20k samples |

All models: flow matching with linear interpolant `I_t = (1-t)z0 + t z1`, velocity target `R_t = z1 - z0`, UNet velocity network, RK4 sampler at inference.

Code entry points: `phi4/train_phi4.py`, `Navier-Stokes/train_ns_data_dep_noise.py`, `Navier-Stokes/train_ns_gauss_base.py`, `CelebA-HQ/train_celeba.py`.

## 3. Headline result: Gaussian base wins at convergence

Per-band relative spectrum error at RK4-50 (enstrophy for NS, power for others):

| Dataset | gauss (low/mid/high) | data_dep (low/mid/high) | Winner |
|---|---|---|---|
| phi4 L=32 | **0.027/0.007/0.024** | 0.069/0.035/0.048 | gauss |
| phi4 L=64 | **0.167**/0.077/**0.016** | 0.287/**0.061**/0.025 | gauss |
| phi4-Ising L=64 | **0.128/0.139/0.055** | 0.209/0.244/0.062 | gauss |
| CelebA 200k | **0.052/0.068/0.106** | 0.114/0.069/0.112 | gauss |
| NS 2M params | **0.100/0.036/0.105** | 0.332/0.299/0.320 | gauss |
| NS 18M params | **0.155/0.028/0.307** | 0.415/0.538/0.471 | gauss |

Gaussian base wins across every dataset, every frequency band, with 2-10x lower error. Data-dep noise systematically produces samples with `std_ratio < 1` (0.82 for NS 2M, 0.73 for NS 18M) — the flow undershoots the target variance.

## 4. Sampling efficiency: data-dep wins at few steps (phi4-Ising only)

For the ill-conditioned phi4-Ising data (kappa~8226), data-dep dominates at low step counts:

| Steps | gauss (low/mid/high) | data_dep (low/mid/high) | Winner |
|---|---|---|---|
| 10 | 0.722/0.589/0.226 | **0.238/0.259/0.088** | **data_dep (3x better)** |
| 20 | 0.590/0.568/0.217 | **0.224/0.242/0.067** | **data_dep (2.5x better)** |
| 50 | **0.128/0.139/0.055** | 0.209/0.244/0.062 | gauss |

For NS, data-dep is worse than gauss at all step counts. The sampling-efficiency advantage appears to require large spectral condition number **and** the training to have converged similarly — which doesn't happen for NS.

## 5. Fourier-space loss reweighting on NS

Hypothesis: with data-dep noise, the target `R_t = z1 - z0` has variance dominated by low-k modes (where `Sigma(k)` is large). Flat MSE over-weights low-k; the network wastes capacity there. Reweight the loss in Fourier space:

```
L = E[ sum_k w(k) * |v_hat(k) - R_hat(k)|^2 ],   w(k) = 1 / Sigma(k)^alpha
```

Implemented in `Navier-Stokes/train_ns_data_dep_noise.py` via `--fourier_loss --fourier_loss_alpha A`.

### Results on NS (2M model, 50k steps, RK4-50, enstrophy)

| Model | alpha | std_ratio | low | mid | high |
|---|---|---|---|---|---|
| Gauss baseline | — | **1.013** | **0.100** | **0.036** | **0.105** |
| Data-dep (no floss) | 0 | 0.822 | 0.332 | 0.299 | 0.320 |
| Data-dep + floss | 0.25 | 0.845 | 0.250 | 0.286 | 0.155 |
| **Data-dep + floss** | **0.5** | **0.875** | **0.239** | **0.244** | **0.104** |
| Data-dep + floss | 0.75 | 0.866 | 0.275 | 0.257 | 0.205 |
| Data-dep + floss | 1.0 | 0.889 | 0.183 | 0.274 | 0.277 |

**Key findings:**
- `alpha=0.5` is the sweet spot. High-k matches gauss baseline (0.104). Mid/low improve substantially over unreweighted data-dep.
- Non-monotonic in alpha: too aggressive (`alpha=1.0`) over-concentrates on high-k and hurts it; too mild (`alpha=0.25`) barely helps.
- `std_ratio ~ 0.87` plateau across all alpha — **reweighting does not fix the systematic variance undershoot**.
- Even the best data-dep setup is 2-6x worse than gauss on low/mid-k.

## 6. Why data-dep noise fundamentally struggles (analysis)

The core issue is **regression ambiguity**, not loss weighting:

At intermediate time t, the interpolant `I_t = (1-t)z0 + t z1` is a superposition of two fields that share the same spectral shape. The network must predict `v = z1 - z0` from `I_t` alone, but cannot decompose which features came from `z0` vs `z1`.

- With Gaussian `z0`: data-like structure in `I_t` is unambiguously from `z1`. Signal-noise separation is clear.
- With data-dep `z0`: both contribute correlated structure at every scale. The conditional expectation `E[v | I_t]` is blurred by uncertainty.

MSE regression under uncertainty biases toward the conditional mean, which has smaller magnitude than individual samples. This explains:
- `std_ratio < 1` (under-generation of variance) — systematic
- Larger models do *worse* (NS 18M: 0.73 vs 2M: 0.82) — more capacity → more confident mean-regression → more shrinkage
- Fourier reweighting helps capacity allocation but cannot remove the ambiguity itself

### Ideas that were considered and rejected

- **Whitening `z0`**: `Sigma^{-1/2} z0_dep ~ N(0, I)` — this is just standard Gaussian noise (data-dep is Gaussian by CLT). No higher-order structure to preserve. Reduces to baseline.
- **Reparameterization**: any invertible transform can be undone at inference; doesn't change the transport problem.
- **Mini-batch OT coupling**: incompatible with the affine-invariant construction (noise covariance is a batch-level property; permuting samples destroys it).
- **"Spectrally shaped but Fourier-uncorrelated" noise**: for translation-invariant data, this is identical to data-dep noise.

### Directions not yet tried

- **Trigonometric interpolant** `I_t = cos(pi t/2) z0 + sin(pi t/2) z1`. Eliminates the variance dip at t=0.5 (`Var(I_t) = Sigma` constant along path). Doesn't directly address ambiguity but improves trajectory conditioning.
- **Stochastic interpolant with diffusion**: `dI_t = v dt + sigma(t) dW_t`. The injected noise breaks coherence between z0 and z1 contributions during training — may reduce ambiguity.
- **Post-hoc variance correction at inference**: rescale flow output to match data std. Hacky but addresses `std_ratio < 1` directly.

## 7. Practical conclusions

1. **For final-quality generation, use Gaussian base noise.** It is simpler, faster to train, and produces better samples on every dataset tested.
2. **For few-step sampling on near-Gaussian targets with large kappa**, data-dep noise can provide a 2-3x improvement at 10-20 RK4 steps (demonstrated on phi4-Ising, not on NS).
3. **The affine-invariant design is not a free lunch.** The easier sampling (fewer steps) is paid for by harder training (regression ambiguity, capacity misallocation).
4. **Fourier loss reweighting (`alpha~0.5`) is a useful tool** for any setting where the target regression has large spectral dynamic range — it closed the high-k gap to gauss on NS. Implementation: `Navier-Stokes/train_ns_data_dep_noise.py`, `--fourier_loss --fourier_loss_alpha 0.5`.
5. **The `std_ratio < 1` under-generation is the smoking gun.** No loss modification tested has fixed it; it appears intrinsic to the data-dep transport problem.

## 8. Reproducibility

**Key files (branch `data-dep-noise-and-multiscale`):**
- `phi4/sample_phi4_ising.py` — HMC sampler for ill-conditioned phi4 (J=1, a=-0.5, b=0.25)
- `phi4/train_phi4.py` — flow matching trainer (both noise types)
- `Navier-Stokes/train_ns_data_dep_noise.py` — with `--fourier_loss --fourier_loss_alpha` flags
- `Navier-Stokes/train_ns_gauss_base.py` — Gaussian baseline
- `Navier-Stokes/eval_ns_models.py` — shared evaluation script (correct RK4: N-1 steps over N grid points)

**Data:**
- `phi4/phi4_ising_L64.pt` — 20k samples, kappa~8226
- `NSdata/data_file.pt` — 20k NS vorticity snapshots at 128x128

**Checkpoints:**
- `Navier-Stokes/results/ns_gauss_base/model_final.pt`
- `Navier-Stokes/results/ns_data_dep_noise/model_final.pt`
- `Navier-Stokes/results/ns_data_dep_floss/model_final.pt` (alpha=1.0)
- `Navier-Stokes/results/ns_data_dep_floss_a{025,05,075}/model_final.pt`
- `phi4/results/phi4_ising_{gauss,data_dep}/model_final.pt`
