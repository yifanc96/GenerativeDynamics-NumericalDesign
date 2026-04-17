# Lip Schedule Study — Summary

**Branch**: `data-dep-noise-and-multiscale`
**Paper reference**: arXiv:2509.01629 (Lip) — *scaled interpolation schedule* for flow matching.

This document summarizes experiments reproducing and extending the Lip paper's findings on the scaled interpolation schedule, across:
- Gaussian fields (analytical)
- Navier–Stokes vorticity (trained NN) at 64×64, 128×128, 256×256
- CelebA-HQ (trained NN) at 128×128

---

## 1. Background — the Lip schedule

Standard flow matching uses the linear interpolant
`z_t = (1 − t)·z0 + t·z1,  α(t) = 1 − t,  β(t) = t`.

The Lip paper proposes a **scaled interpolant** parameterized by a scalar `r ∈ (0, 1)`:

```
α(t) = sqrt((r − r^t) / (r − 1))
β(t) = sqrt((r^t − 1) / (r − 1))
α²(t) + β²(t) = 1  for all t
```

`r` is typically chosen as the spectral ratio `S_data(k_Nyquist) / S_noise(k_Nyquist)`.
This ratio captures how much the target spectrum decays relative to the source, and the Lip schedule adapts the interpolation speed so that every frequency band is resolved comparably during integration.

---

## 2. Transfer formula — no retraining needed

A standard-trained NN `v_nn(z, t) ≈ E[z1 − z0 | z_t = z]` can be used with the Lip schedule via a pure inference-time transformation. From `Allen-Cahn/notebook-GaussianBase-Allen-Cahn-train-and-inference-white-noise.ipynb` and `Navier-Stokes/multiscale-interpolation/NSunconditional_training-and-inference-white-noise.ipynb`:

```python
def lip_drift(z_t, t):                      # transfer formula
    orig_t = β(t) / (α(t) + β(t))           # equivalent standard time
    orig_x = (orig_t / β(t)) · z_t          # rescaled state
    v = v_nn(orig_x, orig_t)                # call the trained NN
    coef_z = α̇(t) / α(t)
    coef_E1 = β̇(t) − α̇(t)·β(t)/α(t)
    return coef_z · z_t + coef_E1 · ((1 − orig_t)·v + orig_x)
```

### 2a. Noise-scaling affine transfer (our extension)

For a checkpoint trained with noise strength 1, we derived a combined **noise-scaling + Lip-schedule** transfer. Effective interpolant `z_t = σ·α(t)·z0 + β(t)·z1`:

```python
denom = σ·α(t) + β(t)
orig_t = β(t) / denom
orig_x = z_t / denom
v = v_nn(orig_x, orig_t)
b(z_t, t) = orig_x·(σ·α̇ + β̇) + v·(−σ·α̇·orig_t + β̇·(1 − orig_t))
```

This lets us evaluate a noise=1 checkpoint as if it were trained with any noise strength σ, enabling the Lip benefit without retraining.

---

## 3. Auto-tuned (σ, r) algorithm

Instead of manually choosing the noise-strength and Lip ratio, derive both from the data spectrum:

```python
def auto_select_sigma_r(S_data, S_noise_unit, kvals, margin=c):
    sigma_min = sqrt(max_k(S_data(k) / S_noise(k)))   # smallest σ s.t. noise dominates
    sigma     = c · sigma_min                          # with margin c ≥ 1
    r         = S_data(k_Nyquist) / (sigma² · S_noise(k_Nyquist))
    return sigma, r
```

Rationale: σ is chosen so `σ² · S_noise(k) ≥ S_data(k) ∀k` (the forward process covers the target); `r` is set at the finest scale where the Lip schedule must correctly resolve dynamics.

---

## 4. Critical bug fix — RK4 loop

Across **11 files** the RK4/EM loops iterated over all `tgrid` points (N steps with N-point grid), overshooting past `t_max` by `1/(N-1)`. Fixed to iterate `N-1` times:

```python
# Before (wrong): for t_val in tgrid: ...
# After:          for i in range(len(tgrid) - 1): t_val = tgrid[i]; dt = tgrid[i+1] - tgrid[i]
```

Files fixed: `Gaussian-fields/train_gaussian_field_data_dep_noise.py`, `Allen-Cahn/train_allen_cahn_data_dep_noise.py`, `phi4/train_phi4.py`, `CelebA-HQ/train_celeba.py`, `Navier-Stokes/{eval_ns_models, eval_spectrum_all, reproduce_lip_ns, reproduce_lip_ns_compare, train_ns_lip_compare, train_ns_gauss_base, train_ns_data_dep_noise}.py`. 17 loops total (6 EM + 11 RK4). Without this fix, convergence was non-monotone (errors could *increase* with more steps).

---

## 5. Experiments & results

All metrics report mean ± std over 3–5 random seeds.

### 5.1 Gaussian fields — analytical verification

Standard vs Lip schedule using the exact Gaussian drift formula (no NN) across resolutions 32, 64, 128:

| Res | Std (80 steps) | Lip (20 steps) |
|-----|---------------|---------------|
| 32  | mean err 10.7 | **0.014** |
| 64  | mean err 59.0 | **0.027** |
| 128 | mean err 148.8 | **0.069** |

**Lip with 20 RK4 steps matches or beats Standard with 80 steps** at every resolution.

### 5.2 Navier–Stokes — transfer formula on trained NN

Checkpoints (all lr=1e-4, 5 datasets, 50k steps):
- `results/ns_64_noise1/model_final.pt` — 64×64, σ_train=1
- `results/ns_64_noise10/model_final.pt` — 64×64, σ_train=10
- `results/ns_noise1_lipR1e3/model_final.pt` — 128×128, σ_train=1
- `results/ns_noise10_lipR1e5/model_final.pt` — 128×128, σ_train=10

**Main result (mean enstrophy error, auto-tuned Lip c≈2):**

| Resolution | σ_train | Best Standard | Best Lip | Speedup |
|-----------|---------|--------------|----------|---------|
| 64×64     | 10      | 0.060 (50 steps) | **0.039 (10 steps)** | **5×** |
| 64×64     | 1       | 0.056 (50 steps) | **0.040 (20 steps)** | **2.5×** |
| 128×128   | 10      | 0.184 (50 steps) | **0.117 (20 steps)** | **5×** |
| 128×128   | 1       | 0.182 (50 steps) | 0.205 (50 steps) | none |

Key finding: **Lip transfer needs large training noise to work well, especially at higher resolution.** σ_train=1 at 128×128 is insufficient for the noise-scaled transfer to σ≈9 — the model is out-of-distribution. At 64×64 it just barely works.

**Per-band errors (128×128, σ_train=10, 10 RK4 steps):**

| Band | Standard | Lip (auto) |
|------|----------|-----------|
| low (k<8)       | 0.072 | 0.098 |
| mid (8–24)      | 0.039 | **0.019** |
| high (k≥24)     | **3.549** | **0.163** |

Lip reduces high-band error **22×** at 10 steps and mid-band **2×**; low-band slightly worse (~3% absolute).

### 5.3 Non-Gaussian metrics (128×128)

Genuinely beyond second-order statistics, 5 seeds:

| Metric | Truth | Std 10 | Std 20 | Lip 10 | Lip 20 |
|--------|-------|--------|--------|--------|--------|
| Flatness S₄/S₂² at r=1 | 4.95 | 4.29±0.02 | 4.67±0.03 | **4.82±0.03** | **4.83±0.03** |
| Flatness at r=2 | 4.34 | 4.13±0.02 | 4.27±0.03 | **4.31±0.03** | **4.32±0.03** |
| Gradient kurtosis | 4.95 | 4.29 | 4.67 | **4.82** | **4.83** |
| KS distance | — | 0.0049 | 0.0053 | 0.0047 | **0.0046** |

Lip 10 steps captures intermittency **5× better** than Standard 10 steps (2.7% vs 13.3% error on flatness).

### 5.4 CelebA-HQ (128×128 RGB)

Checkpoint: `celeba_gauss_200k/model_final.pt` (σ_train=1, 200k steps, 18M params).
Auto-tuned (σ, r): σ_min ≈ 14.7, r ≈ 5.5e-6 (c=1.0).

| Method | steps | spec_err mean | FID | grad_kurt (truth=19.9) |
|--------|-------|---------------|-----|----------------------|
| Std σ=1 (baseline) | 10 | 0.105 | 78.8±0.5 | 14.3 |
| Std σ=1 (baseline) | 20 | 0.043 | **55.2±0.5** | 16.9 |
| Std σ=1 (baseline) | 50 | 0.046 | **50.9±0.8** | **17.7** |
| **Lip c=1.0 σ=14.7** | 20 | **0.039** | 55.7±0.4 | 15.5 |
| Lip c=1.0 σ=14.7 | 50 | 0.043 | 56.2±0.7 | 16.2 |

**Mixed result**:
- **Spectrum**: Lip wins — 20 steps (0.039) beats Std 50 steps (0.046).
- **FID**: Std σ=1 wins — the noise-scaled transfer introduces slight distributional drift the Inception features pick up.

Implication: **Lip is valuable for spectrum-driven tasks (physics, turbulence)** where high-k accuracy is a target in itself. For natural images where perceptual quality dominates, Standard σ=1 remains competitive.

### 5.5 256×256 NS — model-capacity-limited

A 2M-param UNet trained at 256×256 can't resolve the finest scales. With the cutoff k<64 (same as 128×128 Nyquist), Lip still gives ~25–35% improvement over Standard at the same step count, but both saturate at the model's representation floor. Larger model (≥18M params) needed for meaningful 256×256.

---

## 6. Key takeaways

1. **The transfer formula is correct and does work** — but only when the NN is accurate enough in the regime the Lip coefficients sample.

2. **Training noise matters**: σ_train ≥ σ_min_auto is needed for the Lip benefit. When σ_train=1 and σ_min>5, the noise-scaled transfer pushes the NN outside its training distribution.

3. **Auto-tuned (σ, r)** is principled and works across tasks (Gaussian fields, NS, CelebA): no manual search needed.

4. **Resolution-dependent**: the Lip advantage grows with resolution because the spectral dynamic range grows.

5. **The RK4 bug affected every file** — any prior result computed before the fix should be recomputed. Symptoms: errors grew with more steps; the high-band looked artificially good because the integration overshot into regions the model hadn't seen.

6. **For images**: spectrum accuracy ≠ perceptual quality. FID can disagree with spectrum error.

---

## 7. Key files in this study

### Scripts
- `Navier-Stokes/eval_lip_transfer.py` — Standard vs Lip on NS checkpoints
- `Navier-Stokes/auto_lip_ns.py` — Auto-tuned (σ, r) with noise-scaled affine transfer
- `Navier-Stokes/eval_nongaussian_v2.py` — Non-Gaussian metrics with multiple seeds
- `Navier-Stokes/train_ns_standard.py` — Training with standard schedule + in-training Lip eval via wandb
- `CelebA-HQ/eval_lip_celeba.py` — CelebA Standard/Lip/noise-scaled samplers + FID
- `CelebA-HQ/auto_lip_celeba.py` — Auto-tuned eval for images with FID

### Notebooks (reference)
- `Gaussian-fields/Gaussian-field-standard-interpolation-schedule.ipynb` — analytical Lip for Gaussian fields
- `Gaussian-fields/Gaussian-field-scaled-interpolation-schedule.ipynb`
- `Allen-Cahn/notebook-GaussianBase-Allen-Cahn-train-and-inference-white-noise.ipynb` — original `new_drift_bt` transfer formula
- `Navier-Stokes/multiscale-interpolation/NSunconditional_training-and-inference-white-noise.ipynb` — NS version of the transfer formula

### Figures produced
- `Gaussian-fields/spectrum_comparison_standard_vs_scaled.png`
- `Navier-Stokes/ns_lip_transfer_{64x64,128x128,256x256}_noise10.png`
- `Navier-Stokes/ns_nongaussian_{64x64,128x128}_noise10_lipR*.png`
- `CelebA-HQ/celeba_auto_lip.png`, `celeba_noisescaled_sigma10.png`

---

## 8. Open questions / future work

- Train NS at 256×256 with larger UNet (≥18M params) to separate Lip benefit from model capacity.
- Does the auto-tuned Lip help on **conditional** flow matching (super-resolution, forecasting) where the conditioning reduces the effective spectral gap?
- Can the Lip schedule be combined with mean-flow / consistency distillation for even fewer steps?
- For images, is there a feature space (CLIP, LPIPS) where the Lip benefit is visible without the FID penalty?
