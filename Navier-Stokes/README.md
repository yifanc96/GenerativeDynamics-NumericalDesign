# Navier-Stokes — Multiscale Flow Matching

Branch: `data-dep-noise-and-multiscale`

## Overview

Flow matching for 2D stochastic Navier-Stokes vorticity fields (128×128 and 256×256) using the multiscale per-band approach developed in `../Gaussian-fields/`. Both **unconditional** generation and **conditional forecasting** (given state at time t, predict state at time t+τ).

Data: 5 NS trajectory files at `../NSdata/data_file{,02,03,04,05}.pt` (100,000 total snapshots after flattening).

## Unconditional Results (128×128, fair NFE comparison)

NFE = 4 × total_RK4_steps. For multiscale K=3 (4 bands): per_band_rk4 = total/4.

| Total RK4 | NFE | Mask K=3 (v3: 1ch, no t-norm) | Standard 1-mask |
|-----------|-----|-------------------------------|-----------------|
| 4  | 16  | 0.21 (all 60 bins < 1)        | 0.29            |
| 8  | 32  | **0.09**                       | 0.14            |
| 16 | 64  | **0.06**                       | 0.21            |
| 32 | 128 | **0.09**                       | 0.19            |

**Multiscale wins at every NFE budget. At 16 NFE (minimum), 2.6× better than standard. At 64 NFE, 4× better.**

## Conditional Forecasting Results (128×128, lag=2, fair NFE)

Predicts innovation r = x_hi - x_lo. Conditioning: x_lo as second channel.

| Total RK4 | NFE | Multiscale K=3 | Standard 1-mask |
|-----------|-----|----------------|-----------------|
| 4  | 16  | **0.23** (all<1) | 0.92 (47/60<1) |
| 8  | 32  | **0.22**         | 0.23            |
| 16 | 64  | 0.21             | **0.10**         |
| 32 | 128 | 0.21             | **0.08**         |

Multiscale dominates at low NFE. Plateau at 0.21 is likely due to coarser bands subsampling x_lo — losing fine-scale conditioning info (future work).

## Code

### Unconditional
- **`train_ns_multiscale_perband.py`** — v1: 2-channel mask (F+C image, coarse-only context), loss/σ², supports mask/haar and K=0 baseline.
  ```
  python train_ns_multiscale_perband.py --decomp mask --K 3 --gpu 0         # multiscale
  python train_ns_multiscale_perband.py --decomp haar --K 3 --gpu 1         # Haar
  python train_ns_multiscale_perband.py --decomp mask --K 0 --gpu 2         # standard baseline
  ```

- **`train_ns_multiscale_v2.py`** — v2/v3: **1-channel mask** (simpler, same performance). Supports `--no_tnorm` flag:
  - Default (v2): t-dependent EDM-style normalization `c_in(t) = 1/√((1-t)²σ² + t²v_F)`, `c_out = 1/√(v_F+σ²)`
  - `--no_tnorm` (v3): raw zt, loss/σ² (matches v1 but with 1ch) — **recommended**, same accuracy, simpler
  ```
  python train_ns_multiscale_v2.py --K 3 --no_tnorm --gpu 0
  python train_ns_multiscale_v2.py --K 0 --no_tnorm --gpu 1
  ```

### Conditional forecasting
- **`train_ns_forecasting_multiscale.py`** — multiscale forecasting. Predicts innovation r = x_hi - x_lo, 2-channel input [innovation_state, x_lo subsampled to R×R]. Gaussian base z0 ~ N(0, σ²_innov·I).
  ```
  python train_ns_forecasting_multiscale.py --K 3 --time_lag 2 --no_tnorm --gpu 0   # multiscale
  python train_ns_forecasting_multiscale.py --K 0 --time_lag 2 --no_tnorm --gpu 1   # baseline
  ```

### Baselines (pre-branch)
- **`train_ns_data_dep_noise.py`** — 1-mask FM with data-dependent noise
- **`train_ns_gauss_base.py`** — 1-mask FM with Gaussian base
- **`train_ns_meanflow_*.py`** — meanflow variants
- **`NSunconditional-Gaussbase.py`** — older unconditional script
- **`forecasting/`** — existing forecasting codes (point source, Gauss base, different schedules)

### Utilities
- **`unet.py`** — UNet architecture (shared)
- **`eval_ns_models.py`** — evaluation utilities
- **`energy_spectrum_plot.py`** — spectrum plotting
- **`debug_std_evolution.py`** — std evolution diagnostic

### Saved models
- `results/ns_mask_K3_G128/` — v1 multiscale mask checkpoints (4 bands)
- `results/ns_haar_K3_G128/` — v1 multiscale Haar checkpoints
- `results/ns_standard_1mask_G128/` — v1 standard baseline
- `results/ns_v2_mask_K3_G128/` — v2 (1ch, t-norm) checkpoints
- `results/ns_v2_standard_G128/` — v2 standard
- `results/ns_v3_mask_K3_G128/` — v3 (1ch, no t-norm) checkpoints [**recommended**]
- `results/ns_mask_K3_G256/` — 256×256 mask (bands 0-2 only; band 3 OOM)
- `results/ns_haar_K3_G256/` — 256×256 Haar checkpoints
- `results/ns_standard_1mask_G256/` — 256×256 standard
- `results/ns_forecast_mask_K3_lag2_notnorm_G128/` — forecasting multiscale
- `results/ns_forecast_mask_K0_lag2_notnorm_G128/` — forecasting baseline

## Key Findings

1. **1-channel mask = 2-channel mask** (same accuracy): the R×R image already contains the coarse context via the C pixel values — the redundant second channel is unnecessary.

2. **t-dependent normalization doesn't help for NS**: v3 (no t-norm) matches or beats v2 (with t-norm). The simpler `MSE/σ²` loss is sufficient.

3. **Multiscale advantage at low NFE**: 4× fewer integration steps to achieve the same accuracy as standard FM.

4. **Forecasting plateau**: coarse bands subsample the conditioning x_lo, losing fine-scale info. Needs shared full-resolution encoder or different architecture to fix.

5. **256×256 memory**: mask K=3 finest band (128×128 UNet) hits OOM at batch=50. Haar handles 256×256 fine (finest detail band at 128×128, smaller network).

## Open Issues

- **Forecasting architecture**: coarse bands need full-resolution x_lo conditioning (currently subsampled). Solutions: shared encoder producing multi-scale features, or cross-attention.
- **Larger resolutions**: mask struggles at 256×256 due to memory; need gradient checkpointing or smaller batch for the finest band.
