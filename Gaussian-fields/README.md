# Gaussian Fields — Multiscale Per-Band Flow Matching

Branch: `data-dep-noise-and-multiscale`

## Overview

Flow matching for 2D Matern Gaussian random fields with multiscale decomposition. The data distribution is Gaussian so the optimal velocity has a closed form — useful as a testbed to verify the multiscale approach. Implements both **mask** (pixel-space) and **Haar wavelet** decompositions with per-band independent training.

## Method

For each band F (with coarser bands in C):
1. Estimate conditional variance σ² via ridge regression
2. Sample noise z0 ~ N(0, σ²I)
3. Interpolant: zt = (1-t)·z0 + t·x_F,   target: x_F - z0
4. Loss: MSE / σ² (reweighted so all bands have O(1) loss)
5. Scale-appropriate UNet per band at resolution R

Trained coarse-to-fine; each band's network conditions on previously generated coarser bands (through the input image).

## Results (G=64, Matern s=3, 4 RK4 steps per band)

| Method | mean_rel≤30 | max_rel≤30 | All bins < 1? |
|--------|-------------|-------------|---------------|
| Oracle mask (analytic) | 0.028 | 0.13 | Yes |
| Oracle haar (analytic) | 0.038 | 0.41 | Yes |
| **Mask network** | **0.21** | 0.69 | **Yes (30/30)** |
| **Haar network** | **0.30** | 0.83 | **Yes (30/30)** |

Spectrum plot: `results/spectrum_comparison_final.png`

## Code

### Main script
- **`train_multiscale_perband.py`** — full training + eval for mask and Haar, supports 4 normalization variants:
  - `raw` (recommended): `z0 ~ N(0, σ²I)`, target = `x_F - z0`, loss = `MSE/σ²`
  - `rescale`, `center`, `center_rescale` (for ablation)
  
  Usage:
  ```
  python train_multiscale_perband.py --decomp mask --variant raw --gpu 0
  python train_multiscale_perband.py --decomp haar --variant raw --gpu 1
  ```

### Sanity checks and diagnostics
- **`test_oracle_exact.py`** — oracle (analytic velocity) test for both decompositions
- **`test_conditioning_diagnosis.py`** — verifies conditional covariance condition numbers (~363 for mask, ~64-150 for Haar detail bands)
- **`test_oracle_mask.py`**, **`test_oracle_wavelet.py`** — older oracle tests
- **`test_screening.py`**, **`test_mask_correct.py`** — screening effect verification

### Related (pre-branch)
- **`train_multiscale_interpolation.py`** — original multiscale trainer (HierarchicalMasks definition)
- **`train_wavelet_multiscale_v2.py`** — older Haar wavelet trainer
- **`train_gaussian_field_data_dep_noise.py`**, **`train_gaussian_field_meanflow.py`** — 1-mask baselines

### Saved models
- `results/bench/mask_raw_K3/` — mask variant checkpoints (4 bands: R=8, 16, 32, 64)
- `results/bench/haar_raw_K3/` — Haar variant checkpoints (4 bands: LL, det_L1..L3)

### Report
- **`REPORT_multiscale_perband.md`** — detailed report with conditioning analysis and method comparison

## Key Findings

1. **Screening effect verified**: conditioning fine bands on coarser bands gives bounded condition numbers (~64-360). Earlier cond=12,400 was a ridge parameter bug (ridge=1e-4 → 1e-6 fix).

2. **Raw variant with 1/σ² loss reweighting wins**: simpler than centering, works for both mask and Haar. Centering hurts Haar (ridge M_op overfits).

3. **Mask Nyquist issue**: the finest mask band has ~2× excess energy at k=31-32 in the oracle — can't be fixed without more bands or wavelet decomposition.

4. **Haar advantage**: uniformly good conditioning and handles Nyquist correctly. Disadvantage: slightly worse network accuracy (0.30 vs 0.21 for mask).

## Connection to Papers

- **Chen-Vanden-Eijnden-Xu (2509.01629)**: designed schedules for single-scale FM — our multiscale approach achieves comparable accuracy with 4× fewer steps by decomposing into well-conditioned subproblems.
- **Guth-Coste-De Bortoli-Mallat (2208.05003, WSGM)**: wavelet score-based diffusion — our Haar implementation follows the same philosophy (factorize into conditional distributions of wavelet coefficients across scales).
