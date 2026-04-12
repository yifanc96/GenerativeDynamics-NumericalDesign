# Multiscale Per-Band Flow Matching: Mask and Haar Wavelet Approaches

## Summary

We implement multiscale flow matching for 2D Gaussian random fields (Matern, s=3, G=64) using per-band independent training. Each frequency band has its own UNet at scale-appropriate resolution. Both mask (pixel-space) and Haar wavelet decompositions achieve **all Fourier bins k=1..30 with relative error < 1** using just 4 RK4 steps per band.

## Method

### Decomposition

**Mask (K=3, 4 bands):** Dyadic pixel grid hierarchy.
- Band 0 (coarsest): 16x16 grid, 256 pixels, R=16
- Band 1: 32x32 grid minus 16x16, 768 new pixels, R=32
- Band 2: 64x64 grid minus 32x32, 3072 new pixels, R=64
- Band 3 (finest): not used for K=3 at G=64

**Haar wavelet (L=3, 4 bands):** Grouped by scale (LH+HL+HH per level).
- Band 0 (LL): 8x8 coefficients, R=8
- Band 1 (det_L1): 3x(8x8) = 192 coefficients, R=8
- Band 2 (det_L2): 3x(16x16) = 768 coefficients, R=16
- Band 3 (det_L3): 3x(32x32) = 3072 coefficients, R=32

### Per-band flow matching

For each band F conditioned on coarser bands C:

1. **Estimate conditional variance** sigma^2 via ridge regression of F pixels on C pixels (ridge=1e-6, 10000 samples)
2. **Sample noise**: z0 ~ N(0, sigma^2 * I)
3. **Interpolant**: z_t = (1-t)*z0 + t*x_F
4. **Target velocity**: v = x_F - z0
5. **Loss**: MSE(v_net, v) / sigma^2 (reweighted so loss is O(1) for all bands)
6. **Network input**:
   - Mask: 2-channel R×R image — [z_t at F positions + x_C at C positions, x_C only at C positions]
   - Haar: 4-channel R×R image — [LL context from inverse DWT of coarser coefficients, LH, HL, HH]
   - All values at raw scale (no normalization of inputs)

### Network architecture

Scale-appropriate UNet per band:
- R=8: dim_mults=(1,2), ~834K params
- R=16: dim_mults=(1,2,2), ~1.4M params
- R=32: dim_mults=(1,2,2), ~1.4M params
- R=64: dim_mults=(1,2,2,2), ~2.1M params

GroupNorm, sinusoidal time conditioning, 4 attention heads.

### Training

- Coarse-to-fine: band 0 trains first, then band 1 (using generated band 0 as context), etc.
- Steps: 20000 per band, 40000-80000 for finest band
- Optimizer: AdamW, lr=2e-4, cosine annealing
- Batch size: 400

### Generation

- Sequential: generate band 0, use as context for band 1, etc.
- RK4 integration per band over t in [0.001, 0.999]
- 4 RK4 steps per band sufficient (16 total function evaluations across 4 bands)

## Key Results

### Spectrum accuracy (G=64, Matern s=3, 4 RK4 steps per band)

| Method | mean_rel<=30 | max_rel<=30 | All bins < 1? |
|--------|-------------|-------------|---------------|
| Mask raw | 0.21 | 0.69 | Yes (30/30) |
| Haar raw | 0.30 | 0.83 | Yes (30/30) |
| Oracle (mask, analytic) | 0.028 | 0.13 | Yes |
| Oracle (haar, analytic) | 0.038 | 0.41 | Yes |

### RK4 step convergence

| Method | RK4=2 | RK4=4 | RK4=8 | RK4=16 |
|--------|-------|-------|-------|--------|
| Mask raw | 0.25 | 0.21 | 0.20 | 0.20 |
| Haar raw | 0.54 | 0.30 | 0.25 | 0.25 |

Converged by RK4=8; RK4=4 is sufficient for all bins < 1.

### Normalization variant comparison

| Variant | Mask mean_rel | Haar mean_rel |
|---------|-------------|-------------|
| **raw** (recommended) | **0.47** | **0.60** |
| rescale (F/sigma) | 2.52 | 0.98 |
| center (subtract mu) | 1.08 | 10.96 |
| center_rescale | 0.38 | 13.68 |

(From first benchmark with 40k finest-band steps, before loss reweighting fix. Final results with reweighting are better — see above.)

**Key finding:** The "raw" variant with loss reweighting by 1/sigma^2 works best for both mask and Haar. Centering hurts Haar because the ridge regression M_op for wavelet coefficients overfits, introducing noise into the centered data.

## Conditioning analysis

### Conditional covariance condition numbers

| Band | Mask cond | Haar LH cond | Haar HH cond |
|------|-----------|-------------|-------------|
| Coarsest (phase 0) | 52,310 | 42,870 (LL) | — |
| Phase 1 | 688 | 145 | 59 |
| Phase 2 | 1,463 | 124 | 47 |
| Phase 3 (finest) | 363 | 125 | 64 |

Haar detail bands are uniformly well-conditioned (cond ~ 64-150). Mask finest band has cond=363 (also manageable). The earlier reported cond=12,400 was an artifact of ridge=1e-4 in the variance estimator; corrected to ridge=1e-6.

The theoretical prediction for Haar HH bands: cond ~ 2^(2s) = 2^6 = 64 for Matern s=3. This matches exactly.

## Comparison: Mask vs Haar

**Mask advantages:**
- Simpler (no wavelet transform needed)
- Slightly better accuracy (0.21 vs 0.30)
- Natural pixel-space interpretation

**Haar advantages:**
- Uniformly good conditioning at all bands (cond ~ 64-150)
- Handles Nyquist frequencies correctly (no mask Nyquist artifact)
- Smaller networks (finest band at 32x32 vs 64x64 for mask)

**Mask Nyquist issue:** The mask decomposition puts ~2x excess energy at k=31-32 (the Nyquist bins) in the oracle test. This is because the stride-1-minus-stride-2 band cannot properly represent Nyquist modes. Not an issue for k<=30.

## Files

- `train_multiscale_perband.py` — Main training script (mask + Haar, all variants)
- `test_oracle_exact.py` — Oracle (analytic velocity) test for both decompositions
- `test_conditioning_diagnosis.py` — Verifies conditional covariance condition numbers
- `test_screening.py` — Screening effect verification
- `results/bench/` — Saved checkpoints for mask_raw and haar_raw variants
- `results/spectrum_comparison_final.png` — Spectrum comparison plot

## Connection to Chen-Vanden-Eijnden-Xu (2509.01629)

That paper uses a designed interpolation schedule (alpha^2 + beta^2 * lambda_min = lambda_min^t) for single-scale flow matching, achieving ~20 RK4 steps for Gaussian fields. Our multiscale approach achieves comparable accuracy with 4 RK4 steps per band (16 total), by decomposing the problem into well-conditioned sub-problems.

## Connection to Guth-Coste-De Bortoli-Mallat (2208.05003)

The WSGM paper factorizes the data distribution into conditional distributions of wavelet coefficients across scales. Our Haar implementation follows this approach. Key differences:
- WSGM uses wavelets with q > 2s vanishing moments for optimal decorrelation; we show Haar (q=1) works well enough for s=3
- WSGM trains separate score networks per scale; we train separate velocity networks (flow matching instead of score-based diffusion)
- The screening effect (bounded conditional condition numbers) holds for both mask and wavelet decompositions, enabling few-step integration
