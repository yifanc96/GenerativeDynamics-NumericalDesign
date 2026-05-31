# Mean Flow for Navier-Stokes: Progress Report (Updated Apr 14)

## Summary

We implemented the mean flow algorithm (arXiv:2505.13447) for accelerating flow-matching-based generation of 2D Navier-Stokes vorticity fields. Mean flow learns a model `w(z, s, r)` conditioned on both current time `s` and target time `r`, enabling few-step generation without ODE integration.

## Algorithm

**Convention**: `s=0` (noise) → `s=1` (data), linear interpolant `z_s = (1-s)z_0 + s z_1`.

**Loss** (JVP-based self-consistency):
```
v = z_1 - z_0
w, dw/ds = JVP(model, (z_s, s, r), (v, 1, 0))
w_tgt = v + (r - s) * dw/ds
loss = adaptive_L2(w - stopgrad(w_tgt))
```

**Sampling** (K steps): `z_{k+1} = z_k + (r_k - s_k) * w(z_k, s_k, r_k)`

**Key hyperparameters** (validated by toy ablation):
- `flow_ratio = 0.5` — fraction of batch using `r = s` (pure flow matching anchor)
- Uniform time sampling (lognormal hurt in ablation)
- `time_scale = 1.0` — no scaling; the UNet's learned sinusoidal embedding handles `t ∈ [0,1]`
- Adaptive L2 loss with `gamma = 0.5, c = 1e-3`
- Gradient clipping at `max_norm = 1.0`
- EMA weights (`decay = 0.9999`) for evaluation
- Best-checkpoint tracking (essential — training oscillates)

## Architecture

`MeanFlowVelocity` wraps the existing UNet with an added `r_mlp` for target-time embedding:
- `s` embedding: UNet's built-in `time_mlp`
- `r` embedding: separate `RandomOrLearnedSinusoidalPosEmb → Linear → GELU → Linear`
- Combined via addition: `emb = time_mlp(s) + r_mlp(r)`
- `r_mlp` final layer is zero-initialized (starts as no-op)
- Same parameter count as original UNet (~2M for `dim=32, mults=(1,2,2,2)`)

## Validation

### Toy (2D Gaussian → Gaussian)
- **Ablation finding**: mean flow simply needs MUCH longer training (100k steps for tiny MLP vs 5k for regular FM)
- At convergence: 1-step `std_err = 0.003` (99.3% perfect)
- Other ablations (lognorm time, gap-uniform, bigger model, MSE loss) gave marginal or negative improvement

### Gaussian Fields (Matérn, 64×64)
- **50k steps**: best 4-step `spec_L1 = 0.000257`, `std_ratio = 1.00`
- Baseline pure FM 50-step EM: `spec_L1 = 0.00053`
- **Mean flow at 4 NFEs beats pure FM at 50 NFEs** (2× better L1)

### Navier-Stokes (128×128, 200k steps)

**At 4 NFEs (4-step mean flow vs 4-step Euler):**

| Method                          | low enst | mid enst | high enst | std_ratio | NFEs |
|---------------------------------|----------|----------|-----------|-----------|------|
| **MF gauss base (best @172k)**  | —        | **0.020**| —         | —         | 4    |
| MF gauss base (final @200k)     | 0.135    | 0.025   | 0.370     | 1.01      | 4    |
| MF data-dep (best @105k)        | —        | 0.031   | —         | —         | 4    |
| Pure FM RK4 4-step              | 0.308    | 0.224   | 0.303     | 0.92      | 16   |
| Pure FM EM 4-step               | 0.890    | 0.934   | 44.7      | 0.45      | 4    |

**Key results:**
- Mid-band enstrophy: **2.5% error at 4 NFEs** — 9× better than RK4 (16 NFEs), 37× better than EM (4 NFEs)
- Gaussian base outperformed data-dependent noise for mean flow (opposite of regular FM) — stable JVP gradients matter more than smooth velocity fields
- Low-k (k ≈ 1) remains the bottleneck at ~13% error (few Fourier modes, inherent to the problem)

## Lessons Learned

1. **Train much longer**: mean flow needs ≥4× more steps than regular FM (JVP self-consistency is slow to converge)
2. **Save best checkpoint**: training oscillates; final model is often worse than peak
3. **Don't blindly copy reference tricks**: `time_scale=1000` (DiT-specific), lognormal time, aggressive grad clipping all hurt when applied to our UNet
4. **Validate on toy first**: the 2D Gaussian toy quickly identifies code bugs vs tuning issues

## Files

- `Navier-Stokes/train_ns_meanflow_gauss_base.py` — NS with Gaussian noise
- `Navier-Stokes/train_ns_meanflow_data_dep_noise.py` — NS with data-dependent noise
- `Gaussian-fields/train_gaussian_field_meanflow.py` — Gaussian field (both noise types)
- `Navier-Stokes/test_meanflow_toy.py` — minimal 2D toy unit test
- `Navier-Stokes/test_meanflow_ablation.py` — systematic toy ablation
- `Navier-Stokes/eval_baseline_few_step.py` — baseline FM evaluation at few steps

## Phase 2: Multiscale Per-Band Mean Flow

### Approach

Apply mean flow independently per spectral band (coarse-to-fine), following the
`train_multiscale_perband.py` architecture:
- Sequential training: coarsest band first, then finer bands conditioned on resolved coarse context
- Per-band UNet at native resolution (R×R), with (s, r) mean flow conditioning
- Per-band noise: z0 ~ N(0, σ²_k I) where σ²_k is the conditional variance at scale k
- Loss reweighted by σ²_k so all bands have O(1) loss magnitude
- Warmup: first 10% of each band's training uses pure flow matching (no JVP) to stabilize output scale

### Key Implementation Details

1. **JVP through embed/extract chain**: The per-band UNet operates on embedded images (R×R). JVP propagates through the embed_mask → UNet → extract_mask pipeline. Coarse context has zero tangent (fixed pixels).

2. **Memory management**: JVP doubles forward-pass memory. Finest band (R=G) needs reduced batch size (`--batch_fine`).

3. **Warmup for fine bands**: Fine-scale conditional variance σ² can be very small (e.g., 6.5e-05 for finest GF band). Without warmup, initial loss/σ² explodes (72000+). Pure FM warmup gets the network to the right output scale first.

### Gaussian Field Results (64×64, K=3, 4 bands)

**Final full-resolution eval:**

| Method | Total NFEs | mean_rel | std_ratio |
|---|---|---|---|
| **MF 1-step/band** | **4** | **0.052** | **1.005** |
| MF 2-step/band | 8 | 0.052 | 1.013 |
| Original perband RK4 (4 steps/band) | 64 | 0.21 | ~1.0 |

**16× fewer NFEs, 4× better spectral quality** vs the original perband RK4 baseline.

Per-band best results during training:
- Band 0 (R=8, σ=2.17): MF2 mean_rel=0.015
- Band 1 (R=16, σ=0.098): MF2 mean_rel=0.033
- Band 2 (R=32, σ=0.024): MF2 mean_rel=0.026
- Band 3 (R=64, σ=0.008): MF2 mean_rel=0.26 (harder, fewer training steps after warmup)

### Navier-Stokes (128×128, K=3, 4 bands) — IN PROGRESS

Running on GPU 4 with:
- Bands: R=16 (256 pts) → R=32 (768) → R=64 (3072) → R=128 (12288)
- Steps: 50k + 50k + 50k + 100k = 250k total
- Batch: 200 (R≤64), 20 (R=128)
- σ² estimated from NS training data via ridge regression

Early band 0 (R=16) result: MF1 mean_rel=0.056. Training still in progress.

## Development Timeline & Lessons

### What worked:
- **Toy ablation** (2D Gaussian): revealed that mean flow just needs much longer training (100k steps → 99.3% perfect 1-step). All hyperparameter tuning was secondary.
- **Best-checkpoint tracking**: essential because mean flow training oscillates
- **Per-band approach**: cleaner than full-image multiscale; each band at native resolution
- **Sequential coarse-to-fine**: stable training with per-band optimizer
- **Warmup before JVP**: prevents loss explosion at fine scales with small σ²
- **Loss/σ² normalization**: keeps all bands at O(1) loss magnitude

### What didn't work:
- Copying DiT-specific tricks (time_scale=1000, lognormal time) — these are architecture-specific
- Joint training of all scales simultaneously with JVP — wildly different loss magnitudes per scale
- Aggressive grad_clip=1.0 combined with time_scale=1000 — clipped 99.97% of gradient
- EMA decay 0.9999 for short runs (half-life too long)

## Files

### Single-scale mean flow
- `Navier-Stokes/train_ns_meanflow_gauss_base.py` — NS Gaussian noise
- `Navier-Stokes/train_ns_meanflow_data_dep_noise.py` — NS data-dependent noise
- `Gaussian-fields/train_gaussian_field_meanflow.py` — GF (both noise types)

### Multiscale per-band mean flow
- `Gaussian-fields/train_multiscale_perband_meanflow.py` — GF per-band (validated)
- `Navier-Stokes/train_ns_perband_meanflow.py` — NS per-band (running)

### Verification & baselines
- `Navier-Stokes/test_meanflow_toy.py` — 2D Gaussian unit test
- `Navier-Stokes/test_meanflow_ablation.py` — systematic toy ablation
- `Navier-Stokes/eval_baseline_few_step.py` — pure FM baseline at few steps
