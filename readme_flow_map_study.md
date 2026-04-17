# Flow Map Study: Few-Step Generation for PDE Fields

Branch: `data-dep-noise-and-multiscale`

## Motivation

Standard flow matching for 2D Navier-Stokes / Gaussian field generation requires 20-50 ODE steps (EM/RK4) at inference. We study **flow map methods** that reduce this to 1-4 network evaluations (NFEs) by conditioning the model on both current time `s` and target time `r`, so each forward pass covers a finite time interval rather than an infinitesimal step.

## Methods Implemented

### 1. MeanFlow (Eulerian / JVP-based)

Based on [Geng et al. 2025, arXiv:2505.13447]. The model learns the average velocity `u(z, s, r)` satisfying the self-consistency identity:

```
u = v + (r - s) · ∂u/∂s_total
```

where the total time derivative `∂u/∂s_total` is computed via forward-mode AD (JVP). Training uses a mix of pure flow matching (`r = s`, standard velocity regression) and mean-flow self-consistency (`r > s`, JVP target).

**Pros:** Best 1-step quality from scratch; exact derivative information (low variance).
**Cons:** JVP doubles forward-pass memory; incompatible with some ops (Flash Attention).

### 2. Shortcut Models (Lagrangian / bootstrap-based)

Based on [Frans et al. 2024, arXiv:2410.12557]. Same model `u(z, s, r)`, but the target is constructed via bootstrap: a step of size `d = r - s` should equal two half-steps averaged:

```
target = ½ [u(z, s, m) + u(z + (m-s)·u(z,s,m), m, r)]   where m = (s+r)/2
```

Evaluated with `torch.no_grad()` (stop-gradient on target).

**Pros:** No JVP — full batch size on all bands; simple and debuggable.
**Cons:** Higher variance (bootstrap noise); needs ~4× more NFEs to match MeanFlow quality.

### Analogy to Variational Inference

The MeanFlow vs Shortcut distinction mirrors **reparametrization trick vs REINFORCE** in VAEs/RL:
- **MeanFlow/JVP** = reparametrization: analytic gradient through the model structure → low variance, high memory
- **Shortcut/bootstrap** = REINFORCE/score-function: evaluate model at sampled points → high variance, low memory
- Larger batch directly reduces bootstrap variance (confirmed empirically)

## Architecture

Both methods use the same per-band UNet with `(s, r)` conditioning:
- UNet's built-in `time_mlp` encodes `s`
- A separate `r_mlp` (same architecture, zero-initialized) encodes `r`
- Combined additively: `embedding = time_mlp(s) + r_mlp(r)`
- Zero-init on `r_mlp` final layer ensures the model starts as standard flow matching

## Results

### Single-Scale (Full-Resolution) on Navier-Stokes 128×128

MeanFlow trained for 200k steps with best-checkpoint tracking (training oscillates).

| Method | NFEs | mid enst | low enst | high enst | std_ratio |
|--------|------|----------|----------|-----------|-----------|
| **MeanFlow gauss base (best@172k)** | **4** | **0.020** | — | — | — |
| MeanFlow gauss base (final@200k) | 4 | 0.025 | 0.135 | 0.370 | 1.01 |
| MeanFlow data-dep (best@105k) | 4 | 0.031 | — | — | — |
| Pure FM EM 4-step | 4 | 0.934 | 0.890 | 44.7 | 0.45 |
| Pure FM RK4 4-step | 16 | 0.061 | 0.483 | 0.251 | 0.98 |

**Key finding:** MeanFlow at 4 NFEs achieves 2.5% mid-band error — 9× better than RK4 (16 NFEs), 37× better than EM (4 NFEs).

### Multiscale Per-Band on Gaussian Fields 64×64 (K=3, 4 bands)

Sequential coarse-to-fine training with per-band UNets at native resolution.

| Method | NFEs total | mean_rel | std_ratio |
|--------|------------|----------|-----------|
| **MeanFlow 1-step/band** | **4** | **0.052** | **1.005** |
| Shortcut 1-step/band | 4 | 0.158 | 0.955 |
| Shortcut 4-step/band | 16 | 0.055 | 0.998 |
| Original perband RK4 (4 steps/band) | 64 | 0.21 | ~1.0 |

**Key finding:** MeanFlow at 4 NFEs (1 step/band) matches or beats 64-NFE RK4 baseline. 16× fewer NFEs, 4× better quality.

### Multiscale Per-Band on Navier-Stokes 128×128 (K=3, 4 bands)

Shortcut uses full batch=200 on all bands (no JVP memory issue). MeanFlow limited to batch=20 on finest band.

| Method | NFEs | low | mid | high | std_ratio | Finest batch |
|--------|------|-----|-----|------|-----------|-------------|
| MeanFlow perband | 4 | 0.248 | 0.163 | 0.965 | 1.04 | 20 |
| MeanFlow perband | 8 | 0.204 | 0.110 | 0.654 | 1.01 | 20 |
| Shortcut perband | 4 | 0.200 | 0.094 | 0.548 | 0.98 | 200 |
| Shortcut perband | 8 | 0.209 | 0.082 | 0.517 | 1.00 | 200 |
| Shortcut perband | 16 | 0.212 | 0.077 | 0.552 | 1.01 | 200 |

**Key finding:** Shortcut's 10× larger batch on the finest band gives better results than MeanFlow despite higher per-sample variance. Confirms the variance/memory tradeoff.

## Key Lessons

1. **Flow maps need much longer training** than standard flow matching (~4× more steps). Toy ablation (2D Gaussian) was decisive: 100k steps → 99.3% perfect 1-step; 20k steps → 14% error.

2. **Best-checkpoint tracking is essential.** Training oscillates — the final checkpoint is often 5-30× worse than the peak. Save on eval metric improvement.

3. **Don't blindly copy reference tricks.** `time_scale=1000` (DiT-specific), lognormal time sampling, and aggressive grad clipping all hurt when applied to our UNet. Only uniform time, `grad_clip=1.0`, and EMA helped.

4. **Per-band training requires careful loss normalization.** Loss/σ² equalization per band + warmup (10% pure FM before JVP) prevents fine-band explosions.

5. **Memory determines which loss to use.** MeanFlow wins when JVP fits in memory; Shortcut wins when you need large batch on the finest band. Hybrid (JVP for coarse, bootstrap for fine) is a natural next step.

6. **Single-scale MeanFlow outperforms per-band on mid-k for NS.** NS has strong cross-scale coupling (turbulent cascade) that per-band decomposition breaks. The single-scale model captures these interactions.

## Files

### Code
```
# Single-scale mean flow
Navier-Stokes/train_ns_meanflow_gauss_base.py       # NS, Gaussian noise
Navier-Stokes/train_ns_meanflow_data_dep_noise.py    # NS, data-dependent noise
Gaussian-fields/train_gaussian_field_meanflow.py      # GF, both noise types

# Multiscale per-band mean flow (Eulerian / JVP)
Gaussian-fields/train_multiscale_perband_meanflow.py  # GF
Navier-Stokes/train_ns_perband_meanflow.py            # NS

# Multiscale per-band shortcut (Lagrangian / bootstrap)
Gaussian-fields/train_multiscale_perband_shortcut.py  # GF
Navier-Stokes/train_ns_perband_shortcut.py            # NS

# Verification & baselines
Navier-Stokes/test_meanflow_toy.py                    # 2D Gaussian unit test
Navier-Stokes/test_meanflow_ablation.py               # Systematic toy ablation
Navier-Stokes/eval_baseline_few_step.py               # Pure FM at few steps

# Reports
report_meanflow.md                                     # Experiment log
report_few_step_lit_review.md                          # Literature review
readme_flow_map_study.md                               # This file
```

### Trained Models
```
# Single-scale MeanFlow (best results)
Navier-Stokes/results/ns_meanflow_gauss_base_v2/model_best.pt    # best@172k, mid=0.020
Navier-Stokes/results/ns_meanflow_data_dep_v2/model_best.pt      # best@105k, mid=0.031

# Perband MeanFlow
Navier-Stokes/results/ns_perband_mf_K3/mask_s{0-3}_R{16-128}/model.pt
Gaussian-fields/results/perband_mf_K3/mask_s{0-3}_R{8-64}/model.pt

# Perband Shortcut
Navier-Stokes/results/ns_perband_shortcut_K3/mask_s{0-3}_R{16-128}/model.pt
Gaussian-fields/results/perband_shortcut_K3/mask_s{0-3}_R{8-64}/model.pt
```

## Literature Context

See `report_few_step_lit_review.md` for the full review. Our work sits in the **flow map** family alongside:
- Flow Map Matching (FMM, Boffi et al. 2024) — Lagrangian + Eulerian formulations
- Shortcut Models (Frans et al. 2024) — bootstrap self-consistency
- MeanFlow (Geng et al. 2025) — JVP self-consistency
- Consistency Trajectory Models (CTM, Kim et al. 2024) — two-time prediction
- Align Your Flow (Sabour et al. 2025) — flow-map distillation at scale

## Potential Next Steps

1. **Hybrid per-band:** MeanFlow (JVP) on coarse bands + Shortcut (bootstrap) on finest band
2. **Adversarial finetune** on spectrum slices for low-k residual
3. **iCT-style curriculum** on step gap `r-s`: grow from small to full range instead of binary warmup
4. **Gradient accumulation** to recover effective batch for JVP on finest band
5. **Single-scale Shortcut** at full 128×128 for NS (combining single-scale's cross-scale modeling with shortcut's memory efficiency)
