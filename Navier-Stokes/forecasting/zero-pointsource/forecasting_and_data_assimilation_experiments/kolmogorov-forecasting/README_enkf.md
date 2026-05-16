# EnKF baselines for the AR data-assimilation experiment

Companion to `README_posterior.md` (twisted-SMC) — same Föllmer prior, same
truth trajectory, same observation operator. Two ensemble-Kalman variants
are compared as classical-DA baselines: a vanilla stochastic EnKF and a
Gaspari–Cohn localised EnKF. Implementation in `enkf.py`; driver in
`ar_animation_frames.py` with `--da_method {enkf, enkf_loc}`.

## Setup (identical to the SMC AR-DA run)

- **Prior**: Föllmer-schedule single-step forecaster trained at lag = 1
  snapshot (`runs/lag1_seed0.pt`).
- **Truth**: held-out trajectory `traj_idx=0`, starting from snapshot 20,
  rolled out 80 AR steps (Δt = 0.05 / step, total = 4 time units ≈ 4
  decorrelation times).
- **Observation operator**: 4× average pool on the truth (128² → 32² =
  1024 obs per step), additive Gaussian noise with $\sigma_y = 0.03$.
  Same factor as the SMC run — matches the conditioning resolution.
- **Ensemble size**: 32. Inflation 1.05 applied to the forecast ensemble
  before assimilation.
- **Method-specific**:
  - **vanilla**: full stochastic EnKF update with the empirical 1024×1024
    forecast-obs covariance — rank-deficient (32 particles ≪ 1024 obs).
  - **localised**: Gaspari–Cohn taper with radius 8 grid pixels (in pixel
    units; periodic distance), Schur-product applied to the forecast
    covariance before computing the Kalman gain.

Launcher:

```bash
bash run_enkf_animation.sh   # runs both variants on GPUs 4 and 5
```

## Results (80 AR steps, RMSE relative units)

| method                        | mean RMSE | max RMSE | final RMSE | mean spread |
|-------------------------------|-----------|----------|------------|-------------|
| no DA (Föllmer prior only)    |  2.556    |  3.990   |  3.990     | n/a         |
| **EnKF (vanilla)**            |  2.158    |  3.982   |  3.773     | 0.069       |
| **EnKF (Gaspari–Cohn, r=8)**  |  **0.321**|  0.359   |  0.354     | 0.285       |
| Twisted SMC (Föllmer, N=32)   |  0.428    |  0.523   |  0.523     | —           |

Numbers reproduced directly from `figs/ar_anim_{prior,enkf,enkf_loc,da}/arrays.npz`
and the trailing summary in `logs_ar_enkf*.log`.

## Takeaways

1. **Vanilla stochastic EnKF diverges** under the same observation
   stream. With 32 particles and 1024 observations the sample
   forecast-obs covariance is rank-deficient, so the Kalman gain picks
   up spurious long-range correlations and collapses the ensemble
   (mean spread → 0.069). The filter ends up nearly indistinguishable
   from the no-DA Föllmer prior (mean RMSE 2.16 vs 2.56).
2. **Localised EnKF wins on RMSE** (mean 0.321 vs 0.428 for SMC, ~25%
   lower). With a well-chosen Gaspari–Cohn radius the spurious
   long-range correlations are zeroed out, the ensemble stays
   calibrated (spread 0.285 ≈ RMSE 0.321), and the filter tracks the
   truth for the full 80-step rollout.
3. **The localised win is conditional on choosing the localisation
   radius.** $r=8$ pixels was hand-tuned to this set-up. Twisted-SMC
   has no comparable hyperparameter; it also returns posterior samples
   rather than just mean + covariance, and extends transparently to
   non-Gaussian / non-linear observation operators (which is the
   setting the paper actually targets). The EnKF rows are reported as
   classical-DA reference points, not as a replacement for the
   generative posterior.

## Files

- `enkf.py` — stochastic EnKF + Gaspari–Cohn taper.
- `ar_animation_frames.py` — AR rollout driver (`--da_method enkf|enkf_loc`).
- `render_ar_animation.py` — 5-row MP4/GIF renderer (truth / prior /
  DA-mean / error / std). Same renderer used for all four columns of the
  table above.
- `run_enkf_animation.sh` — launches both EnKF variants in parallel.
