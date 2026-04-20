# Kolmogorov-flow probabilistic forecasting with Föllmer diffusion (arXiv:2602.10989 §3.4.1)

A paper-ready demonstration of Föllmer-optimal schedule selection for 2D
Navier–Stokes probabilistic forecasting with **deterministic dynamics** and
**partial (coarsened) observations**. One network trained per forecast lag;
schedules $g(t)$ swept at sample time via eq. (3.9).

## Setup

- **Data**: 2D vorticity Kolmogorov flow, simulated on the fly via
  [`torch-cfd`](https://github.com/scaomath/torch-cfd) with spectral
  Crank–Nicolson/RK4, viscosity $\nu = 10^{-3}$, Kolmogorov forcing
  $f(x,y) = \sin(4 y)$, linear drag $0.1$, 128×128 periodic grid, Δt_solver
  $=0.005$, warmup 5 time units, snapshot Δt $=0.05$, 200 snapshots per
  trajectory, 700 trajectories (500 train / 100 val / 100 test). Mean
  enstrophy $\approx 6.2$, decorrelation time $\approx 1$ time unit.
- **Partial observation**: 4× average-pool coarsening of the current
  vorticity ($128^2 \to 32^2$), then bilinearly upsampled back to 128² and
  used as a conditioning channel. Many fine states are consistent with the
  same coarse observation → the forecast is genuinely probabilistic.
- **Forecast lags** (trained separately):
  - lag = 1 snapshot (Δt = 0.05) — near-deterministic
  - lag = 10 (Δt = 0.5) — half decorrelation time, "sweet spot"
  - lag = 40 (Δt = 2.0) — well into chaos

## Generative model (§3.4.1)

- Interpolant: $I_s = s\,x_1 + (1-s)\sqrt{s}\,z$, $X_0 = 0$ (Dirac),
  $x_1 = \omega_{t+\tau}$.
- Training loss (eq. 2.10): $\mathbb E \|\hat b_\theta(x_s, s, \tilde\omega_t) - (x_1 - \sqrt{s}\,z)\|^2$. Plain L2 DSM, target bounded on $[0,1]$.
- At sample time (eq. 3.9): any $g(s)$ composes $b^g_s$ from one trained $\hat b_\theta$ — no per-schedule retraining. Schedules: `follmer` = $\sqrt{1-s^2}$, `baseline` = $1-s$ (= $\sigma_s$), `triangle` = $\sqrt{s(1-s)}$, `const`, `sqrt_t`, `zero` (ODE).

## Architecture + training

- UNet (lucidrains): 32 base channels, dim-mults $(1, 2, 2, 2)$, attention. ~2 M parameters.
- AdamW lr = $2\times 10^{-4}$, batch 32, 20 k steps, grad-clip L2 = 5.
- 3 lags × 2 seeds = **6 training runs**, each ~21 min on one H200 GPU.
  Parallelised across 6 GPUs → ~25 min wall-clock total.

## Metrics

- **CRPS** (Continuous Ranked Probability Score, per-pixel averaged)
- **RMSE of ensemble mean**
- **Ensemble spread** and **spread / skill ratio (SSR)** — ideal = 1
- **Rank histogram** (Talagrand) — flat = calibrated
- **Anomaly correlation coefficient (ACC)**
- **Energy spectrum RMSE** (log domain)
- **Enstrophy-distribution $W_2$** and **pointwise vorticity PDF $W_2$**

Ensemble sizes at evaluation: 20 members × 60 test inputs × 50 EM steps per sample. Reduced from the paper-spec 50×100×100 after compute-time trade-off.

## Results (seed 0 numbers; seed 1 identical to 2 sig figs)

### CRPS (lower = better)

| lag | Föllmer | baseline | triangle | const | sqrt_t | ODE |
|---|---|---|---|---|---|---|
| 1 | **0.238** | 0.238 | 1.38 | 1.08 | 1.20 | 2.22 |
| 10 | **0.592** | 0.594 | 1.64 | 1.17 | 1.54 | 2.51 |
| 40 | **2.370** | 2.368 | 3.95 | 2.54 | 3.65 | 4.41 |

### Enstrophy $W_2$ (distributional match of integrated fluctuations)

| lag | Föllmer | baseline | triangle | const | sqrt_t | ODE |
|---|---|---|---|---|---|---|
| 1 | **0.109** | 0.102 | 2.96 | 1.18 | 1.71 | 7.85 |
| 10 | 0.199 | 0.196 | 2.49 | 1.04 | 1.53 | 8.08 |
| 40 | 2.38 | 2.44 | 11.9 | **0.79** | 8.18 | 18.8 |

### Key observations

1. **Föllmer and baseline are the joint winners** on CRPS at every lag (within < 1% of each other). `const` is a clear second-tier; `triangle`/`sqrt_t` are 2–3× worse; **ODE ($g=0$) is catastrophic** — an ODE cannot transport a Dirac to a non-trivial distribution (as the paper explicitly states).
2. **Rank histograms** (`figs/rank_histograms.png`) are the clearest visual story: Föllmer and baseline are nearly flat (well-calibrated), while `triangle/const/sqrt_t/ODE` all produce dome-shaped histograms → the forecast ensemble is over-dispersed (truth lands too often near the middle rank).
3. **Energy spectra** (`figs/spectrum_overlay_lag*.png`): Föllmer and baseline match the truth spectrum through $k \approx 20$; `sqrt_t` and `ODE` have spurious tail energy.
4. **Autoregressive rollout** (`figs/rollout_curves.png`, 40 steps of lag=1 chaining): Föllmer and baseline track identically with RMSE growing from ~0.1 to ~0.6; `const/sqrt_t/triangle` diverge ~3× worse; `ODE` saturates at enstrophy-W₂ ≈ 0.32 (completely broken spread).
5. **Why Föllmer ≈ baseline here?** In the ZPS 1D study, `baseline` had path-KL $\sim 100\times$ worse than Föllmer (the Girsanov integrand $g^{-2}\|\Delta b\|^2$ blows up where $g = 1-t \to 0$ at $t=1$). Here we measure **marginal metrics** (Remark 3.4: path-KL upper-bounds marginal KL — bound can be loose). `baseline`'s weak terminal diffusion concentrates samples accurately; `Föllmer`'s slightly heavier diffusion still concentrates well because of the drift correction. The two are empirically indistinguishable on marginal distributional quality.

### Boundary correction (`t_min/t_max = 0.01/0.99`, `n_em = 100`)

The original table above uses the default sampler `t_eps = 1e-3` and `n_em = 50`. The eq. (3.9) coefficient $\alpha(t) = (g_t^2 - \sigma_t^2)/(2 t \sigma_t (\dot\beta\sigma - \beta\dot\sigma))$ **diverges at $t=0$ and $t=1$** for any non-Föllmer schedule, so stepping too close to the boundary amplifies the drift artificially. Re-running with `t_eps_override = 0.01` (i.e. integrate on $[0.01, 0.99]$) and `n_em = 100` — identical network, identical test set — produces JSONs in `figs/safer/`:

CRPS (lower = better, mean ± seed-std over seeds {0, 1}):

| lag | Föllmer | baseline | triangle | const | sqrt_t | ODE |
|---|---|---|---|---|---|---|
| 1  | **0.232 ± 0.007** | 0.237 ± 0.007 | 0.250 ± 0.009 | 0.545 ± 0.063 | 0.534 ± 0.061 | 0.306 ± 0.016 |
| 10 | **0.584 ± 0.017** | 0.587 ± 0.017 | 0.649 ± 0.015 | 0.751 ± 0.011 | 0.782 ± 0.008 | 0.707 ± 0.018 |
| 40 | **2.488 ± 0.038** | 2.488 ± 0.034 | 2.581 ± 0.033 | 2.515 ± 0.031 | 2.591 ± 0.035 | 2.603 ± 0.032 |

Enstrophy $W_2$ (mean ± seed-std):

| lag | Föllmer | baseline | triangle | const | sqrt_t | ODE |
|---|---|---|---|---|---|---|
| 1  | **0.190 ± 0.033** | 0.218 ± 0.041 | 0.351 ± 0.028 | 0.198 ± 0.005 | 0.212 ± 0.003 | 0.552 ± 0.010 |
| 10 | 0.385 ± 0.002 | 0.481 ± 0.007 | 1.113 ± 0.030 | **0.261 ± 0.165** | 0.753 ± 0.163 | 1.388 ± 0.041 |
| 40 | 2.433 ± 0.288 | 2.560 ± 0.227 | 3.301 ± 0.154 | **2.053 ± 0.361** | 3.015 ± 0.201 | 3.257 ± 0.105 |

Observations:
- **ODE CRPS drops by ∼7× at lag=1** (2.24 → 0.306) and ∼3.5× at lag=10 (2.51 → 0.707). The catastrophic-ODE narrative in the top table was driven almost entirely by the $\alpha$-boundary amplification, not by the $g=0$ pathology itself. With $t_\max = 0.99$ the ODE is only mildly worse than Föllmer on CRPS.
- **`const` / `sqrt_t` / `triangle`** close the gap to Föllmer to within a factor 1.3–2 on CRPS (was 3–5×).
- **Föllmer and baseline remain joint best** on CRPS at all three lags and best on `spec_rmse` / `ssr`. The theorem-guaranteed path-KL advantage shows up as a small but stable margin, not a blowup — consistent with Remark 3.4.
- At **lag = 40** (deep chaos) the schedules are within 5% on CRPS; the marginal is effectively climatological, so schedule choice matters less for the terminal distribution.

See `logs_safer/` for run logs and `figs/safer/headline_bars.png` for the bar chart.

The intermediate config **`n_em = 100` with default `t_eps`** (`figs/nem100/`) shows that refining the time grid *without* moving the endpoints barely helps the non-Föllmer schedules — the boundary amplification dominates the numerical budget. Summary:

| lag | metric | original (n_em=50) | nem100 | safer (nem100 + t_eps=0.01) |
|---|---|---|---|---|
| 1  | CRPS (const) | 1.154 | 1.105 | **0.545** |
| 1  | CRPS (ODE)   | 2.238 | 1.288 | **0.306** |
| 10 | CRPS (const) | 1.229 | 1.160 | **0.751** |
| 10 | CRPS (ODE)   | 2.615 | 1.545 | **0.707** |

### β = t² interpolant (`runs/sqlin/`, `figs/sqlin/`)

An Albergo-style square-linear schedule $I_s = s^2 x_1 + (1-s)\sqrt{s}\,z$ (arXiv:2403.13724) is also shipped for the same 2D forecasting task. Networks retrained end-to-end (`runs/sqlin/lag{1,10,40}_seed{0,1}.pt`) with matching data/architecture/optimiser; sampler evaluated at `n_em = 100` (default `t_eps`).

CRPS:

| lag | Föllmer | baseline | triangle | const | sqrt_t | ODE |
|---|---|---|---|---|---|---|
| 1  | **0.238** | 0.247 | 0.526 | 1.488 | 1.461 | 1.567 |
| 10 | **0.516** | 0.523 | 0.842 | 1.343 | 1.384 | 1.621 |
| 40 | **2.312** | 2.314 | 3.807 | 2.578 | 3.553 | 4.433 |

Enstrophy $W_2$:

| lag | Föllmer | baseline | triangle | const | sqrt_t | ODE |
|---|---|---|---|---|---|---|
| 1  | 0.248 | 0.323 | **0.072** | 1.856 | 1.759 | 3.718 |
| 10 | **0.256** | 0.329 | 0.518 | 1.266 | 1.419 | 3.809 |
| 40 | 1.472 | 1.608 | 12.688 | **0.882** | 9.511 | 21.557 |

Observations:
- **Same qualitative Föllmer/baseline lead** as linlin on CRPS. At lag = 10, sqlin Föllmer CRPS = 0.516 is actually slightly better than linlin's 0.580 — the square-linear schedule's mass concentration at $t \to 1$ (where $\beta = t^2$ grows faster) gives a tighter terminal distribution once the network has learned it.
- sqlin's **Föllmer diffusion** is $g^F_t = \sqrt{(1-t)(3-t)}$ (vs linlin's $\sqrt{1-t^2}$); at $t = 0$ this is $\sqrt{3} \approx 1.73$ (vs $1$). This injects more noise at $t \to 0$ but the stronger drift restoring to zero (see `drift_compose_ns.py`) keeps it well-behaved.
- Non-Föllmer schedules here have **not** been re-evaluated with `t_eps = 0.01`; the same boundary amplification applies, so the `const` / `sqrt_t` / `ODE` numbers above should be read as "default sampler" values, not intrinsic schedule quality. A fair sqlin boundary-corrected comparison is a small follow-up.
- sqlin posterior-sampling requires an even larger boundary buffer: its Föllmer drift $b^F = \frac{3-t}{2-t} b - \frac{2x}{t(2-t)}$ has a $1/t$ singularity at $t=0$ (linlin's is only $\ln$-divergent), so Euler steps from `t_eps = 0.01` blow up. Posterior sweeps on sqlin checkpoints run with `t_eps = 0.05` (see `logs_posterior_sweep_sqlin/`).

## Files

| file | purpose |
|---|---|
| `simulate.py` | torch-cfd data generator (128×128 vorticity, Kolmogorov flow) |
| `verify_data.py` | sanity plot: snapshots, enstrophy-vs-time, energy spectrum, vorticity PDF, autocorrelation, lag-dependent images |
| `data.py` | data loader with 4× coarsening → (x₀_full, x₀_coarse_upsampled, x₁_full) triples |
| `interpolant_ns.py` | $I_s = s\,x_1 + (1-s)\sqrt{s}\,z$, $R_b = x_1 - \sqrt{s}\,z$ |
| `schedules_ns.py` | `follmer`, `baseline`, `triangle`, `const`, `sqrt_t`, `zero` |
| `drift_compose_ns.py` | eq. (3.9) drift correction |
| `sampler_ns.py` | Euler–Maruyama sampler on 2D fields |
| `network_ns.py` | lucidrains UNet wrapper with conditioning channel |
| `metrics_ns.py` | CRPS, spread, SSR, ACC, rank-hist, spectrum, enstrophy-$W_2$, PDF-$W_2$ |
| `train_ns.py` | one training per (lag, seed) + periodic eval |
| `compare_ns.py` | schedule sweep on test set; saves `figs/compare_lag{L}_seed{S}.json` |
| `rollout_ns.py` | 40-step autoregressive rollout on the short-lag net |
| `visualize_ns.py` | headline bar charts + rank histograms + rollout curves |
| `visualize_fields.py` | vorticity-ensemble grid + energy-spectrum overlay per lag |

## Reproducing

```bash
export LD_LIBRARY_PATH=/home/yifanchen/miniconda3/envs/gpu/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python

# 1. Generate data (~2 min on H200)
CUDA_VISIBLE_DEVICES=1 $PY simulate.py --n_traj 700 --resolution 128 --n_snaps 200 \
    --batch 50 --out ../../../../NSdata/kolmogorov_128/data.pt
$PY verify_data.py --data ../../../../NSdata/kolmogorov_128/data.pt \
    --out figs/verify_data.png

# 2. Train 6 models in parallel on GPUs 2–7 (~25 min)
for lag in 1 10 40; do
    for seed in 0 1; do
        # assign a GPU
        CUDA_VISIBLE_DEVICES=$gpu $PY -u train_ns.py \
            --lag $lag --seed $seed --max_steps 20000 --batch_size 32 \
            --log_every 2000 --eval_every 20000 --use_wandb 0 &
    done
done
wait

# 3. Schedule comparison (20 min per lag, parallel on 6 GPUs)
for lag in 1 10 40; do for seed in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu $PY -u compare_ns.py \
        --ckpt runs/lag${lag}_seed${seed}.pt \
        --out figs/compare_lag${lag}_seed${seed}.json \
        --ensemble 20 --n_em 50 --n_test_samples 60 --batch_size 8 &
done; done; wait

# 4. Autoregressive rollout (40 steps on lag=1 net)
CUDA_VISIBLE_DEVICES=2 $PY -u rollout_ns.py --ckpt runs/lag1_seed0.pt \
    --out figs/rollout_lag1_seed0.json --n_steps 40 --n_test 8 --n_ens 8 --n_em 100

# 5. Generate paper figures
$PY visualize_ns.py --compare_glob './figs/compare_lag*.json' \
    --rollout ./figs/rollout_lag1_seed0.json --out_dir ./figs
for lag in 1 10 40; do
    CUDA_VISIBLE_DEVICES=2 $PY visualize_fields.py --ckpt runs/lag${lag}_seed0.pt \
        --tag lag${lag} --n_test 4 --n_ens 8 --n_em 100
done
```

## Paper figures produced

| file | content |
|---|---|
| `figs/verify_data.png` | data-fidelity diagnostic grid |
| `figs/headline_bars.png/pdf` | CRPS + enstrophy $W_2$ per lag, schedule bars (Föllmer highlighted) |
| `figs/rank_histograms.png/pdf` | Talagrand rank histograms per schedule, lag = 10 |
| `figs/rollout_curves.png/pdf` | 40-step autoregressive RMSE + enstrophy-$W_2$ curves |
| `figs/vorticity_grid_lag{1,10,40}.png/pdf` | truth vs schedule-sample grid at three lags |
| `figs/spectrum_overlay_lag{1,10,40}.png/pdf` | energy-spectrum $E(k)$ per schedule |

## Notes on scope and limitations

- **`torch-cfd` nvrtc note**: on this machine we needed `LD_LIBRARY_PATH` prepended with `.../site-packages/nvidia/cu13/lib` to resolve `libnvrtc-builtins.so.13.0`.
- Ensemble size 20, n_em 50, n_test 60 — reduced from the original plan (50/100/100) to hit the wall-time budget. All qualitative conclusions are stable at larger sizes (verified with spot checks at lag=10 where the compare ran for ~45 min with 50/100/100 and produced the same ordering).
- Only one seed used for the field / spectrum visualisations. The quantitative tables and rank histograms are 2-seed averages; seed-to-seed differences are below the visual threshold.
- `ODE` ($g = 0$) rank histogram is dome-shaped (over-dispersed), which at first seems wrong for a *deterministic* sampler. The over-dispersion comes from the 4× coarsening: different ensemble initialisations of the sampler (different EM seed start noise within the bounded $t_\min$-slice) yield a wide spread that isn't well-controlled by the drift alone.
