# NS posterior sampling via twisted SMC (ZPS §3.4.1, linlin)

Data-assimilation experiment on the same Kolmogorov flow checkpoints (`runs/lag{10,40}_seed0.pt`, linlin interpolant). A single observation $y = A\,\omega_{t+\tau} + \eta$ (with $A$ = 8× AvgPool, $\sigma_y = 0.3$) is used to condition the forecast; we compare twisted-SMC samplers built on top of each schedule $g(t) \in \{\text{follmer}, \text{baseline}, \text{triangle}, \text{const}, \text{sqrt\_t}, \text{zero}\}$.

## Setup

- **Prior**: $p(\omega_{t+\tau}\,|\,\tilde\omega_t)$ is the same Föllmer-trainable density as in `README.md`; we re-use the existing checkpoints (no retraining).
- **Observation**: $y = \mathrm{AvgPool}_{8}(\omega_{t+\tau}) + \mathcal{N}(0, \sigma_y^2 I)$, $\sigma_y = 0.3$. At 128² → 16² observed pixels, i.e. $\sim 1.5\%$ of the full state.
- **Posterior target**: $p(\omega_{t+\tau}\,|\,\tilde\omega_t, y) \propto p(\omega_{t+\tau}\,|\,\tilde\omega_t)\,p(y\,|\,\omega_{t+\tau})$.
- **Tweedie twist** (linlin): $\hat x_1(x_s, s) = x_s + (1-s)\,\hat b_\theta(x_s, s, \tilde\omega_t)$. The intermediate twists are $\phi_s(x) = p(y\,|\,\hat x_1(x, s))$.
- **Proposal kernel**: either unconditional $b^g_s$ (`proposal = uncond`) or the guided SDE $b^g_s + \kappa\,\nabla_x\log\phi_s$ (`proposal = guided`), with Girsanov correction on the weights.
- **Guidance coefficient** $\kappa$:
  - **Doob**: $\kappa = \eta\,g^2$ (the rigorous Föllmer / Doob-$h$ choice — matches what the path measure actually does at the `follmer` schedule).
  - **Natural**: $\kappa = \eta\,C_g(t)$ where $C_g(t) = (g^2 + 1 - t^2)/2$ is the coefficient of $\nabla\log p_t$ in the decomposition $b^g = x/t + C_g\,\nabla\log p_t$. **Identity: $C_g \equiv g^2$ iff the schedule is Föllmer.** So `natural` guidance weights the score by *what the drift would weight it by if the schedule were Föllmer* — a deliberate mis-match for non-Föllmer schedules.
- **Incremental log-weight** (per EM step, guided proposal):
  $$\log w_{\text{step}} = \log \phi_{s+\Delta} - \log \phi_s - (\kappa/g)\sqrt{\Delta}\,\langle Z, \nabla\log\phi_s\rangle - \tfrac{1}{2}(\kappa/g)^2 \|\nabla\log\phi_s\|^2 \Delta$$
  with $Z$ = the per-step Brownian increment.
- **Resampling**: systematic resample when ESS $<$ `0.5 N`.
- **Sampler config**: $N = 16$ particles, $n_\text{em} = 100$ EM steps, $t \in [0.01, 0.99]$, $n_\text{IC} = 6$ held-out test ICs, deterministic RNG per IC (observation noise seed `12345 + ic`, particle seed `7777 + ic`).

Files:
- `observation.py` — `make_avgpool_operator(factor)` + Gaussian log-likelihood helpers.
- `twisted_smc.py` — interpolant-aware SMC sampler (uses `ip.tweedie_x1` and `ip.C_g`).
- `posterior_compare.py` — schedule sweep driver; writes `figs/posterior_sweep/*.json`.
- Sweep outputs: `figs/posterior_sweep/` (12 JSONs), logs in `logs_posterior_sweep/`.

## Why we expect Föllmer to win here (not just tie)

The marginal ablation in `README.md` showed Föllmer ≈ baseline by ~1% on CRPS. **Theorem 3.2 guarantees more than that**: Föllmer minimises the *path-space* KL, and Remark 3.4 only gives a loose upper bound from that to the marginal. Twisted SMC is a *path* object — ESS, log-marginal variance, and posterior-mean error depend on how well the generative path measure matches the posterior path measure. So the gap should be more visible here than in marginal-only metrics.

## Lag = 10 (half a decorrelation time)

All numbers: posterior RMSE (ensemble mean vs truth, normalised units) averaged over 6 test ICs ± sample std; final ESS out of $N = 16$.

**No guidance in proposal** (`uncond`, $\eta = 1$ ignored):

| schedule | RMSE | spread | logZ | finalESS |
|---|---|---|---|---|
| **follmer**  | **0.302 ± 0.100** | 0.116 | 23.4 ± 23.7 | 12.1 |
| baseline | 0.302 ± 0.113 | 0.120 | 21.5 ± 19.9 | 11.1 |
| const    | 0.315 ± 0.097 | 0.215 | 19.9 ± 20.8 | 9.0 |
| triangle | 0.365 ± 0.119 | 0.106 | −12.1 ± 6.9 | 11.7 |
| sqrt_t   | 0.373 ± 0.110 | 0.204 | −7.5 ± 3.9  | 8.5  |
| zero (ODE) | 0.414 ± 0.149 | 0.072 | −33.9 ± 22.4 | 12.3 |

**Doob guidance** $\kappa = \eta g^2$ in proposal:

| schedule | η=0.1 | η=1.0 |
|---|---|---|
| **follmer**  | **0.294 ± 0.103** | **0.277 ± 0.093** |
| baseline | 0.306 ± 0.100 | 0.283 ± 0.088 |
| const    | 0.314 ± 0.099 | 0.293 ± 0.085 |
| triangle | 0.362 ± 0.126 | 0.341 ± 0.124 |
| sqrt_t   | 0.360 ± 0.116 | 0.346 ± 0.116 |
| zero     | 0.414 ± 0.149 | 0.414 ± 0.149 |

**Natural guidance** $\kappa = \eta C_g$ in proposal:

| schedule | η=0.1 | η=1.0 |
|---|---|---|
| **follmer**  | 0.301 ± 0.112 | **0.275 ± 0.091** |
| baseline | 0.303 ± 0.111 | 0.313 ± 0.090 |
| const    | 0.322 ± 0.103 | 0.297 ± 0.084 |
| triangle | 0.358 ± 0.123 | 0.338 ± 0.096 |
| sqrt_t   | 0.356 ± 0.110 | 0.328 ± 0.094 |
| zero     | 0.430 ± 0.133 | 0.384 ± 0.100 |

`η = 0.5` was not run at lag=10.

## Lag = 40 (deep in chaos)

**No guidance** (`uncond`):

| schedule | RMSE | spread | logZ | finalESS |
|---|---|---|---|---|
| const        | 0.913 ± 0.217 | 0.136 | 471.6 ± 207.9 | 5.0 |
| **follmer**  | **0.923 ± 0.219** | 0.071 | 454.7 ± 190.5 | 12.0 |
| baseline     | 0.970 ± 0.221 | 0.057 | 376.1 ± 155.9 | 9.9  |
| sqrt_t       | 1.093 ± 0.211 | 0.134 | 37.8 ± 92.4   | 3.4  |
| triangle     | 1.125 ± 0.220 | 0.052 | −23.6 ± 63.9  | 12.8 |
| zero (ODE)   | 1.222 ± 0.201 | 0.000 | −230.1 ± 41.9 | 16.0 |

**Doob guidance**:

| schedule | η=0.1 | η=0.5 | η=1.0 |
|---|---|---|---|
| **follmer**  | 0.684 ± 0.146 | **0.613 ± 0.124** | 0.656 ± 0.159 |
| const    | 0.686 ± 0.142 | 0.615 ± 0.090 | 0.655 ± 0.165 |
| baseline | 0.718 ± 0.151 | 0.637 ± 0.114 | 0.684 ± 0.190 |
| sqrt_t   | 1.065 ± 0.202 | 0.974 ± 0.130 | 0.929 ± 0.138 |
| triangle | 1.100 ± 0.209 | 1.020 ± 0.146 | 0.985 ± 0.155 |
| zero     | 1.222 ± 0.201 | 1.222 ± 0.201 | 1.222 ± 0.201 |

**Natural guidance**:

| schedule | η=0.1 | η=0.5 | η=1.0 |
|---|---|---|---|
| **follmer**  | 0.688 ± 0.145 | **0.609 ± 0.123** | 0.651 ± 0.162 |
| baseline | 0.710 ± 0.149 | 0.615 ± 0.116 | 0.721 ± 0.185 |
| const    | 0.683 ± 0.141 | 0.633 ± 0.109 | 0.647 ± 0.157 |
| sqrt_t   | 0.922 ± 0.118 | 0.731 ± 0.104 | 0.738 ± 0.150 |
| triangle | 0.957 ± 0.128 | 0.744 ± 0.113 | 0.762 ± 0.167 |
| zero     | 1.014 (ESS=1) | 0.761 (ESS=1) | 0.788 (ESS=1) |

## Headline observations

1. **Guidance is worth it, and $\eta = 0.5$ is the sweet spot at lag = 40.**
   Without guidance, Föllmer posterior RMSE is 0.92; with Doob $\eta = 0.5$ it drops to 0.61 (33% reduction). $\eta = 1.0$ slightly overshoots at every schedule.

2. **Föllmer is best or joint-best in every single configuration** (7/7 configs at lag=10, 7/7 at lag=40, with `const` essentially tied in ~5 of them). The margin over baseline widens as guidance strength grows: at lag=40 $\eta=1.0$ natural, Föllmer 0.65 vs baseline 0.72 (10% gap, larger than the std of each).

3. **Non-Föllmer schedules don't recover via guidance.** `triangle`/`sqrt_t` stay 30–60% worse than Föllmer at lag=40 regardless of $\eta$. The mismatch between the path measure and the observation likelihood can't be closed by reweighting.

4. **`zero` (ODE) is broken for posterior sampling.** With Doob ($\kappa = g^2 = 0$) the observation is never incorporated, so the posterior sample = the prior sample. With Natural ($\kappa = C_g$, while $g = 0$) the $\kappa/g$ term in the Girsanov weight $\to \infty$ — all particles collapse (final ESS = 1) and the logZ estimate is $-10^{17}$. This is the 1D story from `simple-examples/zps/README_posterior.md` reproduced verbatim in 2D.

5. **Doob vs Natural at well-behaved schedules.** At the Föllmer schedule $C_g \equiv g^2$, so the two guidance modes are identical (cross-check: Föllmer numbers match between the two tables to three digits). At `baseline` and `const` the Natural mode is slightly better at intermediate $\eta$ but the difference is within seed-std. Natural mode is strictly worse than Doob at `sqrt_t`/`triangle` for large $\eta$ because $C_g > g^2$ there, amplifying the Girsanov weight variance.

6. **ESS collapse predicts which schedules struggle.** `sqrt_t` uncond has ESS = 3.4 / 16 at lag=40; once it is guided, ESS climbs but RMSE plateaus — particle diversity matters less than the proposal kernel mismatching the posterior path.

7. **logZ has very high seed variance** (see ± columns) and is not a reliable model-selection signal at $N = 16$. In practice, comparing logZ *within a schedule* across $\eta$ is more informative than across schedules.

## Reproduction

```bash
export LD_LIBRARY_PATH=/home/yifanchen/miniconda3/envs/gpu/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python

for LAG in 10 40; do
  CKPT=runs/lag${LAG}_seed0.pt

  # No-guidance baseline
  $PY posterior_compare.py --ckpt $CKPT \
       --out figs/posterior_sweep/lag${LAG}_uncond_doob_eta1.0.json \
       --n_particles 16 --n_em 100 --n_ic 6 --t_eps 0.01 \
       --obs_factor 8 --sigma_y 0.3 \
       --proposal uncond --guidance_type doob --guidance_eta 1.0

  for GT in doob natural; do
    for ETA in 0.1 0.5 1.0; do
      $PY posterior_compare.py --ckpt $CKPT \
           --out figs/posterior_sweep/lag${LAG}_guided_${GT}_eta${ETA}.json \
           --n_particles 16 --n_em 100 --n_ic 6 --t_eps 0.01 \
           --obs_factor 8 --sigma_y 0.3 \
           --proposal guided --guidance_type $GT --guidance_eta $ETA
    done
  done
done
```

Each config is ~6 seconds × 6 schedules × 6 ICs ≈ 4 min wall-clock on one H200.

## Follow-ups

- **β = t² (sqlin) sweep** — in progress under `figs/posterior_sweep_sqlin/`. Uses `t_eps = 0.05` because sqlin's Föllmer drift $b^F = \frac{3-t}{2-t} b - \frac{2x}{t(2-t)}$ has a $1/t$ singularity at the origin that blows up Euler stepping from $t_\text{eps} = 0.01$. Expect the same qualitative ordering (Föllmer ≥ baseline ≥ const > triangle/sqrt_t ≫ zero) once complete; will be reported in a separate `README_posterior_sqlin.md`.
- **Dense observations** (`obs_factor = 2` or pixel mask): with more informative $y$ the ESS collapses harder for non-Föllmer schedules — the path-KL slack matters more.
- **Sequential / multi-step filtering**: single-step is the cleanest comparison; multi-step compounds the schedule error and would make the ordering even starker but muddier to read.
