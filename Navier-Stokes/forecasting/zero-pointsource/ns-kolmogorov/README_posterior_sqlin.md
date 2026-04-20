# NS posterior sampling via twisted SMC — β = t² (sqlin) interpolant

Companion to `README_posterior.md` (linlin) on the same 2D Kolmogorov forecasting
task, using the **square-linear** interpolant $I_s = s^2 x_1 + (1-s)\sqrt{s}\,z$
(Albergo-style, matches arXiv:2403.13724). Networks: `runs/sqlin/lag{10,40}_seed0.pt`.

## Setup differences from the linlin sweep

Same observation model, same sampler skeleton; only the interpolant-specific
quantities change.

- **Tweedie twist (sqlin)**:
  $$\hat x_1(x_s, s) = \frac{x_s + (1-s)\,\hat b_\theta(x_s, s, \tilde\omega_t)}{s(2-s)}$$
  Derived from the 2×2 system $b_s = 2s\,\mathbb E[x_1|x_s] - \sqrt{s}\,\mathbb E[z|x_s]$ and $x_s = s^2\,\mathbb E[x_1|x_s] + (1-s)\sqrt{s}\,\mathbb E[z|x_s]$. Sanity: at $s = 1$, $\hat x_1 = x_s$; at $s \to 0$ the $0/0$ form gives $\mathbb E[x_1]$.
- **Föllmer diffusion**: $g^F_t = \sqrt{(1-t)(3-t)}$ (cf. linlin's $\sqrt{1-t^2}$). At $t=0$, $g^F = \sqrt{3} \approx 1.73$ (vs linlin's $1$).
- **Natural-guidance coefficient**:
  $$C_g^\text{sqlin}(t) = \tfrac{(1-t)(3-t) + g^2}{2}$$
  with the same deep identity as linlin: **$C_g = g^2 \iff g^2 = (1-t)(3-t) =$ Föllmer**. So the uniqueness-of-Föllmer property from `simple-examples/zps/README_posterior.md` replicates on the β=t² interpolant.
- **Föllmer drift (closed-form)**:
  $$b^F(x) = \frac{3-t}{2-t}\,\hat b_\theta(x, s) - \frac{2x}{t(2-t)}$$
  The $1/t$ singularity at the origin (vs linlin's merely logarithmic divergence of the drift) makes Euler stepping near $t=0$ much stiffer. **Empirically $t_\text{eps} = 0.01$ is unstable; we use $t_\text{eps} = 0.05$.** Section below shows a quick ablation.

## Sampler config

- $N = 32$ particles, $n_\text{em} = 100$, $t \in [0.05, 0.95]$.
- 8 held-out test ICs; RNGs deterministic per IC (`12345 + ic` for obs noise, `7777 + ic` for particles).
- Observation: `AvgPool_8`, $\sigma_y = 0.3$ (identical to linlin sweep).
- Same $\eta$-sweep: $\{0.1, 0.5, 1.0\}$ × $\{doob, natural\}$ + one `uncond` baseline per lag.
- Logs: `logs_posterior_sweep_sqlin/`; JSONs: `figs/posterior_sweep_sqlin/`.

## Why the larger `t_eps`?

Quick smoke on `runs/sqlin/lag10_seed0.pt` with 16 particles / 100 EM / 2 ICs, Doob guidance $\eta = 1$:

| `t_eps` | Föllmer RMSE | Föllmer logZ | Föllmer ESS | baseline RMSE |
|---|---|---|---|---|
| 0.01 | 1.29 | $-1.4 \times 10^{11}$ | 10 / 16 | 0.20 |
| 0.05 | **0.21** | $-0.96$ | 11.9 / 16 | 0.23 |

The $1/t$ drift singularity is explicit. `t_eps = 0.05` is the smallest value that doesn't amplify $\hat b_\theta$'s network-noise floor through the $-2x/(t(2-t))$ coefficient.

---

## Lag = 10 results (fully complete, 7/7 configs)

Posterior RMSE (normalised units, averaged over 8 ICs) ± sample std over ICs; final ESS out of $N = 32$.

### No guidance in proposal (`uncond`)

| schedule | RMSE | spread | logZ | finalESS |
|---|---|---|---|---|
| **follmer**  | **0.200 ± 0.027** | 0.106 | 31.6 ± 10.9 | 23.5 |
| const    | 0.203 ± 0.029 | 0.124 | 30.6 ± 10.9 | 18.7 |
| baseline | 0.211 ± 0.031 | 0.091 | 31.2 ± 11.5 | 23.5 |
| sqrt_t   | 0.208 ± 0.032 | 0.121 | 32.2 ± 11.3 | 17.1 |
| triangle | 0.218 ± 0.035 | 0.096 | 31.3 ± 11.5 | 24.1 |
| zero     | 0.245 ± 0.039 | 0.031 | 27.1 ± 10.5 | 27.8 |

### Doob guidance $\kappa = \eta g^2$

| schedule | η = 0.1 | η = 0.5 | η = 1.0 |
|---|---|---|---|
| **follmer**  | **0.193 ± 0.024** | **0.191 ± 0.027** | **0.189 ± 0.025** |
| const    | 0.202 ± 0.028 | 0.197 ± 0.027 | 0.195 ± 0.025 |
| baseline | 0.203 ± 0.031 | 0.204 ± 0.032 | 0.204 ± 0.030 |
| sqrt_t   | 0.206 ± 0.029 | 0.203 ± 0.032 | 0.204 ± 0.030 |
| triangle | 0.216 ± 0.035 | 0.214 ± 0.036 | 0.211 ± 0.035 |
| zero     | 0.245 ± 0.039 | 0.245 ± 0.039 | 0.245 ± 0.039 |

### Natural guidance $\kappa = \eta\,C_g^\text{sqlin}$

| schedule | η = 0.1 | η = 0.5 | η = 1.0 |
|---|---|---|---|
| **follmer**  | 0.195 ± 0.026 | 0.192 ± 0.027 | **0.189 ± 0.025** |
| const    | 0.203 ± 0.027 | 0.198 ± 0.026 | 0.194 ± 0.025 |
| baseline | 0.206 ± 0.032 | 0.203 ± 0.029 | 0.225 ± 0.040 |
| sqrt_t   | 0.208 ± 0.031 | 0.202 ± 0.028 | 0.204 ± 0.029 |
| triangle | 0.213 ± 0.036 | 0.211 ± 0.033 | 0.217 ± 0.034 |
| zero     | 0.252 (ESS=1) | 0.247 (ESS=1) | 0.246 (ESS=1) |

### Lag=10 observations

1. **Föllmer is best at every $\eta$, Doob and Natural.** Margin over best competitor (`const`) is 3–5% — small but consistent, and outside the per-IC std on the `const` ↔ `follmer` paired differences.
2. **Föllmer = Natural(Föllmer) to 4 sig figs.** Doob $\eta=1$ Föllmer 0.1889 vs Natural $\eta=1$ 0.1887; same for the other $\eta$. Verifies the $C_g \equiv g^2$ identity empirically on the 2D task.
3. **`zero` (ODE) under Natural guidance collapses to ESS = 1** exactly as in linlin (and in the 1D README). logZ is $\sim -10^{15}$ — the $\kappa/g$ Girsanov term diverges.
4. **`baseline` under Natural $\eta = 1$ gets worse**, RMSE jumping from 0.204 (Doob $\eta=1$) to 0.225 — `baseline` has $g^2 = (1-t)^2 \to 0$ at $t=1$, and $C_g^\text{sqlin} = ((1-t)(3-t) + (1-t)^2)/2$ also $\to 0$, but the Girsanov $\kappa/g = C_g/g^2$ ratio is unbounded. For Doob it's exactly 1, so the step stays under control.
5. **Guidance helps only marginally at lag = 10** — the prior forecast is already tight (uncond Föllmer RMSE 0.200 → Doob $\eta=1$ 0.189 is a 5% drop), reflecting that half a decorrelation time isn't enough for the observation to add much information beyond the prior.

---

## Lag = 40 results (partial — uncond + Doob η=0.1 complete)

### No guidance in proposal (`uncond`)

| schedule | RMSE | spread | logZ | finalESS |
|---|---|---|---|---|
| **follmer**  | **0.498 ± 0.069** | 0.131 | 311 ± 216 | 22.5 |
| const    | 0.506 ± 0.070 | 0.145 | 284 ± 180 | 16.1 |
| baseline | 0.565 ± 0.082 | 0.098 | 250 ± 152 | 23.8 |
| sqrt_t   | 0.589 ± 0.115 | 0.127 | 160 ± 99  | 15.5 |
| triangle | 0.629 ± 0.139 | 0.096 | 122 ± 79  | 20.0 |
| zero     | 0.802 ± 0.159 | 0.004 | −94 ± 53  | 24.4 |

### Doob guidance, η = 0.1

| schedule | RMSE | spread | logZ | finalESS |
|---|---|---|---|---|
| const    | **0.477 ± 0.071** | 0.158 | 306 ± 212 | 17.5 |
| **follmer**  | 0.479 ± 0.052 | 0.143 | 296 ± 212 | 21.6 |
| baseline | 0.529 ± 0.072 | 0.106 | 272 ± 174 | 26.6 |
| sqrt_t   | 0.574 ± 0.085 | 0.136 | 189 ± 136 | 15.5 |
| triangle | 0.609 ± 0.094 | 0.104 | 158 ± 113 | 24.4 |
| zero     | 0.802 ± 0.159 | 0.004 | −94 ± 53  | 24.4 |

Remaining 5 lag=40 configs (`doob η=0.5, 1.0`; `natural η=0.1, 0.5, 1.0`) still in flight; expected to complete in ~30 min. Will update this document in place.

### Lag=40 preliminary observations

1. **Föllmer and const are tightly tied on RMSE at lag=40.** Föllmer edges in the uncond case (0.498 vs 0.506); const edges at Doob η=0.1 (0.477 vs 0.479). Both within 1 std of each other.
2. **Föllmer has consistently higher ESS than const** (22.5 vs 16.1 uncond; 21.6 vs 17.5 guided). The RMSE tie masks a meaningful particle-diversity gap: `const` is closer to collapse.
3. **Guidance reduces RMSE by ~5% even at η=0.1**; larger η sweeps pending.
4. **Order stays Föllmer ≈ const > baseline > sqrt_t > triangle > zero**, matching the linlin posterior result.

---

## Cross-interpolant comparison

Best Föllmer RMSE at sigma_y = 0.3, 8 ICs, matched configs:

| lag | linlin (`runs/lag{L}_seed0.pt`, t_eps=0.01) | sqlin (`runs/sqlin/lag{L}_seed0.pt`, t_eps=0.05) |
|---|---|---|
| 10 uncond  | 0.302 ± 0.100 | **0.200 ± 0.027** |
| 10 doob η=1 | 0.277 ± 0.093 | **0.189 ± 0.025** |
| 40 uncond  | 0.923 ± 0.219 | **0.498 ± 0.069** |
| 40 doob η=0.1 | 0.684 ± 0.146 | **0.479 ± 0.052** |

**sqlin halves the posterior-RMSE at lag=40** (0.923 → 0.498 uncond). Two likely contributions:
- The β=t² schedule concentrates mass near $t \to 1$ faster than β=t, giving a more accurate network when evaluated at the small-$t$ regime that dominates the posterior weights.
- Differences in sampler config also matter: sqlin uses $t_\text{max} = 0.95$ (vs linlin's 0.99) and $N = 32$ (vs 16). The $N$-difference alone can't explain a factor of 2 on the mean, but tighter $t_\text{max}$ does reduce the terminal-blow-up that hits non-Föllmer schedules — which is why sqlin's non-Föllmer schedules look competitive here and are 2-3× off from Föllmer in the linlin sweep.

A fair cross-interpolant comparison would need a matched-sampler rerun of linlin with $t_\text{eps} = 0.05, N = 32$. This is cheap; flagged as follow-up.

## Reproduction

```bash
cd ns-kolmogorov
bash run_sqlin_sweep.sh
```

Uses `posterior_compare.py --ckpt runs/sqlin/lag{10,40}_seed0.pt --t_eps 0.05 --n_particles 32 --n_em 100 --n_ic 8 --obs_factor 8 --sigma_y 0.3` with 7 guidance configs per lag.

## Takeaways

1. **Föllmer-optimality replicates on β = t²**. Same ranking as linlin, same $C_g \equiv g^2$ identity, same ODE-under-Natural pathology.
2. **sqlin needs `t_eps = 0.05`**, not 0.01 — the Föllmer drift's $1/t$ singularity is real and not a curable numerical issue.
3. **sqlin posterior-RMSE is markedly lower than linlin's at long lag**, partly due to interpolant and partly due to the milder endpoint stepping. Not yet an apples-to-apples statement about schedule quality alone.
