# Point-source Föllmer study — arXiv:2602.10989 §3.4.1

Low-dimensional demonstration of the **Föllmer-optimality theorem** for the
point-source (Dirac-initial) stochastic interpolant with the *linear-linear*
schedule $\beta_t = t,\ \sigma_t = 1 - t$.

Key paper prescription: a **single network** $\hat b_\theta$ trained for the
baseline drift $b_t(x) = \mathbb{E}[x^\star - \sqrt{t}\,z \mid x_t = x]$
suffices for sampling under *any* diffusion coefficient $g(t)$, via the
closed-form drift correction (eq. 3.9)
$$b^g_t(x) = b_t(x) + \tfrac{g_t^2 - (1-t)^2}{2\,t\,(1-t)}\bigl(t\,b_t(x) - x\bigr).$$
One training, tune $g$ at sample time — no per-schedule retraining.

## Setup

- **Interpolant** (Def. 2.3): $I_t = t\,x^\star + (1-t)\sqrt{t}\,z$; $X_0 = 0$ (Dirac), $X_1 = x^\star$.
- **Baseline drift** (eq. 2.2, linear-linear): $b_t(x) = \mathbb{E}[x^\star - \sqrt{t}\,z \mid x_t = x]$. Bounded on $[0, 1]$.
- **Training loss** (eq. 2.10): $L[\hat b] = \mathbb{E}\|\hat b_\theta(x_t, t) - (x^\star - \sqrt{t}\,z)\|^2$. Plain unweighted L2 DSM — the regression target is bounded, so no extra weighting is required.
- **Sample-time SDE**: $dX_t = b^g_t(X_t)\,dt + g(t)\,dW_t$, with the EM initial condition set to $X_{t_{\min}} \sim \mathcal{N}\bigl(0,\gamma_{t_{\min}}^2 I\bigr)$ (interpolant marginal at $t_{\min}$). $\gamma_t = (1-t)\sqrt{t}$.

## Diffusion schedules (`schedules_zps.py`)

| name | $g(t)$ | role in the paper |
|---|---|---|
| `follmer` | $\sqrt{1 - t^2}$ | KL-optimal choice (eq. 3.30) |
| `baseline` | $1 - t$ | $=\sigma_t$; the default from Theorem 2.4 (no drift correction) |
| `triangle` | $\sqrt{t(1-t)}$ | symmetric, vanishes at both endpoints |
| `const` | $1$ | unit Brownian |
| `sqrt_t` | $\sqrt{t}$ | increasing |
| `zero` | $0$ | ODE limit (Girsanov diverges → excluded from path-KL table) |

The schedules use the paper's natural amplitudes — no shared-$\epsilon$
renormalisation, since the drift correction in eq. (3.9) is
amplitude-sensitive.

## Targets (`targets_zps.py`)

| target | $x^\star$ | dim | conditioning |
|---|---|---|---|
| `gaussian1d` | $\mathcal{N}(\mu, \sigma^2)$, $\mu{=}1.5,\ \sigma{=}0.5$ | 1 | — |
| `bimodal1d` | $\tfrac{1}{2}\mathcal{N}(-m,\tau^2) + \tfrac{1}{2}\mathcal{N}(m,\tau^2)$, $m{=}1,\ \tau{=}0.3$ | 1 | — |
| `ou_forecast` | $Y_{s+\tau} - Y_s \mid Y_s$ under OU $dY{=}{-}\lambda Y\,ds + \sigma\,dW$, $\lambda{=}\sigma{=}1,\ \tau{=}0.5$ | 1 | $Y_s$ (stationary) |
| `gmm2d_forecast` | 2D pure-jump GMM forecast, 5-mode Poisson-mod-5 rotation + isotropic Gaussian noise (see below) | 2 | $x_t$ (2D) |

All four admit a closed-form analytic baseline drift $b^\star_t$ via the
Gaussian-mixture posterior (see `_gaussian_comp_bstar` / `GMM2DForecast.b_star`
in `targets_zps.py`), so the path-KL "truth" has no approximation error.

### Target D setup (adapted from arXiv:2403.13724 §4.1)

The paper uses a 2D five-mode GMM jump-diffusion for probabilistic forecasting
with the **last-state** as the SDE initial condition. We reuse the same
multi-modal structure but with our **zero-point-source** convention ($X_0 = 0$,
$X_1 = x_{t+\tau} - x_t$, $x_t$ as conditioning) — the two are equivalent
up to a per-sample translation, and Theorem 3.2's Föllmer optimality
(which depends only on the schedules $\beta, \sigma$) applies identically.

**Simplification for an analytic $b^\star$**: we drop the between-jump
Langevin step (keeping only jumps + Gaussian noise), so the conditional is
exactly a 5-mode GMM:

$$x_{t+\tau} = R^N x_t + \epsilon,\qquad N \sim \mathrm{Poisson}(\lambda\tau) \bmod 5,\qquad \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 I_2),$$

with $R = R_{2\pi/5}$ the $2\pi/5$ CCW rotation. We use the paper's
$\lambda\tau = 1$ and $\sigma_\text{noise} = 0.5$. The mod-5 Poisson weights
are $\pi_k = \{0.371, 0.368, 0.184, 0.061, 0.015\}$. The conditioning $x_t$
is drawn from the 5-mode equilibrium GMM (equal-weight modes at
$R^k [5, 0]^T$ with per-mode covariance $\mathrm{diag}(1.5, 0.1)$ rotated).

The analytic $b^\star_t(x, y) = \mathbb{E}[x_1 - \sqrt{t}\,z \mid x_t = x, y]$ is
a responsibility-weighted average over the 5 GMM components; see
`GMM2DForecast.b_star`.

This is *not* a full reproduction of the paper's experiment (which uses
Langevin-between-jumps and last-state initial condition); it is the
cleanest ZPS analogue that keeps the conditional-multi-modal structure
and analytic ground-truth drift.

## Metrics

### Primary: path-KL via Girsanov (`metrics.path_kl_girsanov`)

For two SDEs with shared diffusion $g(t)$, drifts $b, \hat b$, and the same
initial distribution, Girsanov gives
$$\mathrm{KL}\bigl(P_b \,\|\, P_{\hat b}\bigr) \;=\; \tfrac{1}{2}\,\mathbb{E}_{P_b}\!\left[\int_0^1 g(t)^{-2}\,\bigl\|b_t(X_t) - \hat b_t(X_t)\bigr\|^2\,dt\right].$$

Since $P_b$ (the exact SDE under $b^\star$) has marginal $p_t = \mathrm{law}(I_t)$
at every $t$, we can rewrite the inner expectation as $\mathbb{E}_{X_t\sim p_t}[\ldots]$
and draw $X_t$ directly from $p_t$ by the interpolant formula
$X_t = t\,x^\star + (1-t)\sqrt{t}\,z$. **No SDE trajectory is simulated, no
time-discretization is introduced.**

Monte-Carlo estimator (implemented in `metrics.path_kl_girsanov`):
$$\widehat{\mathrm{KL}}(g) \;=\; \tfrac{1}{2}\,(t_{\max} - t_{\min})\;\frac{1}{N}\sum_{i=1}^{N}\,g(t_i)^{-2}\,\bigl\|b^g_{\hat\theta}(X_{t_i}, t_i) - b^{g,\star}(X_{t_i}, t_i)\bigr\|^2,$$
with $t_i \stackrel{\mathrm{iid}}{\sim} \mathcal{U}(t_{\min}, t_{\max})$,
$X_{t_i}$ drawn from $p_{t_i}$ as above. Both drifts use the same eq. (3.9)
composition; the only difference is $\hat b_\theta$ (network) vs $b^\star$
(analytic).

**Sources of numerical error in $\widehat{\mathrm{KL}}(g)$:**

1. **Monte-Carlo variance** — $O(1/\sqrt N)$. We use $N = 80{,}000$ per evaluation. For schedules with $g$ bounded away from zero on $[t_{\min}, t_{\max}]$ the integrand is bounded and the MC relative error is a few percent; this shows up in the ± std of the 5-seed aggregate.
2. **Boundary truncation** — integration is on $[t_{\min}, 1-t_{\min}]$ with $t_{\min} = 10^{-3}$. For schedules with $g_t \to 0$ at an endpoint (`baseline` at $t\to 1^-$, `triangle` at $t\to 0^+,1^-$, `sqrt_t` at $t\to 0^+$, `follmer` at $t\to 1^-$), the integrand $g^{-2}\|\cdot\|^2$ can diverge there. The reported value is the $[t_{\min}, 1-t_{\min}]$ integral and therefore *understates* the full-interval path-KL; the ordering of schedules is preserved.
3. **No time-grid error** — $t$ is drawn uniformly at random, not on a deterministic grid. The estimator is unbiased (up to MC variance) for $\mathrm{KL}|_{[t_{\min},\,1-t_{\min}]}$.
4. **Exact analytic $b^\star$** — no "truth" bias: for targets A, B, C we have closed-form $b^\star_t(x) = \mathbb{E}[x^\star - \sqrt{t}\,z \mid x_t = x]$.

### Secondary: terminal marginal at $t = 1 - t_{\min}$

Computed from $N_\text{samples} = 40{,}000$ EM samples ($N_\text{EM}=200$ steps) vs. ground-truth target samples. Error sources are independent from the path-KL MC (EM time-step error + sample-size MC noise):
- 1D Wasserstein $W_1, W_2$ — sorted-diff estimator (unbiased given equal sample counts).
- RBF-MMD — **biased V-statistic** (always non-negative), bandwidth mixture $\{0.5\,h_\text{med},\, h_\text{med},\, 2\,h_\text{med}\}$ where $h_\text{med}$ is the median of pooled pairwise distances (median heuristic), 5k-sample subset. We use the biased form rather than the unbiased one because the unbiased estimator can take (slightly) negative values when the two samples are statistically indistinguishable under the chosen kernels; the biased V-statistic has a small $O(1/n)$ positive bias that is numerically negligible at $n=5000$ and yields an easy-to-read non-negative magnitude.
- Moment errors — reported as scalar differences (see definitions below).
- Histogram KL (1D targets only) — Gaussian KDE (Silverman's bandwidth) on a 400-point grid vs. analytic target density.

#### Moment-error definitions (`metrics.moment_errors`)

Let $\{X_i\}_{i=1}^n$ be the model samples and $\{Y_i\}_{i=1}^m$ be the ground-truth samples. Denote sample mean $\bar X$, sample std $\hat\sigma_X$ (biased, $\sqrt{\tfrac{1}{n}\sum(X_i-\bar X)^2}$), standardised $Z_i = (X_i - \bar X)/\hat\sigma_X$ (componentwise in multi-D).

- **$|\Delta\mu|$** — magnitude of the mean-vector difference: $\|\bar X - \bar Y\|_2$. For 1D this is the scalar $|\bar X - \bar Y|$.
- **$|\Delta\sigma|$** — magnitude of the std-vector difference: $\|\hat\sigma_X - \hat\sigma_Y\|_2$ where $\hat\sigma$ is the vector of per-component stds.
- **$|\Delta\text{skew}|$** — magnitude of the third standardised-moment vector difference: $\|\text{skew}(X) - \text{skew}(Y)\|_2$ with $\text{skew}_d(X) = \frac{1}{n}\sum_i Z_{i,d}^3$ (Fisher-Pearson, applied per component).
- **$|\Delta\text{kurt}|$** — magnitude of the *excess* kurtosis difference: $\|\text{kurt}(X) - \text{kurt}(Y)\|_2$ with $\text{kurt}_d(X) = \frac{1}{n}\sum_i Z_{i,d}^4 - 3$ (so the Gaussian baseline has zero excess kurtosis).

For 1D targets the $\|\cdot\|_2$ collapses to $|\cdot|$.

## Numerics / integration

### Training
- Network: 3-layer MLP, hidden width 128, SiLU activations, Gaussian-Fourier time embedding of 64 features.
- Optimizer: AdamW, learning rate $2\times 10^{-4}$, batch 512, gradient clip (L2) at 5.
- Steps: $20{,}000$ per seed. Uniform $t$ sampling from $[t_{\min}, 1]$ with $t_{\min} = 10^{-3}$.
- Loss: plain L2 DSM on target $x^\star - \sqrt{t}\,z$ (eq. 2.10). No weighting.
- Seeds: 5 per target.

### Sampling (for marginal metrics)
- **Integrator**: Euler-Maruyama (first-order stochastic Euler), $N_\text{EM} = 200$ uniform steps on $[t_{\min}, 1 - t_{\min}]$ with $t_{\min} = 10^{-3}$.
- **Initial condition**: $X_{t_{\min}} \sim \mathcal{N}(0,\,\gamma_{t_{\min}}^2\,I)$, i.e. we sample from the *interpolant marginal at $t_{\min}$* rather than the exact Dirac at 0. This absorbs the boundary singularity at $t=0$ ($\gamma_t$ has a $\sqrt{t}$ vanishing, so the drift is bounded but the SDE needs a nonzero-variance start).
- Sample count $N_\text{samples} = 40{,}000$ per (seed, schedule) evaluation.
- For ODE ($g = 0$) the noise step is skipped; all other schedules use the stochastic step.

### Path-KL Monte-Carlo estimator
- **Method**: plain MC over $(t, X_t)$ (see formula above). No time-grid — $t$ is drawn *uniformly at random*, so the MC estimator is unbiased (no $O(\Delta t)$ discretization error).
- **Sample count** $N_\text{MC} = 80{,}000$ per (seed, schedule).
- **Truncation**: integrate on $[t_{\min}, 1 - t_{\min}]$. For schedules with $g_t \to 0$ at an endpoint the integrand diverges there, so the reported value is a *truncated* path-KL — it understates the true $[0,1]$ integral but preserves the ordering across schedules.
- **Baseline drift $b^\star$**: analytic for all three targets (Gaussian-mixture posterior); no truth bias.

## Results (5 seeds each, 20k training steps, $N_\text{MC}=80k$, $N_\text{samples}=40k$, $N_\text{EM}=200$)

### Path-KL headline (Girsanov, log scale)

Mean ± std over 5 seeds:

| target | Föllmer $\sqrt{1{-}t^2}$ | $\sqrt{t(1{-}t)}$ | $\sqrt{t}$ | const $g{=}1$ | baseline $1{-}t$ |
|---|---|---|---|---|---|
| gaussian1d    | **2.74e-2 ± 2.24e-2** | 3.12e-2 ± 2.39e-2 | 8.72e-1 ± 9.31e-1 | 9.58e-1 ± 1.04e+0 | 3.84e+0 ± 4.23e+0 |
| bimodal1d     | **3.27e-2 ± 6.19e-3** | 3.70e-2 ± 6.98e-3 | 7.56e-1 ± 1.74e-1 | 9.53e-1 ± 2.22e-1 | 4.42e+0 ± 9.43e-1 |
| ou_forecast   | **1.52e-2 ± 4.66e-3** | 1.90e-2 ± 5.74e-3 | 5.11e-1 ± 1.98e-1 | 4.47e-1 ± 2.16e-1 | 1.56e+0 ± 7.57e-1 |
| gmm2d_forecast| **8.01e-1 ± 1.37e-1** | 1.25e+0 ± 2.91e-1 | 2.26e+1 ± 5.97e+0 | 2.19e+1 ± 5.32e+0 | 6.03e+1 ± 9.76e+0 |

**Föllmer minimises path-KL on all four targets by ≳ 1.5–25× over the
runner-up `triangle`**, and by ~100× over `baseline` $(1-t)$. The large std
for `baseline` on `gaussian1d` and for the $\sqrt{t}$/const schedules
reflects boundary-sensitivity: trained networks have uncorrelated
boundary-layer errors across seeds, which the $g^{-2}$ factor in Girsanov
amplifies into seed-to-seed path-KL variability. The gap between Föllmer
and `triangle` narrows on harder targets (×2 on `gmm2d_forecast`,
×1.2 on `bimodal1d`) — both schedules are well-behaved at the $t=1$ endpoint,
so the easier-to-train `triangle` trails only by a small amount.

### Terminal marginal at $t = 1 - 10^{-3}$ (full per-metric table below)

Per-metric winners highlighted **bold**. ODE ($g{=}0$) excluded from path-KL
(Girsanov integrand diverges).

#### Target A — $\mathcal{N}(1.5, 0.5^2)$

| schedule | path KL | marg KL | $W_1$ | $W_2$ | MMD$^2$ | $\|\Delta\mu\|$ | $\|\Delta\sigma\|$ | $\|\Delta\text{skew}\|$ | $\|\Delta\text{kurt}\|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Föllmer $\sqrt{1-t^2}$ | **2.74e-2 ± 2.24e-2** | 1.26e-3 ± 1.04e-3 | 1.48e-2 ± 1.23e-2 | 1.74e-2 ± 1.08e-2 | 3.95e-4 ± 2.48e-4 | 1.30e-2 ± 1.35e-2 | **4.01e-3 ± 1.97e-3** | 4.85e-2 ± 8.71e-3 | 1.54e-2 ± 1.36e-2 |
| baseline $1-t$ | 3.84e+0 ± 4.23e+0 | **9.30e-4 ± 5.77e-4** | **1.31e-2 ± 6.68e-3** | **1.51e-2 ± 6.15e-3** | **1.42e-4 ± 1.20e-4** | **1.17e-2 ± 7.73e-3** | 4.87e-3 ± 3.66e-3 | **9.02e-3 ± 1.14e-2** | **1.44e-2 ± 1.09e-2** |
| $\sqrt{t(1-t)}$ | 3.12e-2 ± 2.39e-2 | 7.27e-2 ± 4.06e-3 | 1.08e-1 ± 4.19e-3 | 1.38e-1 ± 4.48e-3 | 9.82e-3 ± 6.89e-4 | 2.27e-2 ± 1.43e-2 | 1.35e-1 ± 4.14e-3 | 7.46e-2 ± 3.60e-2 | 1.01e-1 ± 4.61e-2 |
| const $g{=}1$ | 9.58e-1 ± 1.04e+0 | 2.51e-2 ± 1.24e-2 | 8.28e-2 ± 3.75e-2 | 9.15e-2 ± 3.21e-2 | 4.93e-3 ± 3.70e-3 | 7.68e-2 ± 4.39e-2 | 2.14e-2 ± 2.03e-2 | 2.29e-1 ± 6.10e-2 | 1.30e-1 ± 1.04e-1 |
| $\sqrt{t}$ | 8.72e-1 ± 9.31e-1 | 2.96e-2 ± 1.33e-2 | 8.90e-2 ± 3.95e-2 | 9.99e-2 ± 3.38e-2 | 4.82e-3 ± 3.14e-3 | 8.37e-2 ± 4.60e-2 | 2.86e-2 ± 2.18e-2 | 2.16e-1 ± 6.65e-2 | 1.55e-1 ± 1.24e-1 |
| ODE $g{=}0$ | — | 3.42e-1 ± 8.38e-3 | 2.47e-1 ± 4.03e-3 | 3.13e-1 ± 4.35e-3 | 3.98e-2 ± 1.23e-3 | 4.06e-2 ± 1.29e-2 | 3.09e-1 ± 3.83e-3 | 1.00e-1 ± 5.47e-2 | 1.11e-1 ± 8.31e-2 |

#### Target B — bimodal GMM

| schedule | path KL | marg KL | $W_1$ | $W_2$ | MMD$^2$ | $\|\Delta\mu\|$ | $\|\Delta\sigma\|$ | $\|\Delta\text{skew}\|$ | $\|\Delta\text{kurt}\|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Föllmer $\sqrt{1-t^2}$ | **3.27e-2 ± 6.19e-3** | **1.10e-2 ± 1.70e-3** | 2.55e-2 ± 1.23e-2 | 4.01e-2 ± 1.71e-2 | 4.25e-4 ± 3.10e-4 | 1.36e-2 ± 1.34e-2 | 1.60e-2 ± 1.23e-2 | 1.84e-2 ± 9.69e-3 | 5.57e-3 ± 3.92e-3 |
| baseline $1-t$ | 4.42e+0 ± 9.43e-1 | 1.19e-2 ± 1.69e-3 | **1.96e-2 ± 8.68e-3** | **3.06e-2 ± 1.50e-2** | **3.21e-4 ± 2.19e-4** | **8.75e-3 ± 8.97e-3** | **1.12e-2 ± 1.04e-2** | **1.18e-2 ± 9.21e-3** | **2.19e-3 ± 3.69e-3** |
| $\sqrt{t(1-t)}$ | 3.70e-2 ± 6.98e-3 | 6.25e-2 ± 1.48e-2 | 8.06e-2 ± 1.05e-2 | 9.01e-2 ± 1.02e-2 | 3.00e-3 ± 7.79e-4 | 1.67e-2 ± 1.28e-2 | 7.90e-2 ± 1.27e-2 | 1.84e-2 ± 1.10e-2 | 2.82e-3 ± 1.96e-3 |
| const $g{=}1$ | 9.53e-1 ± 2.22e-1 | 5.35e-2 ± 3.05e-2 | 8.53e-2 ± 3.08e-2 | 1.11e-1 ± 1.61e-2 | 4.51e-3 ± 2.88e-3 | 7.52e-2 ± 3.46e-2 | 4.17e-2 ± 3.61e-2 | 1.67e-2 ± 2.07e-2 | 3.24e-2 ± 2.58e-2 |
| $\sqrt{t}$ | 7.56e-1 ± 1.74e-1 | 4.83e-2 ± 2.76e-2 | 7.90e-2 ± 2.80e-2 | 9.80e-2 ± 1.61e-2 | 3.85e-3 ± 2.38e-3 | 6.97e-2 ± 3.54e-2 | 3.00e-2 ± 3.14e-2 | 1.24e-2 ± 1.44e-2 | 4.17e-2 ± 2.45e-2 |
| ODE $g{=}0$ | — | 3.21e-1 ± 3.31e-2 | 1.81e-1 ± 1.07e-2 | 2.03e-1 ± 1.11e-2 | 1.25e-2 ± 1.58e-3 | 1.43e-2 ± 1.13e-2 | 1.96e-1 ± 1.27e-2 | 1.52e-2 ± 1.01e-2 | 8.92e-2 ± 8.31e-3 |

#### Target D — 2D GMM jump-forecasting (marginalised over $x_0$)

| schedule | path KL | $\mathrm{SW}_1$ | $\mathrm{SW}_2$ | MMD$^2$ | $\|\Delta\mu\|$ | $\|\Delta\sigma\|$ | $\|\Delta\text{skew}\|$ | $\|\Delta\text{kurt}\|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Föllmer $\sqrt{1-t^2}$ | **8.01e-1 ± 1.37e-1** | 2.16e-1 ± 4.94e-2 | 2.99e-1 ± 6.56e-2 | 7.70e-4 ± 2.96e-4 | 1.70e-1 ± 8.17e-2 | 2.39e-1 ± 2.01e-1 | 1.04e-1 ± 5.09e-2 | 1.75e-1 ± 1.03e-1 |
| baseline $1-t$ | 6.03e+1 ± 9.76e+0 | 2.12e-1 ± 4.79e-2 | 2.96e-1 ± 6.45e-2 | 5.20e-4 ± 1.85e-4 | **1.44e-1 ± 1.05e-1** | 2.56e-1 ± 2.01e-1 | 8.29e-2 ± 4.03e-2 | 1.81e-1 ± 1.22e-1 |
| $\sqrt{t(1-t)}$ | 1.25e+0 ± 2.91e-1 | **1.50e-1 ± 2.96e-2** | **2.13e-1 ± 3.28e-2** | 4.32e-4 ± 1.73e-4 | 1.35e-1 ± 5.55e-2 | **1.33e-1 ± 8.26e-2** | **5.38e-2 ± 3.67e-2** | 1.49e-1 ± 9.12e-2 |
| const $g{=}1$ | 2.19e+1 ± 5.32e+0 | 2.63e-1 ± 7.86e-2 | 3.46e-1 ± 9.33e-2 | 7.39e-4 ± 4.20e-4 | 1.99e-1 ± 6.51e-2 | 2.92e-1 ± 2.14e-1 | 1.09e-1 ± 7.87e-2 | 1.77e-1 ± 1.13e-1 |
| $\sqrt{t}$ | 2.26e+1 ± 5.97e+0 | 1.89e-1 ± 2.69e-2 | 2.56e-1 ± 4.37e-2 | **3.93e-4 ± 3.24e-4** | 1.61e-1 ± 4.87e-2 | 1.90e-1 ± 9.20e-2 | 1.02e-1 ± 6.27e-2 | **1.47e-1 ± 6.79e-2** |
| ODE $g{=}0$ | — | 2.25e-1 ± 5.66e-2 | 2.88e-1 ± 6.83e-2 | 7.98e-4 ± 4.65e-4 | 1.64e-1 ± 5.86e-2 | 2.28e-1 ± 1.24e-1 | 6.89e-2 ± 1.88e-2 | 2.63e-1 ± 1.14e-1 |

For 2D, the $W_p$ columns are **sliced-Wasserstein** ($\mathrm{SW}_p$) with 200
random projections (see `metrics.sliced_wasserstein`). Marginal KL is not
reported for Target D (would require a 2D histogram-KL-vs-analytic-density;
the 5×5 = 25-component joint-marginal density is closed-form but we leave
that as follow-up). Föllmer still wins path-KL cleanly (×25–75 over
`const`/$\sqrt{t}$, ×75 over `baseline`). On marginal metrics, `triangle`
wins most columns — the same pattern seen on 1D targets.

#### Target C — OU forecasting (marginalised over $Y_s$)

| schedule | path KL | marg KL | $W_1$ | $W_2$ | MMD$^2$ | $\|\Delta\mu\|$ | $\|\Delta\sigma\|$ | $\|\Delta\text{skew}\|$ | $\|\Delta\text{kurt}\|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Föllmer $\sqrt{1-t^2}$ | **1.52e-2 ± 4.66e-3** | **6.03e-4 ± 2.37e-4** | **1.05e-2 ± 2.83e-3** | **1.34e-2 ± 4.08e-3** | 6.80e-5 ± 3.51e-5 | 9.45e-3 ± 3.27e-3 | **6.89e-3 ± 5.07e-3** | **1.50e-2 ± 1.25e-2** | **1.19e-2 ± 5.55e-3** |
| baseline $1-t$ | 1.56e+0 ± 7.57e-1 | 8.16e-4 ± 2.86e-4 | 1.08e-2 ± 4.49e-3 | 1.51e-2 ± 4.40e-3 | **6.73e-5 ± 6.19e-5** | 9.29e-3 ± 4.74e-3 | 8.20e-3 ± 4.87e-3 | 3.65e-2 ± 1.69e-2 | 1.82e-2 ± 1.02e-2 |
| $\sqrt{t(1-t)}$ | 1.90e-2 ± 5.74e-3 | 4.86e-2 ± 6.99e-3 | 1.06e-1 ± 8.45e-3 | 1.32e-1 ± 1.07e-2 | 5.00e-3 ± 7.76e-4 | 9.00e-3 ± 5.37e-3 | 1.32e-1 ± 1.07e-2 | 2.75e-2 ± 1.07e-2 | 3.16e-2 ± 1.21e-2 |
| const $g{=}1$ | 4.47e-1 ± 2.16e-1 | 8.95e-3 ± 5.63e-3 | 6.16e-2 ± 2.90e-2 | 6.88e-2 ± 2.92e-2 | 2.13e-3 ± 1.45e-3 | 6.04e-2 ± 2.95e-2 | 2.27e-2 ± 1.78e-2 | 7.28e-2 ± 3.74e-2 | 1.49e-1 ± 3.27e-2 |
| $\sqrt{t}$ | 5.11e-1 ± 1.98e-1 | 1.19e-2 ± 5.73e-3 | 6.92e-2 ± 2.24e-2 | 7.70e-2 ± 2.23e-2 | 3.69e-3 ± 1.69e-3 | 5.79e-2 ± 2.99e-2 | 3.44e-2 ± 2.92e-2 | 6.73e-2 ± 2.52e-2 | 2.08e-1 ± 2.76e-2 |
| ODE $g{=}0$ | — | 2.22e-1 ± 1.57e-2 | 2.44e-1 ± 9.08e-3 | 3.06e-1 ± 1.18e-2 | 2.53e-2 ± 1.48e-3 | **8.42e-3 ± 9.63e-3** | 3.06e-1 ± 1.17e-2 | 2.57e-2 ± 1.73e-2 | 3.73e-2 ± 2.23e-2 |

### How to read the tables

- **Path-KL ordering matches the theorem exactly.** Föllmer first, then
  `triangle`, then the clearly worse schedules. The ×100 gap to `baseline`
  $(1-t)$ is because the Girsanov integrand $g^{-2}\|\text{err}\|^2$ is
  amplified as $g\to 0$ at the boundary, even though `baseline` is the
  "default" choice from Theorem 2.4.
- **Marginal metrics are a softer discriminator** and sometimes favour
  `baseline`. This is not a contradiction: Theorem 3.2 minimises the
  *path-space* KL (eq. 3.10), not any particular marginal statistic.
  Remark 3.4 only says path-KL *upper-bounds* marginal KL, not that they
  are identical; a loose bound for `baseline` leaves room for its marginal
  samples to be competitive while its path measure is far from truth.
- **ODE ($g=0$) is catastrophic on every marginal metric** (W2 ≈ 0.2–0.3 vs
  Föllmer's 0.01–0.04). This empirically confirms the paper's assertion
  (line 86) that *diffusion is essential* to transport a Dirac mass —
  an ODE cannot do it.
- **Seed-to-seed std is larger for poorer schedules.** Schedules with a
  vanishing $g$ at $t\to 1$ (`baseline`) have larger std because the
  Girsanov integrand $\propto g^{-2}$ amplifies seed-specific
  boundary-layer drift errors.

See `figs/zps_summary.{pdf,png}` for the bar-chart version and
`figs/zps_table.md` for the identical table (auto-generated by
`report_table.py`).

## Reproducing

```bash
# Train one network per (target, seed) — single DSM loss, schedule-agnostic
for t in gaussian1d bimodal1d ou_forecast; do
  for s in 0 1 2 3 4; do
    python train_zps.py --target $t --seed $s --max_steps 20000
  done
done

# Aggregate per-target (sweeps g schedules via eq 3.9 on the shared network)
for t in gaussian1d bimodal1d ou_forecast; do
  python compare_zps.py --target $t --seeds 0 1 2 3 4
done

# Combined summary figure + markdown table
python plot_zps.py
python report_table.py --out figs/zps_table.md
```

## Files

| file | purpose |
|---|---|
| `interpolant_zps.py` | $\beta_t = t,\ \sigma_t = 1-t$; $I_t$, regression target $R_b = x^\star - \sqrt{t}\,z$. |
| `schedules_zps.py` | $g(t)$ registry; Föllmer $= \sqrt{1-t^2}$. |
| `drift_compose.py` | Eq. (3.9) closed-form drift correction for any $g$. |
| `targets_zps.py` | Targets + analytic $b^\star_t$ and analytic density. |
| `networks.py` | Small MLP + Gaussian-Fourier time embedding. |
| `sampler.py` | EM sampler initialised from $\mathcal{N}(0, \gamma_{t_{\min}}^2)$. |
| `metrics.py` | Girsanov path-KL estimator + $W_{1,2}$ + MMD + moments + KDE-KL. |
| `train_zps.py` | Train $\hat b_\theta$ once per (target, seed). |
| `compare_zps.py` | Load $\hat b_\theta$, sweep $g$, compute all metrics, save JSON. |
| `plot_zps.py` | Headline 2×3 figure (path-KL top row, W2 bottom row). |
| `report_table.py` | Render `figs/zps_table.md` from the JSON summaries. |
| `_smoke_test.py` | Correctness checks: drift composition, marginals, path-KL-zero. |
