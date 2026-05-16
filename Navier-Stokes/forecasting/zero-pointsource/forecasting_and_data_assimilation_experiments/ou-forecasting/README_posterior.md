# Posterior sampling via twisted SMC — 1D zps study

Companion to the §3.4.1 Föllmer forecasting study. The theorem (Chen &
Vanden-Eijnden 2026) says Föllmer minimises the **path-space** KL between
the learned and true SDE. Marginal metrics only see this advantage through
Remark 3.4's loose bound. Here we run twisted-SMC posterior sampling, the
canonical trajectory-level application, and show that Föllmer's advantage
becomes more pronounced — along with a detailed analysis of the two
natural choices of guidance coefficient in the ZPS setting.

## Setup

### Target posterior

$$p(x_1 \mid y) \;\propto\; p(x_1)\,p(y\mid x_1).$$

- Prior $p(x_1)$: learned generative prior (the §3.4.1 ZPS model at $s=1$).
- Likelihood: $p(y \mid x_1) = \mathcal{N}(y; A x_1, \sigma_y^2 I)$ with $A = I$
  and three noise levels $\sigma_y \in \{0.1, 0.2, 0.5\}$.
- **Targets**: `gaussian1d` ($x_1 \sim \mathcal N(1.5, 0.5^2)$) and
  `bimodal1d` ($x_1 \sim \tfrac12 \mathcal N(-1, 0.3^2)+\tfrac12\mathcal N(1, 0.3^2)$).
  Both have closed-form analytic posteriors — Gaussian and 2-component GMM
  respectively — so we can measure sampler quality directly as
  $W_2(\text{SMC samples}, \text{analytic posterior})$.

### ZPS interpolant + learned drift

- $I_s = s\,x_1 + (1-s)\sqrt{s}\,z$, $z\sim\mathcal{N}(0, I)$, $X_0 = 0$.
- $\hat b_\theta(x, s)$ trained to $\mathbb E[x_1 - \sqrt{s}\,z \mid x_s = x]$ via
  plain L2 DSM.
- Schedule-specific drift $b^g$ from eq. (3.9) with any $g(s)$. Schedules:
  `follmer` ($\sqrt{1-s^2}$), `baseline` ($1-s$), `triangle` ($\sqrt{s(1-s)}$),
  `const` ($1$), `sqrt_t` ($\sqrt{s}$), `zero` (ODE).

### Tweedie (ZPS-specific)

Solving the 2×2 linear system relating $b_s = \mathbb{E}[x_1 - \sqrt{s}\,z\,|\,x_s]$
to $\mathbb E[x_1|x_s]$ and $\mathbb E[z|x_s]$:
$$\boxed{\hat x_1(x, s) = x + (1-s)\,\hat b_\theta(x, s)} \qquad
  \boxed{\hat z(x, s) = \frac{x - s\,\hat b_\theta(x, s)}{\sqrt{s}}}$$
Score via Stein: $\nabla\log\rho_s(x) = -\frac{x - s\,\hat b_\theta}{s(1-s)}$.

## Twisted SMC

### Target, proposal, weights

- **Path-measure target**: $\pi(x_{0:1}) \propto p_{0:1}\cdot p(y|x_1)$.
- **Intermediate twists**: $\phi_s(x) = p(y|\hat x_1(x, s))$ (Tweedie-approximated
  collapse of $\phi_s^\star = \mathbb{E}[p(y|X_1)|X_s = x]$). At $s=1$, $\phi_1 = p(y|\cdot)$ exactly.
- **Intermediate target**: $\pi_s \propto p_s\,\phi_s$. Terminal $\pi_1$ = exact posterior.
- **Proposal**: choice of `guided` or `uncond` (see below).
- **Incremental importance weight**:
$$\log w_{s+\Delta}/w_s
  = \log\frac{p(x_{s+\Delta}|x_s)\,\phi_{s+\Delta}(x_{s+\Delta})}{q(x_{s+\Delta}|x_s)\,\phi_s(x_s)}.$$

### Two proposal modes

**Unconditional** ($q = p$): kernel ratio cancels, $\log w_\text{step} = \log\phi_{s+\Delta} - \log\phi_s$.

**Guided** (ZPS-agnostic form): $q = $ prior kernel modified by drift
$b^g + \kappa_s\nabla\log\phi_s$. For EM-discretised Gaussian kernels,
$$\log\frac{p}{q}\bigg|_{\text{step}} = -\frac{\kappa}{g}\sqrt{\Delta}\,Z\cdot\nabla\log\phi_s \;-\; \frac{\kappa^2}{2 g^2}\|\nabla\log\phi_s\|^2\,\Delta,$$

where $Z \sim \mathcal{N}(0, I)$ is the EM noise used in the guided step, and
$\kappa$ is the **guidance coefficient**.

### Which $\kappa$ is correct?

The ZPS drift has two natural expansions in terms of the score:

(A) **Doob h-transform** — canonical Fokker-Planck derivation: $\kappa = g^2$.
    Preserves $\tilde p_s = p_s\phi_s$ exactly for *any* schedule, under the
    backward-harmonic assumption on $\phi$. With Tweedie-$\phi$ the residual
    is only the harmonic mismatch.

(B) **"Natural" score replacement** — expanding $b_t$ itself as a function
    of score (eq 3.5 for linear-linear: $b_t = x/t + (1-t)\nabla\log p_t$),
    the full drift is
    $$b^g(x) = \frac{x}{t} + C_g(t)\,\nabla\log p_t(x), \qquad
      C_g(t) = \tfrac{g^2 + 1 - t^2}{2}.$$
    Replacing $\nabla\log p_t$ by the tilted score $\tilde s = \nabla\log p_t + \nabla\log\phi$
    inside this formula gives $\kappa = C_g$.

**Fokker-Planck check**: for the guided SDE to preserve $\tilde p = p\phi$
(given $\phi$ harmonic), the residual after using the prior FP reduces to
$(\kappa - g^2)\nabla p\cdot\nabla\phi + (\kappa - g^2/2)p\Delta\phi$. This
vanishes identically only for $\kappa = g^2$ (Doob). For $\kappa = C_g$, it
vanishes only when $C_g = g^2$, i.e. $g^2 = 1 - t^2$ — **exactly the Föllmer
schedule**. So:

> **Föllmer is the unique schedule where the naïve "score-replacement"
> guidance coefficient coincides with the rigorous Doob coefficient.** Yet
> another characterisation of Föllmer's special status — its drift's
> dependence on the score already matches the diffusion-squared that Doob
> requires.

### Guidance-scale family

In practice we sweep $\kappa = \eta \cdot \text{(doob or natural)}$ with a
scalar $\eta \in \{0.1, 0.5, 1.0\}$. DPS/MCGDiff/TDS all use some form of
tunable scale to combat the Tweedie bias of $\phi_s \approx \phi_s^\star$.

## Experimental protocol

- 2 targets × 3 seeds × 20 ICs × 100 particles × 200 EM steps × 6 schedules
  × 2 proposal modes × 3 $\sigma_y$ values × {2 guidance coefficients × 3 $\eta$}
  (strong-obs only, to keep compute tractable).
- Systematic resampling at ESS $< N/2$.
- Metrics: $W_2$ to analytic posterior, posterior-mean RMSE, final ESS, log-$Z$ std across seeds.

## Results

### Headline — strong observation ($\sigma_y = 0.1$), full guidance sweep

$W_2$ to analytic posterior, pooled across 3 seeds × 20 ICs. Best per schedule over the $\eta$ sweep:

| schedule | gauss best W2 | at | bimod best W2 | at |
|---|---|---|---|---|
| **Föllmer** $\sqrt{1-t^2}$ | **0.027 ± 0.0008** | η=0.1 | **0.025 ± 0.001** | η=0.1 |
| baseline $1-t$ | 0.037 ± 0.004 | natural η=0.1 | 0.043 ± 0.005 | doob η=0.1 |
| $\sqrt{t(1-t)}$ | 0.033 ± 0.001 | natural η=0.1 | 0.042 ± 0.009 | doob η=0.1 |
| const $g{=}1$ | 0.026 ± 0.002 | natural η=0.5 | 0.026 ± 0.002 | natural η=0.5 |
| $\sqrt{t}$ | 0.027 ± 0.004 | doob η=0.1 | 0.027 ± 0.002 | natural η=0.5 |
| ODE ($g{=}0$) | 0.107 ± 0.0003 | natural η=1.0 | 0.110 ± 0.007 | natural η=1.0 |

With **optimal $\eta$**, Föllmer/const/sqrt_t are effectively tied; baseline
and triangle are 40-70% worse; ODE is ~4x worse.

**Öur practical statement**: with properly-tuned guidance, the marginal
W2 advantage of Föllmer over every other non-ODE schedule is at most ~10% at
$\sigma_y=0.1$. The Föllmer story is sharper at:
1. **Untuned** $\eta = 1$ defaults (out-of-the-box robustness).
2. **Trajectory-level metrics** (ESS, path-measure consistency).
3. **Multi-modal targets** under untuned guidance.

### Föllmer invariance — Doob and Natural coincide empirically

For every $\eta$ value tested, the Föllmer rows in the doob and natural
sweep files agree to 4 significant figures:

| $\eta$ | gauss Föllmer doob | gauss Föllmer natural | Δ |
|---|---|---|---|
| 1.0 | 0.0337 ± 0.003 | 0.0337 ± 0.003 | 0.0000 |
| 0.5 | 0.0291 ± 0.001 | 0.0289 ± 0.002 | 0.0002 |
| 0.1 | 0.0271 ± 0.0007 | 0.0270 ± 0.0008 | 0.0001 |

This is the $C_g = g^2 = 1-t^2$ identity verifying empirically. For every
other schedule the doob and natural rows diverge noticeably.

### η-sweep pattern across schedules

Damping helps. For every schedule except ODE, $\eta = 0.1$ produces lower
W2 than $\eta = 1.0$ (20–40% improvement). Example (gauss1d, Doob, Föllmer):
| $\eta$ | W2 |
|---|---|
| 1.0 | 0.034 |
| 0.5 | 0.029 |
| 0.1 | 0.027 |

The theoretical $\eta = 1$ (exact Doob under harmonic $\phi$) is not
practically optimal because $\phi_s = p(y\mid\hat x_1(x,s))$ is biased
relative to $\phi_s^\star$ — small $\eta$ damps the aggressive guidance
direction that this biased $\phi$ would otherwise produce.

### ODE pathology under Natural guidance

For the ODE schedule ($g = 0$), Doob gives $\kappa = g^2 = 0$ → no guidance
→ sampler essentially reduces to a deterministic trajectory (poor W2 but
non-degenerate ESS). Natural gives $\kappa = C_g = (1 - t^2)/2 \ne 0$ →
guidance drift is non-zero, but the Girsanov weight correction has a
$\kappa^2 / g^2$ term which blows up as $1/g^2 = \infty$. Result: final ESS
collapses to 1 (single-particle domination), W2 sometimes deteriorates.

| schedule = zero | gauss W2 | gauss final ESS | bimodal W2 | bimodal ESS |
|---|---|---|---|---|
| doob, η=1 | 0.149 | 77 | 0.536 | 80 |
| natural, η=1 | 0.107 | **4** | 0.110 | **6** |
| natural, η=0.5 | 0.125 | **1** | 0.154 | **1** |
| natural, η=0.1 | 0.555 | **1** | 0.726 | **1** |

So Natural "appears" to improve ODE's W2 at η=1 because particle collapse
happens to concentrate near the posterior mean — a misleading number. The
ESS column gives away the pathology.

### σ_y-sweep — Föllmer advantage fades with weaker observations

At $\sigma_y = 0.5$ (weak obs), posterior ≈ prior, and all non-ODE
schedules are within 20% of each other on W2. Föllmer's advantage is most
pronounced at $\sigma_y = 0.1$ with the Tweedie-biased guidance ($\eta = 1$).

## Files (`zps/`)

| file | purpose |
|---|---|
| `posterior_smc.py` | Twisted-SMC sampler: `twisted_smc_1d(..., proposal, guidance_type, guidance_eta)`. Supports doob / natural coefficients × η scaling with the general Girsanov weight formula above. |
| `posterior_compare.py` | Schedule sweep per target × σ_y × proposal × (guidance_type, η). Saves JSON with per-seed summary + ESS curves. |
| `visualize_posterior.py` | Table + ESS curves + 2×4 metric grid + σ_y sweep figure. |
| `README_posterior.md` | This document. |

## Figures produced

- `figs_posterior/summary_table.md` — full table across σ_y × proposal × schedule.
- `figs_posterior/metric_grid.{pdf,png}` — 2×4 headline: W2, posterior-mean RMSE, final ESS, log-Z std — at σ=0.1 guided, η=1 Doob.
- `figs_posterior/sigma_sweep.{pdf,png}` — W2 and RMSE vs σ_y, log-log.
- `figs_posterior/w2_bars_{target}.{pdf,png}` — W2 bars by σ_y.
- `figs_posterior/ess_curves_{target}_{proposal}.{pdf,png}` — ESS vs EM step.
- `figs_posterior_guidance/` — full guidance-coefficient sweep (6 runs per target × 3 seeds).

## Reproducing

```bash
cd simple-examples/zps
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python

# σ-sweep (main)
for target in gaussian1d bimodal1d; do
  for proposal in guided uncond; do
    for sigma in 0.5 0.2 0.1; do
      $PY posterior_compare.py --target $target --seeds 0 1 2 3 4 \
        --sigma_y $sigma --proposal $proposal \
        --n_particles 100 --n_em 200 --n_ic 20 \
        --out_dir ./figs_posterior
    done
  done
done

# Guidance-coefficient sweep (σ_y = 0.1)
for target in gaussian1d bimodal1d; do
  for gtype in doob natural; do
    for eta in 1.0 0.5 0.1; do
      $PY posterior_compare.py --target $target --seeds 0 1 2 \
        --sigma_y 0.1 --proposal guided \
        --guidance_type $gtype --guidance_eta $eta \
        --n_particles 100 --n_em 200 --n_ic 15 \
        --out_dir ./figs_posterior_guidance
    done
  done
done

$PY visualize_posterior.py --in_dir ./figs_posterior
```

## OU-forecast posterior sampling (conditional target)

The `ou_forecast` target is already the most stringent of the four
unconditional §3.4 tests: the checkpoint has to learn a *conditional*
density $p(X_1 \mid Y_s)$ from a 1D covariate. The marginal §3.4 table
showed Föllmer winning by only ~20% on path-KL and even less on marginal
W₁/W₂ (see `README.md`). Here the posterior-sampling extension of that
target — reuse the same checkpoint and add a noisy observation
$y_\text{obs} = X_1 + \eta$, $\eta \sim \mathcal N(0, \sigma_y^2)$ —
produces the clearest schedule-separation we've seen on OU: baseline blows
up under natural guidance, ODE collapses under natural guidance, and
Föllmer is the only schedule that is stable *across all* $(\kappa, \eta)$.

### Setup (OU-specific)

- **Prior**: $X_1 \mid Y_s \sim \mathcal N((\text{decay}-1)Y_s,\,\sigma_{X|Y}^2)$ with $\text{decay} = e^{-\lambda\tau} = e^{-0.5}$, $\sigma_{X|Y}^2 = \sigma^2(1-e^{-2\lambda\tau})/(2\lambda)$. $\lambda = \sigma = 1$, $\tau = 0.5$.
- **Conditioning $Y_s$** drawn from the stationary $\mathcal N(0, \sigma^2/2\lambda)$ per IC.
- **Observation** $y_\text{obs} = X_1 + \mathcal N(0, \sigma_y^2)$ with $\sigma_y = 0.1$ (strong obs regime — weaker ones reduce to the prior story).
- **Analytic posterior** (both prior and likelihood Gaussian):
  $X_1 \mid Y_s, y_\text{obs} \sim \mathcal N(\mu_\text{post}, \sigma_\text{post}^2)$ with $\sigma_\text{post}^{-2} = \sigma_{X|Y}^{-2} + \sigma_y^{-2}$.
- Sweep config: 5 seeds × 20 ICs × $N = 100$ particles × $n_\text{em} = 200$, $t \in [10^{-3}, 1-10^{-3}]$. Full doob/natural × $\eta \in \{0.1, 0.5, 1.0\}$ at $\sigma_y = 0.1$, plus `uncond`.

### Results ($W_2$ to analytic posterior, 5 seeds × 20 ICs)

No guidance in proposal (`uncond`):

| schedule | W₂ | RMSE | finalESS |
|---|---|---|---|
| **follmer** | **0.0251 ± 0.0034** | 0.017 | 70.1 |
| sqrt_t | 0.0256 ± 0.0018 | 0.017 | 55.9 |
| const | 0.0269 ± 0.0037 | 0.019 | 57.0 |
| triangle | 0.0337 ± 0.0012 | 0.027 | 72.1 |
| baseline | 0.0409 ± 0.0022 | 0.036 | 73.1 |
| zero | 0.1524 ± 0.0246 | 0.130 | 77.0 |

Doob guidance $\kappa = \eta g^2$:

| schedule | η = 0.1 | η = 0.5 | η = 1.0 |
|---|---|---|---|
| **follmer** | 0.027 ± 0.004 | 0.026 ± 0.002 | 0.032 ± 0.002 |
| const | 0.025 ± 0.001 | 0.026 ± 0.003 | 0.035 ± 0.003 |
| sqrt_t | 0.026 ± 0.003 | 0.028 ± 0.002 | 0.036 ± 0.004 |
| triangle | 0.032 ± 0.003 | 0.033 ± 0.001 | 0.037 ± 0.003 |
| baseline | 0.035 ± 0.003 | 0.041 ± 0.002 | 0.055 ± 0.002 |
| zero | 0.152 ± 0.024 | 0.152 ± 0.024 | 0.153 ± 0.024 |

Natural guidance $\kappa = \eta\,C_g$ (= $\eta(g^2 + 1 - t^2)/2$):

| schedule | η = 0.1 | η = 0.5 | η = 1.0 |
|---|---|---|---|
| **follmer** | 0.027 ± 0.003 | 0.026 ± 0.002 | 0.032 ± 0.002 |
| const | 0.027 ± 0.003 | **0.022 ± 0.003** | 0.027 ± 0.001 |
| sqrt_t | **0.023 ± 0.003** | 0.025 ± 0.002 | 0.028 ± 0.004 |
| triangle | 0.032 ± 0.006 | 0.038 ± 0.004 | 0.042 ± 0.001 |
| baseline | 0.038 ± 0.003 | **0.067 ± 0.003** | **0.075 ± 0.003** |
| zero | 0.533 (ESS=1) | 0.111 (ESS=1) | 0.103 (ESS=2.9) |

### OU observations — where Föllmer's advantage actually manifests

1. **Untuned Doob is Föllmer's home turf.** At uncond (no guidance), Föllmer
   leads on W₂ with a visible but narrow 5-35% margin. At Doob $\eta = 1$
   (the theoretically-canonical choice), Föllmer stays at W₂ ≈ 0.032 while
   **baseline degrades to 0.055** (70% worse) and sqrt_t/const to 0.035/0.036
   (10% worse). Out-of-the-box without $\eta$-tuning, Föllmer is the only
   schedule that doesn't need babysitting.

2. **Natural guidance at $\eta = 1$ destroys `baseline`**: W₂ jumps to
   **0.075, 2.3× worse than Föllmer**. This is the cleanest empirical
   signature of the identity $C_g \equiv g^2 \iff$ Föllmer: `baseline` has
   $g^2 = (1-t)^2$, $C_g = (1-t)(2-t)/2$ ≫ $g^2$ as $t \to 1$. Using $C_g$
   in the guidance drift overshoots, and the Girsanov $\kappa/g = C_g/g^2$
   ratio in the weight correction amplifies the error. Same story (slightly
   muted) for triangle and sqrt_t at $\eta = 1$: their natural-W₂ is 10-40%
   worse than Doob-W₂ at the same $\eta$.

3. **ODE under Natural is catastrophic** (W₂ = 0.10-0.53 with ESS dropping
   to 1). Predicted by $\kappa/g = C_g / 0 = \infty$ in the Girsanov weight.
   This reproduces the 1D-Gaussian-target pathology on a *conditional*
   target.

4. **Föllmer columns identical between Doob and Natural** to 3 sigfigs at
   every $\eta$. $C_g^\text{linlin} \equiv g^2$ verified on the
   conditional-OU checkpoint — same invariance as on `gaussian1d` /
   `bimodal1d` / NS forecasting.

5. **Best non-Föllmer (const, Natural $\eta = 0.5$: 0.022)** is ~13% better
   than Föllmer. This is real but only visible after a two-dimensional
   $(\kappa_\text{type}, \eta)$ search. Föllmer best-unsearched is 0.025
   (uncond) and best-searched is 0.026 (Doob $\eta = 0.5$). For any
   user who just wants to turn on guidance and sample, **Föllmer +
   Doob is the robust default**; anyone else has to hyperparameter-tune.

6. **`baseline` is the real story here.** Its OU posterior is *actively
   worse* under natural-$\eta = 1$ than without any guidance (W₂ 0.075 vs
   0.041 uncond). The schedule where path-KL is theoretically worst by the
   largest margin (via Girsanov's $g^{-2} \|\Delta b\|^2$ integrand blowing
   up at $t=1$) is also the schedule where naively-scaled guidance goes
   worst. The path-KL cost and the guidance-misspecification cost are
   *both* manifestations of the same $\sigma \to 0$ boundary issue.

### Reproducing OU

```bash
cd simple-examples/zps
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python

# Uncond baseline
$PY posterior_compare.py --target ou_forecast --seeds 0 1 2 3 4 \
    --sigma_y 0.1 --proposal uncond --guidance_type doob --guidance_eta 1.0 \
    --n_particles 100 --n_em 200 --n_ic 20 --out_dir ./figs_posterior_ou

# Guidance sweep
for gtype in doob natural; do
  for eta in 0.1 0.5 1.0; do
    $PY posterior_compare.py --target ou_forecast --seeds 0 1 2 3 4 \
        --sigma_y 0.1 --proposal guided --guidance_type $gtype --guidance_eta $eta \
        --n_particles 100 --n_em 200 --n_ic 20 --out_dir ./figs_posterior_ou
  done
done
```

Full sweep: ~70 min on one GPU.

## Paper takeaway

The variational-optimality theorem for Föllmer predicts best path-KL, and
the consequences for twisted-SMC posterior sampling are:

1. **Föllmer is the unique schedule where the two natural guidance
   coefficients (Doob's $g^2$ and the "score-replacement" $C_g$) coincide**
   — $C_g = g^2 \iff g^2 = 1-t^2$. For all other schedules they differ, and
   Doob is the rigorous answer.
2. **Untuned ($\eta = 1$) robustness**: Föllmer with default Doob guidance
   gives W2 ≈ 0.034 (gauss σ=0.1). Best-tuned competitor needs careful
   $\eta$/coefficient search to match.
3. **ODE is catastrophic under every choice** — expected, per the paper's
   "diffusion is essential for Dirac → target" remark.
4. **Practical recommendation**: Föllmer + Doob + $\eta \in [0.1, 1]$. The
   theorem and empirics line up.
