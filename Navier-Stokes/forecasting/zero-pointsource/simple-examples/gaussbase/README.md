> **Note (2026-04-18):** This Gaussian-base study is **not the setting of
> arXiv:2602.10989**. The paper's Föllmer theorem applies to the **point-source
> (Dirac-initial)** interpolant. See `../zps/` for the correct Section-3.4.1
> implementation. This folder is kept as-is for reference/ablation purposes.

# Gaussian-base ablation (kept for reference)

Companion to the NS forecasting scripts in `../`. Low-dimensional toy targets,
analytic ground truth, and a direct test of the claim (arXiv:2602.10989) that
the Föllmer diffusion coefficient minimizes the **path-space** KL divergence
between the learned and exact SDEs — even though marginal distances may not
distinguish schedules.

## Setup (Gaussian base)

- Interpolant: $I_t = t\,X_1 + (1-t)\,Z$ with $Z \sim \mathcal{N}(0, I_D)$, so
  $I_0 = Z$ (Gaussian base) and $I_1 = X_1$.
- One pair of networks $(v_\theta, s_\theta)$ trained per target via standard
  L2 DSM (velocity) + weighted L2 DSM (score: $\lVert\gamma_t s_\theta + z\rVert^2$).
- At sample time, compose SDE drift $b^{(g)}_t = v_\theta + (g^2/2)\,s_\theta$
  for any diffusion coefficient $g(t)$. No per-schedule training.
- Diffusion schedules normalized to $\int_0^1 g^2\,dt = \epsilon$ (same total
  noise budget). Registry: `const`, `sqrt(1-t)`, `sqrt(t(1-t))`, `sqrt(t)`,
  `1-t`, `0` (ODE).

In this Gaussian-base setting the classical Föllmer diffusion (Brownian motion
as reference) has $g(t) \equiv$ const. In the zero-point-source setting
(delta initial, as in the NS scripts) the Föllmer choice is $g(t) \propto
\sqrt{1-t}$ instead — a follow-up `zps/` subfolder will test that variant
with per-schedule training.

## Targets

- `gaussian1d`: $X_1 \sim \mathcal{N}(1.5, 0.5^2)$, unconditional.
- `bimodal1d`: $X_1 \sim \tfrac12\mathcal{N}(-1, 0.3^2) + \tfrac12\mathcal{N}(1, 0.3^2)$,
  unconditional.
- `ou_forecast`: $X_1 = Y_{s+\tau} - Y_s$ conditional on $Y_s$, where
  $dY = -\lambda Y\,ds + \sigma\,dW$ (simplest dynamical-system forecast
  analogue of the NS scripts).

All three admit analytic $v^\star, s^\star$ via Gaussian-mixture posteriors,
so path-KL is computable without bias.

## Metrics

- **Path KL (Girsanov estimator)** — primary.
  $\widehat{\mathrm{KL}}(g) = \tfrac12\,\mathbb{E}_{t,X_t}\bigl[g^{-2}\lVert b^{(g)}_\theta - b^{(g)\star}\rVert^2\bigr]$.
  With analytic $b^\star$, the MC estimate is unbiased.
- **Marginal $W_2$** — secondary. Reported for completeness; expected (and
  observed) to be nearly flat across schedules.

## Results (5 seeds each, 20k training steps, $N_{\rm EM}=200$, $\epsilon=0.5$)

See `figs/summary_all_targets.{pdf,png}`. Path-KL ordering (lower is better):

| target | best schedule | path KL | 2nd / follmer-$\sqrt{1-t}$ |
|---|---|---|---|
| gaussian1d | `const` | 7.1e-3 ± 3.8e-3 | `triangle` 9.7e-3, `sqrt(1-t)` 13e-3 |
| bimodal1d  | `triangle` | 17.7e-3 ± 5.2e-3 | `const` 20e-3, `sqrt(1-t)` 22e-3 |
| ou_forecast | `const` | 4.6e-3 ± 0.6e-3 | `triangle` 11e-3, `sqrt(1-t)` 12e-3 |

`const` (classical Föllmer for Gaussian base) is the winner or very close
second in all three. `lin_decay` ($g{=}1-t$, so $g\to 0$ as $t\to 1$) is
catastrophically worse (path KL 100–1000× the best).

Marginal $W_2$ is flat across schedules (≈0.015–0.05 for all, error bars
overlapping): consistent with the theorem's prediction that it is a
**path-level**, not a marginal-level, statement.

## Reproducing

```bash
# Train 5 seeds per target (≈80 s each, ~20 min total on a single GPU)
for t in gaussian1d bimodal1d ou_forecast; do
  for s in 0 1 2 3 4; do
    python toy_train.py --target $t --seed $s --max_steps 20000 --use_wandb 0
  done
done

# Aggregate + per-target bar charts
for t in gaussian1d bimodal1d ou_forecast; do
  python compare_schedules.py --target $t --seeds 0 1 2 3 4 --out_dir ./figs
done

# Combined headline figure
python plot_summary.py --out_dir ./figs
```

Enable wandb logging with `--use_wandb 1` (project
`interpolants_follmer_toy`, entity `yifanc96` — adjust in `toy_train.py`).

## Files

| file | purpose |
|---|---|
| `interpolant.py` | `GaussianBaseInterpolant` ($\beta_t=t$, $\gamma_t=1-t$). |
| `schedules.py` | Diffusion-coefficient registry; `compose_drift(v, s, g)`. |
| `targets.py` | Three targets with analytic $v^\star, s^\star$. |
| `networks.py` | Small MLP + Gaussian-Fourier time embedding. |
| `sampler.py` | EM sampler accepting any drift callable. |
| `metrics.py` | Girsanov path-KL estimator + 1D $W_{1,2}$ + MMD. |
| `toy_train.py` | Schedule-agnostic training of $(v_\theta, s_\theta)$. |
| `compare_schedules.py` | Load checkpoint, sweep $g$ at sample time, aggregate. |
| `plot_summary.py` | Combined headline figure. |
