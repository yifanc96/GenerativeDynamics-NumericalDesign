# Source code: forecasting and data-assimilation experiments

Numerical experiments accompanying *Variational Optimality of Föllmer
Processes in Generative Diffusions* (Chen & Vanden-Eijnden). Only the
Python sources, shell launchers, and per-experiment READMEs are included
here — no checkpoints, training logs, or generated figures.

## Contents

```
forecasting_and_data_assimilation_experiments/
├── ou-forecasting/         §5.1   1D OU forecasting + posterior sampling
└── kolmogorov-forecasting/ §5.2   2D Navier–Stokes Kolmogorov flow
```

### `ou-forecasting/` — §5.1

1D Ornstein–Uhlenbeck dynamics with a Dirac point-source interpolant,
$\beta_t = t,\ \sigma_t = 1-t$. Demonstrates the Föllmer-optimality
theorem on both:

- **Marginal forecasting** — train one network, sweep diffusion schedules
  $g(t)$ at sample time via eq. (3.9).
- **Posterior sampling** with a Tweedie-twist guidance term (equivalent
  to the "natural guidance" Föllmer identity $\kappa = g^2$).

Top-level entry points: `train_zps.py` (training),
`compare_zps.py` (marginal sweep), `posterior_compare.py` /
`posterior_smc.py` (posterior sweep), `visualize_posterior.py` (figures).
See `README.md` and `README_posterior.md` inside the folder.

### `kolmogorov-forecasting/` — §5.2

2D Navier–Stokes Kolmogorov flow at $\nu = 10^{-3}$ on a $128^2$ torus.
Same interpolant family, one network per forecast lag.

- **Unconditional forecast** — `train_ns.py`, `rollout_ns.py`,
  `sampler_ns.py`, `compare_ns.py`.
- **Conditional / posterior** via twisted SMC with Tweedie reweighting —
  `observation.py`, `twisted_smc.py`, `posterior_compare.py`.
- **Stochastic ensemble Kalman filter** (vanilla + Gaspari–Cohn
  localised), the comparison baseline used in the paper —
  `enkf.py`.
- **Autoregressive data-assimilation animation** —
  `ar_animation_frames.py`, `render_ar_animation.py`,
  `make_da_figure.py`, plus the launchers `run_ar_animation.sh` and
  `run_enkf_animation.sh`.
- **Data generation** — `simulate.py` (calls `torch-cfd`),
  `verify_data.py`.

See `README.md`, `README_posterior.md`, and `README_posterior_sqlin.md`
inside the folder.

## Reproduction

Each folder is self-contained. Training and inference scripts expect to
be run from inside their own folder so relative `runs/`, `logs/`, and
`figs/` paths resolve. Inspect the in-folder README files for exact
commands.
