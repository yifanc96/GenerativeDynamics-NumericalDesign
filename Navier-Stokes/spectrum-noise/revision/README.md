# Revision experiments — SISC #M179447

All code, checkpoints, figures, and results produced during the SIAM SISC revision
of *"Scale-Adaptive Generative Flows for Multiscale Scientific Data"* (Chen &
Vanden-Eijnden). Cross-references below point to sections of `ex_article_revise.tex`
and the reviewer comments addressed in `rebuttal.tex`.

All scripts assume `HOME = os.path.dirname(os.path.abspath(__file__))` and read
data / write outputs in this folder, so the layout is intentionally flat.

## Scripts

### Architecture & shared utilities
- `unet.py` — UNet used by all NS experiments (dim=32, mults=(1,2,2,2), ~2M params).

### Gaussian closed-form experiments (Sec. 4.1, 5.1; Reviewer 1 Q2, Reviewer 2 B1, B4)
- `exp_gaussian_revision.py` — full Gaussian sweep (smoothness, schedule, NFE) using
  the analytical drift `B(t) = (αα'C0 + ββ'C1)(α²C0 + β²C1)⁻¹`. Produces
  `gaussian_results.txt`.
- `exp_gaussian_smoothness_sweep.py` — sweeps `s_0 ∈ {0,1,2,3}` at fixed `s_1=3`
  to validate the Lipschitz scaling `(τ²+4π²|m|²)^((s_1−s_0)/2)`. Produces
  `gaussian_smoothness_sweep_results.txt`.
- `exp_gaussian_figure.py` — replot from saved results: smoothness-sweep curve and
  wavenumber-dependent vs scalar schedule comparison
  (`gaussian_smoothness_sweep.pdf`, `gaussian_wavenumber_vs_scalar.pdf`).
- `exp_lipschitz_validation.py` — empirical drift Lipschitz `‖B(t)‖_2` vs `t` for
  four prior/schedule combinations (linear/white, linear/matched, designed/white,
  linear/smoother `s_0=5`). Validates Proposition 3.3 and the
  `½|log λ*|` bound from Proposition 5.1. Produces `lipschitz_validation_gaussian.pdf`.
- `exp_lip_resolution_scaling.py` — Lipschitz vs grid `N ∈ {32, 64, 128, 256, 512}`
  showing the bound does not blow up with resolution. Produces `lip_resolution_scaling.pdf`.

### Cameron–Martin norm validation (App. F.A; Reviewer 2 B4)
- `exp_cameron_martin_validation.py` — computes `‖x_1‖_V²` for Gaussian / Allen–Cahn /
  Navier–Stokes data at multiple resolutions under matched/rougher/smoother priors.
  Separates the three regimes (stable / linear / quadratic growth in mode count).

### Navier–Stokes spectrum-noise training (Sec. 4.2.3, 5.2; Reviewer 1 Q3, Reviewer 2 M2a)
- `train_ns_spectrum_noise.py` — UNet training with three noise variants:
  `matched` (per-mode empirical std), `mulk` (rougher: matched × |k|), and
  white-noise baseline. Saves checkpoints + loss curve + final evaluation table.
  This produced `ns_spectrum_matched_step50000_hi128.pt` (50k steps, ~4h on H200,
  final loss 0.98) used by `eval_matched_multiseed.py`.
- `eval_matched_multiseed.py` — re-evaluates the trained matched-spectrum UNet at
  3 seeds × 200 samples × {10, 20, 50} RK4 steps × {linear, designed} for
  error-barred Table 5.4. Uses the auto-`λ*` rule `λ* ≈ S_truth/S_noise` at the
  Nyquist band. Produces `ns_spectrum_matched_results_50k_3seeds.txt`.

### Navier–Stokes drift Lipschitz validation (App. F.B; Reviewer 2 B4)
- `exp_lip_ns.py` — empirical Lipschitz of the *trained* NS drift along sampling
  trajectories under designed/white setup.
- `exp_lip_v_norm_ns.py` — V-norm of NS samples across resolutions (subsamples
  native 256 down to 64 and 32 for the multi-resolution table).

### Reproducing previously published figures
- `reproduce_paper_fig2.py`, `reproduce_paper_figures.py` — regenerates Fig 2
  (Gaussian), Fig 3 (Allen–Cahn), Fig 5 (NS spectrum comparison), Fig 6
  (NS white + designed) from saved spectra. Used to sanity-check that the new
  code reproduces the original-paper numbers before adding revisions on top.

### Pre-existing background code (from `main-Lip` / original paper, pre-2026-04-17)
These scripts were authored before the revision cycle but are cited or directly
relevant to the revised manuscript. Copied here so `revision/` is self-contained.

- `auto_lip_ns.py` — autograd-based empirical Lipschitz of the trained NS drift.
- `lipschitz_check.py`, `lipschitz_perband.py` — pointwise and per-band
  Lipschitz diagnostics used in App. F.B.
- `eval_lip_transfer.py` — designed-schedule inference transfer formula
  evaluation at 64/128/256 (App. F.B background and Sec. 5.2).
- `eval_nongaussian.py`, `eval_nongaussian_v2.py` — flatness / kurtosis / KS
  diagnostics on generated vs. ground-truth NS ensembles; feeds Table 5.5.
- `train_ns_lip_compare.py` — training driver for the white-noise + designed
  schedule comparison reused from the `main-Lip` paper.
- `reproduce_lip_ns.py`, `reproduce_lip_ns_compare.py` — regenerate the
  `main-Lip` NS results (the same NS example as Sec. 5.2).
- `make_perband_figure.py` — per-band error plotting helper.

Companion PNG outputs from these scripts: `ns_lip_*.png`, `ns_nongaussian_*.png`,
`ns_spectrum_eval_all.png`.

## Outputs

### Checkpoints
| file | content |
|---|---|
| `ns_spectrum_matched_step50000_hi128.pt` | matched-spectrum UNet, 50k steps (Reviewer 1 Q3 retraining) |
| `ns_spectrum_mulk_step5000_hi128.pt` | rougher (mul-k) UNet, 5k steps |
| `ns_spectrum_noise_step{200,5000}_hi128.pt` | matched-spectrum UNet checkpoints (intermediate) |

### Data
- `enstrohpy_spectrum_amplitude.pt` — per-mode empirical std for matched / mul-k
  spectrum noise (loaded by training and eval scripts).

### Results
- `*_results*.txt` — eval tables (steps × schedule → mid/high band errors).
- `*_loss_*.npy` — training loss curves.

### Figures (PDF, all readable at the sizes used in the paper)
- Gaussian: `gaussian_smoothness_sweep.pdf`, `gaussian_wavenumber_vs_scalar.pdf`,
  `lipschitz_validation_gaussian.pdf`, `lip_resolution_scaling.pdf`.
- NS designed-vs-linear sweeps: `ns_spectrum*_des_vs_lin_steps{10,20,50}_step*.pdf`.
- NS Lipschitz: `lip_validation_ns.pdf`.
- Reproductions: `reproduced_paper_fig{2,3,5,6}_*.pdf`.

### Logs
- `ns_spec_train.log`, `ns_spec_matched_continue.log`, `ns_spec_mulk_train.log`,
  `gaussian_run.log` — stdout from long training/eval runs.

## How to rerun

A representative end-to-end example for Reviewer 1 Q3 (matched-spectrum NS + designed schedule):

```bash
# 1) Train the matched-spectrum UNet (uses GenerativeDynamics-NumericalDesign/NSdata)
python train_ns_spectrum_noise.py --noise matched --max_steps 50000

# 2) Multi-seed error-barred evaluation (3 seeds × 200 samples × {10,20,50} steps × {linear,designed})
python eval_matched_multiseed.py
# → ns_spectrum_matched_results_50k_3seeds.txt
```

Data paths inside the scripts point to
`/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file{,02,03,04,05}.pt`;
edit `load_data()` in the relevant script if running elsewhere.
