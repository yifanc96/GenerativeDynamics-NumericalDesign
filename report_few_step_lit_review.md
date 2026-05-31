# Few-Step Generative Models: A Literature Review

**Scope.** Methods that collapse the many-step sampling of diffusion / flow-matching models down to 1–8 network evaluations (NFEs). Organized by the underlying mathematical object being learned. Focus on the core idea, training objective, and failure modes of each family, with notes on relevance to our Navier–Stokes / Gaussian-field work.

---

## 1. Taxonomy

At a high level there are five families. They differ in *what* the network represents and in *where* the supervision comes from.

| Family | Network represents | Supervision | Prototype |
|---|---|---|---|
| **Instantaneous velocity / score** (slow baseline) | `v(x,t)` or `∇log p_t` | regression against closed-form target | Flow Matching, DDPM |
| **Consistency / trajectory models** | Map from any `(x_t,t)` to a fixed anchor `x_0` | self-consistency along the PF-ODE | CM, iCT, CTM |
| **Flow maps** | Two-time map `Φ(x,s,r)` | Eulerian or Lagrangian consistency between instantaneous and averaged dynamics | FMM, MeanFlow, Shortcut, Align-Your-Flow |
| **Straightening / reflow** | Standard velocity, but on rectified pair couplings | couple `(x_0,x_1)` via previous ODE, then regress | Rectified Flow, InstaFlow |
| **Distributional distillation** | A direct generator `G(z)` | KL / adversarial / moment matching vs teacher | DMD, ADD, SDXL-Turbo, Moment Matching |

Consistency models and flow maps are the theoretically cleanest families; distributional distillation currently dominates large-scale T2I benchmarks. Mean Flow is the first fully-from-scratch method competitive with distillation at 1 NFE.

---

## 2. Consistency Models and Descendants

### 2.1 Consistency Models (CM) — Song, Dhariwal, Chen, Sutskever, 2023
Train `f_θ(x_t,t)` to map any point on a PF-ODE trajectory to its starting point `x_0`, enforcing `f_θ(x_t,t) = f_θ(x_{t'},t')` along trajectories. Two regimes:
- **Consistency Distillation (CD):** uses a pretrained score network to solve the PF-ODE one step and generate `(x_t, x_{t-Δ})` pairs for supervision.
- **Consistency Training (CT):** pure-from-scratch, replacing the PF-ODE step with a noise-injection estimate.

Achieved FID 3.55 on CIFAR-10 / 6.20 on ImageNet 64 at 1 NFE.
Paper: https://arxiv.org/abs/2303.01469

### 2.2 Improved CT (iCT) — Song & Dhariwal, ICLR 2024 (oral)
Diagnosed failure modes of CT:
- **Remove EMA** from the teacher network (EMA hurts rather than helps CT).
- Replace LPIPS with **Pseudo-Huber** loss.
- **Lognormal noise schedule** and a **curriculum doubling the discretization steps** every fixed interval.

Gets CT down to FID 2.51 (CIFAR-10) / 3.25 (ImageNet 64), beating distillation.
Paper: https://arxiv.org/abs/2310.14189

### 2.3 Latent Consistency Models (LCM) — Luo et al., 2023
Applied CD in latent space to Stable Diffusion, reducing T2I to 2–4 steps. Cheap to train (~32 A100-hours). LCM-LoRA made the adapter plug-and-play across SD checkpoints.
Papers: https://arxiv.org/abs/2310.04378, https://arxiv.org/abs/2311.05556

### 2.4 Consistency Trajectory Models (CTM) — Kim et al., ICLR 2024
Generalizes CM by letting the network output `G(x_t, t, s)` = prediction of `x_s` at any target time `s ≤ t`. Closes a key weakness of CM: quality plateaus (or degrades) as you add more NFEs. CTM gets monotone improvement with NFEs and hits FID 1.73 on CIFAR-10. Pre-figures "flow maps" formally.
Paper: https://arxiv.org/abs/2310.02279, code: https://github.com/sony/ctm

---

## 3. Flow Maps

Flow maps unify consistency, CTM, progressive distillation, and shortcut models under one object: the **two-time map** `Φ(x, s, r)` that advances the probability-flow ODE from time `s` to time `r`. Instantaneous velocity is the `r → s` derivative; consistency models are the special case `r = 0`.

### 3.1 Flow Map Matching (FMM) — Boffi, Albergo, Vanden-Eijnden, 2024
Mathematical framework unifying consistency models. Two losses:
- **Lagrangian**: matches the integrated action (like CM).
- **Eulerian**: differentiates `Φ` in one of its time arguments and matches the instantaneous velocity (this is the identity MeanFlow later exploits).

Empirically the Lagrangian loss performs better and gets 10–20× sampling speedup on CIFAR-10 / ImageNet-32 with quality comparable to FM.
Paper: https://arxiv.org/abs/2406.07507

### 3.2 Shortcut Models — Frans, Hafner, Levine, Abbeel, 2024
Pragmatic flow-map parameterization: condition the velocity on the **step size `d`**, so `v_θ(x,t,d)` predicts the average velocity over the next `d` units of time. Trained with a mixed objective — FM loss at `d=0` plus a self-consistency loss `v(x,t,2d) ≈ ½[v(x,t,d) + v(x+d·v(…),t+d,d)]`. One network, one training phase; beats CM and reflow across step budgets.
Paper: https://arxiv.org/abs/2410.12557

### 3.3 Mean Flow — Geng, Deng, Bai, Kolter, He, May 2025
Defines the **average velocity** `u(z,s,r) = (1/(r−s)) ∫_s^r v(z_τ, τ) dτ` as the modeled quantity (not instantaneous velocity). Derives the identity `u(z,s,r) = v(z,s) + (r−s) ∂_s u(z,s,r)` and enforces it via a **JVP-based self-consistency loss**, with a fraction of the batch anchored at `s=r` (pure FM). From-scratch 1-NFE FID 3.43 on ImageNet 256×256 — the first time fully-from-scratch 1-step beats distillation-heavy baselines.
Paper: https://arxiv.org/abs/2505.13447

**Relation to our project.** Our `report_meanflow.md` reproduces this identity as the training objective and confirms its applicability to PDE-valued data (NS vorticity, Gaussian fields). Key non-obvious lessons found in our runs — longer training needed, best-checkpoint tracking, `flow_ratio = 0.5`, uniform time — are consistent with (but not explicit in) the paper.

### 3.4 Align Your Flow (AYF) — Sabour, Fidler et al., NVIDIA, June 2025
Two new **continuous-time** flow-map objectives generalizing CM and FM. Scales flow-map distillation to large T2I models. Uses:
- Autoguidance (a low-quality teacher for guidance signal)
- Optional adversarial fine-tune for a final-NFE quality bump

Published at NeurIPS 2025. Currently the strongest flow-map distillation framework at T2I scale.
Paper: https://arxiv.org/abs/2506.14603, project: https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/

### 3.5 Generalized Flow Maps — 2025
Extensions to Riemannian manifolds and to the generalized-CTM setting (ICLR 2025). Relevant if we ever push multiscale FM onto non-Euclidean state spaces (e.g. spherical vorticity).
Papers: https://arxiv.org/abs/2510.21608

---

## 4. Straightening-Based Approaches

### 4.1 Rectified Flow / Reflow — Liu, Gong, Liu, 2022–23
Train `v_θ` via FM, sample `(x_0, x_1)` couplings under the learned ODE, then **retrain** `v_θ` on the new couplings. Each pass straightens the trajectories. A single Euler step on 2-Rectified Flow gets CIFAR-10 FID 4.85. Conceptually simple but requires multiple training phases and synthetic data generation.
Paper: https://arxiv.org/abs/2209.03003

### 4.2 InstaFlow — Liu et al., ICLR 2024
Applies reflow to Stable Diffusion, then distills to 1 step. First 1-step high-res T2I model from pure reflow + distillation.
Paper: https://arxiv.org/abs/2309.06380

### 4.3 Improving the Training of Rectified Flows — 2024
Diagnoses slow straightening and provides schedule/loss fixes.
Paper: https://arxiv.org/abs/2405.20320

**Note.** "Straightness is not your need" (ICLR 2025) argues that flow-map methods can outperform straightened flows by directly optimizing the integrated map.

---

## 5. Distributional Distillation (Generator-Level)

Rather than enforce trajectory-level identities, these methods train a **direct generator** `G_θ(z)` to minimize a *distributional* divergence against a teacher diffusion model.

### 5.1 Distribution Matching Distillation (DMD / DMD2) — Yin et al., CVPR 2024
Trains two auxiliary score networks: one on the real teacher distribution, one on the generator's current fake distribution. Generator gradient is the score difference (an approximation to `∇ KL(p_fake || p_real)`). Reaches ImageNet FID 2.62 at 1-step; matches SDv1.5 quality at 30× speed.
Paper: https://arxiv.org/abs/2311.18828, project: https://tianweiy.github.io/dmd/

### 5.2 Adversarial Diffusion Distillation (ADD) — Sauer et al., Stability AI, 2023
Score distillation signal + adversarial loss from a GAN-style discriminator. Underlies **SDXL-Turbo**: 1–4 step T2I at full SDXL quality.
Paper: https://arxiv.org/abs/2311.17042

### 5.3 Progressive Distillation — Salimans & Ho, ICLR 2022
The ancestor of all few-step distillation: halve the teacher's sampling steps by training a student that matches the two-step output in one step, then recurse. Distills 8192 → 4 steps.
Papers: https://arxiv.org/abs/2202.00512

### 5.4 Moment Matching Distillation — Salimans et al., NeurIPS 2024
Matches conditional moments between teacher and student. Cheaper and more stable than DMD in some regimes.

### 5.5 Score Identity Distillation (SiD) — 2024
Uses the score-matching identity for an exponentially converging distillation objective with no auxiliary fake-score network. Among the strongest 1-step distillation methods on CIFAR-10 / ImageNet.
Paper: https://arxiv.org/abs/2404.04057

---

## 6. Synthesis: What's Best for Our Setting

The field has converged on a single axis — **what's supervised**:

1. **Point-wise trajectory** (CM, iCT): cheap, but quality saturates fast with NFEs.
2. **Two-time map** (CTM, Shortcut, FMM, MeanFlow, AYF): strictly more general; quality monotone in NFEs; JVP or auxiliary time-derivative is the price.
3. **Distributional** (DMD/ADD/SiD): cares only about sample marginals; best for T2I quality but needs a teacher.

For scientific generative modeling (NS, Gaussian fields, multiscale priors), **the flow-map family is the right abstraction**:
- Clean connection to physical averaging (the map *is* an integrator, just learned).
- From-scratch (no teacher dependency, important when also designing noise priors and multiscale decompositions).
- Composes cleanly with per-band training.

### Best from-scratch (no teacher) methods today:

1. **MeanFlow** (Geng 2025) — what we're using. Cleanest theory, best 1-step from-scratch FID.
2. **Shortcut Models** (Frans 2024) — simpler self-consistency (no JVP), step-size conditioning.
3. **iCT** (Song & Dhariwal 2024) — for true 1-step focus, beats distillation.

### Tricks the SOTA uses that we might be missing:

- **iCT curriculum**: doubling the effective step gap `r−s` over training (we use uniform from start). Similar to our warmup but more principled.
- **Pseudo-Huber loss** instead of MSE / adaptive L2: more robust to outliers, dimensional-scaling correct.
- **Lognormal noise schedule** for the time variable. We tried and it hurt — but maybe with proper tuning of (μ, σ) for our problem.
- **Lagrangian flow-map loss** (FMM): integrates instead of differentiates → no JVP cost. Could be a game-changer for our finest band where JVP doubles memory.
- **Adversarial finetune** (AYF): final pass with a discriminator on residuals. For NS, a discriminator on spectrum slices (low/mid/high-k) could target our specific weakness.

### Known weaknesses of MeanFlow vs alternatives:

- **JVP memory cost**: our biggest practical issue at fine bands. Lagrangian formulation (FMM) avoids this.
- **Quality doesn't always monotonically improve with NFEs**: classical CM/CTM-style failure mode. Our Gaussian-field results show MF1 = MF2 ≈ MF4 (mean_rel ≈ 0.052), suggesting we've hit this plateau.
- **Self-consistency is a moving target**: stop-grad on `w_tgt` makes optimization slow (consistent with our observation that long training is needed).

### Specific recommendations for our setting:

1. **Try Shortcut Models** as an alternative to MeanFlow — same goal (step-size conditioning) but no JVP. Should fix our memory bottleneck at the finest band.
2. **Try Lagrangian flow-map loss** (FMM): avoid JVP entirely for the finest band.
3. **Apply iCT-style curriculum** on the step gap `r−s`: start with small gaps (mostly pure FM), grow to full range. More principled than our binary warmup.
4. **For low-k bottleneck**: consider an adversarial discriminator on the spectrum (operates on FFT features at specific k-ranges). This is a non-pixelwise, distributional residual signal that could specifically attack the few-modes-at-low-k problem.
5. **For finest band**: gradient accumulation to recover effective batch size despite JVP memory cost. Or switch that band to Shortcut Models / Lagrangian FMM.
