# Decisions on details the papers leave open

This file records every place where the papers do not pin down an implementation
detail, and what this reimplementation does instead. Sources:

- **EG**: Wang, Sankaran, Wang & Perdikaris, *An Expert's Guide to Training
  Physics-informed Neural Networks*, arXiv:2308.08468 (v1).
- **PN**: Wang, Li, Chen & Perdikaris, *PirateNets: Physics-informed Deep Learning
  with Residual Adaptive Networks*, JMLR 25 (2024), arXiv:2402.00326 (v3).
- **JC**: the jaxpi example configs, `examples/allen_cahn/configs/*.py` in
  PredictiveIntelligenceLab/jaxpi. There is one file per ablation row
  (`default`, `no_fourier_feature`, `no_rwf`, `no_grad_norm`, `no_causal`,
  `plain`), plus `sota.py` for the best run.
  - The files were added in commit `951e49cc` on 2023-08-06, ten days before EG v1.
  - The only later change is a rename, `layer_size` → `hidden_dim` (`1842becd`, 2023-11-20). The values are as used for the paper.
- **JL**: jaxpi's code, read at `main` (2025-09):
  - library: `jaxpi/archs.py`, `models.py`, `samplers.py`;
  - example: `examples/allen_cahn/models.py`, `train.py`, `utils.py`.

  We consulted it only to settle the design decisions below that neither the papers nor JC answer. No jaxpi code is copied here, and the rest of this implementation follows the papers.

Status key:
- **stated**: the paper says it. It is recorded here only because a different section of the paper says something else.
- **from JC**: the paper does not say it, and the value comes from the jaxpi configs.
- **from JL**: the paper and configs do not say it, and the design decision comes from jaxpi's code.
- **provisional**: no source says it, so this is our own choice.
- **open**: not decided yet.

Every value below is a config switch (`piratenets/config.py`), so changing a
decision means editing YAML, not code.

## Allen–Cahn ablation (EG Table 2)

### D1. Backbone network: stated
- **Question:** Table 6 (the best Allen–Cahn run) uses a Modified MLP. Does the ablation use it too?
- **Decision:** No. The ablation uses a plain MLP with 4 hidden layers of 256 tanh units, initialized with Glorot.
- **Source:**
  - EG Sec. 7, paragraph before Table 1: "we employ an MLP with 4 hidden layers, 256 neurons per hidden layer, and tanh". EG also says "the hyper-parameter settings for our ablation studies differ from those yielding the best results".
  - JC `default.py` agrees: `arch_name = "Mlp"`, `num_layers = 4`, `hidden_dim = 256`, `activation = "tanh"`.

### D2. Number of training iterations: from JC
- **Question:** How many steps does each ablation run take? EG Sec. 7 says only that "the total number of training iterations can vary".
- **Decision:** 200,000 (`training.steps`).
- **Source:** JC `default.py`, `max_steps = 200000`, the same in all six row configs. An earlier provisional value of 150,000, read off the x-axis of EG Fig. 4, was wrong.

### D3. Fourier feature scale σ: from JC
- **Question:** What scale σ do the ablation runs use?
- **Decision:** σ = 1.0 (`model.fourier.scale`).
- **Source:** JC `default.py`, `embed_scale = 1`. An earlier provisional value of 2.0 is the value for the best run (EG Table 6, PN Table 4, JC `sota.py`), not the ablation.

### D4. Number of Fourier features: from JC and JL
- **Question:** How many Fourier features?
- **Decision:** B ∈ R^{128×3}, so the embedding has 256 dimensions (`model.fourier.embed_dim: 256`).
- **Source:** JC sets `embed_dim = 256`. JL shows that this is the total output size: there are `embed_dim // 2` frequencies, each giving a cosine and a sine.

### D5. Order of the embeddings, and whether B is trained: from JL
- **Question:** Are the random Fourier features applied to (t, x) or to the periodic embedding? Is B trained?
- **Decision:**
  - The periodic embedding comes first, then the random Fourier features act on v = [t, cos πx, sin πx].
  - B is initialized from N(0, σ²) and is **trained** with the rest of the network (`model.fourier.trainable: true`). `false` keeps B fixed.
- **Source:**
  - EG Sec. 6.3 builds the network as u_θ([t, v(x)]), but EG never says whether B is trained.
  - In JL the Fourier embedding is applied to the output of the periodic embedding.
  - In JL, B is a regular model parameter, and the optimizer updates all parameters.
  - An earlier provisional choice kept B fixed.

### D6. RWF μ and σ: from JC
- **Question:** Which (μ, σ) do the ablation runs use? EG gives three answers: Sec. 4.3 says "μ = 0.5 or 1", Algorithm 2 says μ = 1.0, and Table 6 says μ = 0.5.
- **Decision:** μ = 0.5, σ = 0.1 (`model.rwf.mu`, `model.rwf.sigma`).
- **Source:** JC `default.py`, `reparam = {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}`. An earlier provisional value of 1.0 was wrong. For the best run, see D19.

### D7. Which layers RWF applies to: from JL
- **Question:** The loop in Algorithm 2 runs over l = 1, …, L. Eq. 4.5 and the training step of Algorithm 2 cover l = 1, …, L+1.
- **Decision:** RWF applies to every dense layer, including the output layer, with one scale per output neuron. It does not apply to the Fourier matrix B.
- **Source:** EG eqs. 4.4–4.5. JL factorizes the output layer as well.

### D8. Glorot initialization: from JL
- **Question:** Is the Glorot initialization normal or uniform?
- **Decision:** A normal distribution truncated at ±2 standard deviations, rescaled so that Var(W) = 2/(fan_in + fan_out). Biases start at zero.
- **Source:**
  - PN Sec. 3 gives only "W ~ D(0, 2/(d_l + d_{l−1}))" without naming a distribution.
  - JL uses JAX's `glorot_normal` initializer, which samples from that truncated normal (a property of JAX, documented by JAX).
  - An earlier provisional choice used an untruncated normal.

### D9. Learning-rate schedule and optimizer: from JC and JL
- **Decision:**
  - Adam with β = (0.9, 0.999), ε = 1e-8, and no weight decay.
  - lr(step) = 1e-3 · 0.9^(step / 2000), decayed continuously, not in steps (`optim.staircase: false`).
  - No warmup.
- **Source:**
  - JC gives the optimizer and decay values.
  - JL uses optax's `exponential_decay` with its default (continuous) setting and no warmup.
  - EG Sec. 6.1 advises against weight decay.
  - The 5,000-step warmup from PN Sec. 5 applies to the PirateNet runs only.

### D10. Sampling collocation points and forming causal chunks: from JL
- **Question:** How are the residual points sampled, and how are they split into causal chunks?
- **Decision:**
  - Each step draws 4,096 fresh points uniformly in (t, x).
  - The points are sorted by t and split into M groups of equal size. Chunk i is the i-th group of the earliest points, so its time boundaries fall at the sample quantiles of t rather than at fixed segment edges.
  - The rows with causal weighting off use the same sampler.
- **Source:**
  - EG Algorithm 1 step 5(a) and Sec. 7 say the points are sampled uniformly. EG never says how the chunks are formed.
  - JL sorts the sampled times and reshapes the residuals into `num_chunks` rows.
  - An earlier provisional choice sampled the same number of points inside each of M fixed time segments.

### D11. Initial-condition points: from JL
- **Question:** Which points are used for the initial-condition loss?
- **Decision:** A fixed set: the 512 x-points of the reference grid at t = 0, at every step (`training.ic_sampling: grid`). `random` draws `n_ic` fresh points per step instead.
- **Source:**
  - EG Algorithm 1 step 5(a) says the IC points are also sampled randomly, but never gives how many.
  - JL evaluates the IC loss on the x-grid of the reference data.
  - An earlier provisional choice used 4,096 random points per step.

### D12. Number of causal chunks M: from JC
- **Decision:** M = 32 (`causal.num_chunks`).
- **Source:** JC `default.py`, `num_chunks = 32`. This is the same value as for the best run (EG Table 6, PN Table 4).

### D13. Details of the grad-norm weighting: from JC and JL
- **Decision:**
  - There are two loss terms, IC and residual. There is no BC term, because the periodic condition is exact.
  - ‖∇_θ L‖ is the L2 norm over all trainable parameters, concatenated, including B.
  - The norms are taken of the unweighted L_ic and of the causally weighted L_r from eq. 2.10.
  - λ starts at 1 for both terms, with momentum α = 0.9.
  - On steps 0, 1000, 2000, … the weights are recomputed *after* that step's optimizer update, on the same batch, at the new parameters. So the step-0 update first applies at step 1.
  - λ is a constant for back-propagation.
  - Algorithm 1's default list includes "γ = 1.0", which matches no symbol in the algorithm, so we ignore it.
- **Source:**
  - EG Algorithm 1 steps 4 and 5(c)–(d), and eqs. 2.12–2.15.
  - JC confirms `init_weights = {ics: 1.0, res: 1.0}`, `momentum = 0.9` and `update_every_steps = 1000`.
  - JL gives the update timing and shows that the norms use the causally weighted residual loss.
  - An earlier provisional choice updated the weights before the optimizer step.
  - JL also normalizes the norms differently from the paper; see D20.
- **Consequence of the update timing:**
  - Adam's first step moves every parameter by about lr in the direction of its gradient's sign, so the losses jump briefly. With seed 0 on CPU, L_ic goes 0.07 → 23 and L_r goes 0.06 → 1.3e4. They recover within a few hundred steps (L_ic ≈ 7e-5 by step 250).
  - The step-0 update is taken at that jump, so λ̂_ic ≈ 1.6e3 (paper's sum) or ≈ 8e2 (mean, D20). With the mean, λ_ic = 82.5 from step 1.
  - Updating before the step, as we did earlier, gives λ_ic ≈ 1.0. With that choice λ_ic only reaches its later level after tens of thousands of steps.
  - So the timing mostly affects the first ~20,000 steps of the 200,000.

### D14. Typo in the plain-PINN row of Table 2: provisional
- **Question:** EG Table 2 prints the plain row as "51.74 × 10^−1", which would be greater than 1 and does not fit the paper's text.
- **Decision:** Read it as 5.174 × 10^−1.
- **Reason:** This is close to the 4.98 × 10^−1 that PN Table 2 gives for the original PINN formulation, and to the no-Fourier row (4.35 × 10^−1).

### D15. Causal weights: stated
- **Decision:**
  - L_r^i is the mean squared residual over the points in chunk i.
  - w_1 = 1 and w_i = exp(−ε Σ_{k<i} L_r^k), with ε = 1.0.
  - L_r = (1/M) Σ_i w_i L_r^i.
  - The weights are detached (no gradient flows through them).
  - With causal weighting off, w = 1, so L_r is the plain mean.
- **Source:** EG Sec. 5.1, eqs. 2.10–2.11 and 5.2, and Algorithm 1 defaults. JC sets `causal_tol = 1.0`, and JL computes the same quantities.

### D16. Periodic embedding in every row: stated
- **Decision:** All six rows, including "Plain", use the exact periodic embedding x ↦ (cos πx, sin πx), a single harmonic with ω = 2π/2. The time t enters unscaled, since it already lies in [0, 1].
- **Source:**
  - EG Sec. 7: "Throughout all numerical experiments, when applicable, we enforce the exact periodic boundary conditions". EG Sec. 6.3, eq. 6.2.
  - JC `plain.py` keeps the periodic embedding with `period = π`, and JL uses that value as the angular frequency in cos(π x).

### D17. Error metric and reference grid: stated in JL; grid provisional
- **Decision:** ‖u_θ − u_ref‖₂ / ‖u_ref‖₂ over the full reference grid. JL computes the error the same way.
- **Our grid:** 201 × 512 points (t = 0, 0.005, …, 1; x_j = −1 + 2j/512). The reference solution is ETDRK4 Fourier pseudo-spectral and self-converged to about 1e-6 (`data/generate_allen_cahn.py`).
- **Paper's grid:** PN Appendix B uses a 200 × 512 Chebfun/ETDRK4 dataset.

### D18. Seeds: provisional
- **Question:** How many seeds per row, and which?
- **Decision:** Seed 0, one run per row. The model initialization uses torch's global RNG seeded with `training.seed`. The collocation sampler has its own generator, seeded with `training.seed + 1`.
- **Source:** JC uses `seed = 42`, but JAX's PRNG and PyTorch's differ, so using the same number would not reproduce their initialization. EG does not say whether Table 2 averages over seeds, while PN's ablations average 5. We will add seeds if single runs turn out too noisy to compare with the paper.

### D20. Normalization in the grad-norm weights: from JL (owner's decision)
- **The difference:**
  - EG eqs. 2.12–2.14 use the *sum* of the gradient norms: λ̂_k = (Σ_j ‖∇L_j‖) / ‖∇L_k‖.
  - JL uses the *mean*: λ̂_k = (mean_j ‖∇L_j‖) / ‖∇L_k‖.
  - With two loss terms, JL's λ̂ are half the paper's, while the ratio λ̂_ic / λ̂_r is the same.
- **Decision:** The mean, as in JL (`weighting.grad_norm_reduction: mean`). `sum` gives the formula as printed in the paper.
  - The project owner chose this because the goal is to reproduce jaxpi's results.
  - The rule we follow: where the paper and jaxpi's code disagree, default to jaxpi's behavior, implemented independently, and keep the paper's version as an option.
- **Effect:**
  - Because the moving average starts from λ = 1, the ratio λ_ic / λ_r differs between the two versions until that initial value has decayed (by 0.9^k after k updates, about 20 updates or 20,000 steps).
  - After the step-0 update (D13), with seed 0 on CPU: mean gives λ_ic = 82.5 and λ_r = 0.95, a ratio of 87. Sum gives λ_ic = 164 and λ_r = 1.0, a ratio of 164.
  - Adam is nearly invariant to a common scale on the loss, so only this ratio matters.
  - Later in training the two versions agree.

## Best Allen–Cahn run (EG Table 6), for later

### D19. RWF μ for the best run: from JC (by the D20 rule)
- **The difference:** EG Table 6 prints RWF μ = 0.5, but JC `sota.py` sets `mean = 1.0`.
- **Agreement:** Every other value in `sota.py` matches Table 6: ModifiedMlp, 4 × 256, tanh, `embed_scale = 2`, decay steps 5,000, 300,000 steps, batch 8,192, NTK weighting, causal tolerance 1.0, 32 chunks.
- **Decision:** When that run is set up, follow `sota.py` (μ = 1.0), per the rule in D20.
