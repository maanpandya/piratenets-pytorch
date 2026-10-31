# Specification

What this repository implements. The method follows the two papers, cited as
**EG** (Expert's Guide, arXiv:2308.08468) and **PN** (PirateNets, arXiv:2402.00326).
`docs/decisions.md` lists the details the papers leave open (D1–D21) and where each
answer comes from: the papers, jaxpi's example configs, jaxpi's code (consulted for
design decisions only, nothing copied), or our own choice.

## Problem: 1D Allen–Cahn (EG Sec. 7.1, PN Sec. 3)

    u_t − 1e-4 u_xx + 5u³ − 5u = 0,   t ∈ [0, 1],  x ∈ [−1, 1],
    u(0, x) = x² cos(πx),   u(t, −1) = u(t, 1),   u_x(t, −1) = u_x(t, 1).

The reference solution is `data/allen_cahn_ref.npz`: 201 × 512 values on
t = 0, 0.005, …, 1 and x_j = −1 + 2j/512. It is computed by ETDRK4 Fourier
pseudo-spectral time stepping at N = 4096 and self-converged to about 1e-6.
The metric is the relative L2 error ‖u_θ − u_ref‖₂ / ‖u_ref‖₂ over that grid.

## Model (`piratenets/embeddings.py`, `layers.py`, `models.py`)

The model maps (t, x) to u, built from these pieces:

1. **Periodic embedding** (EG Sec. 6.3): v = [t, cos(πx), sin(πx)]. With it, every x-derivative is exactly periodic, so the loss has no boundary term.
2. **Random Fourier features** (EG Sec. 4.2), optional: γ(v) = [cos(Bv), sin(Bv)], with B ∈ R^{128×3} initialized as B_ij ~ N(0, σ²), σ = 1, and trained with the network (D5).
3. **MLP** (EG eqs. 4.1–4.2): L hidden layers of width 256 with tanh activations, followed by a linear output layer. Weights start from a Glorot normal truncated at ±2 standard deviations (D8), and biases start at zero.
4. **Random weight factorization** (EG Sec. 4.3, Algorithm 2), optional, on every dense layer: W = diag(exp(s)) V. It is initialized from the Glorot W with s ~ N(μ, σ²) and V = W / exp(s). The ablation uses μ = 0.5, σ = 0.1. By default (`model.rwf.scale_param: direct`), training updates (g, V, b) with g = exp(s), as jaxpi does. With `log`, it updates (s, V, b), as in EG Algorithm 2 (`docs/decisions.md`, D21).

## Losses (EG eqs. 2.6, 2.8–2.11)

At each step the trainer samples N_r = 4096 fresh residual points (t, x), uniformly.
It sorts them by t and splits them into M = 32 equal-size chunks (D10).
The initial-condition loss always uses the same points: the 512 x-values of the reference grid (D11).

The losses are:

    L_ic  = mean_i (u_θ(0, x_i) − x_i² cos(π x_i))²
    R     = u_t − 1e-4 u_xx + 5u³ − 5u                    (derivatives via torch.func)
    L_r^k = mean over chunk k of R²,  k = 1..M
    w_k   = exp(−ε Σ_{j<k} L_r^j)   (stop-gradient; w ≡ 1 if causal weighting is off)
    L_r   = (1/M) Σ_k w_k L_r^k
    L     = λ_ic L_ic + λ_r L_r

The derivatives are computed per point under `torch.func.vmap`:
- (u_t, u_x) come from `grad` (reverse mode);
- u_xx comes from `jvp` of that gradient along x (forward over reverse).

A plain `torch.autograd.grad` path gives the same values. It can be selected with `training.derivatives: autograd`.

## Loss weights (EG Algorithm 1, Sec. 5.2)

All λ start at 1. With the grad-norm scheme, on steps 0, 1000, 2000, … the trainer recomputes the
losses after the optimizer step, on the same batch, and computes (D13)

    λ̂_k = mean(‖∇_θ L_ic‖, ‖∇_θ L_r‖) / ‖∇_θ L_k‖,     λ_k ← 0.9 λ_k + 0.1 λ̂_k,

and treats λ as a constant for back-propagation. The mean follows jaxpi (D20). EG eqs. 2.12–2.14 print
the sum instead, available as `weighting.grad_norm_reduction: sum`.

## Optimization (EG Sec. 6.1, Sec. 7)

The optimizer is Adam at lr 1e-3, decayed as lr · 0.9^(step/2000), with no weight decay. The ablation runs 200,000 steps.

## Ablation rows (EG Table 2; `configs/allen_cahn/ablation/`)

| Config | Fourier | RWF | Grad norm | Causal | Paper rel. L2 |
|---|:-:|:-:|:-:|:-:|--:|
| `default.yaml` | yes | yes | yes | yes | 5.84e-4 |
| `no_fourier.yaml` | no | yes | yes | yes | 4.35e-1 |
| `no_rwf.yaml` | yes | no | yes | yes | 6.62e-3 |
| `no_grad_norm.yaml` | yes | yes | no | yes | 7.51e-3 |
| `no_causal.yaml` | yes | yes | yes | no | 1.59e-3 |
| `plain.yaml` | no | no | no | no | ~5.2e-1 |

`tests/test_config.py` checks that each row differs from `default.yaml` only in its switches.

## Engineering requirements

- **float32 with full-precision matmuls.** `set_float32_matmul_precision("highest")` is set and TF32 is off.
- **Logging.** Every 100 steps a row goes to `<out_dir>/metrics.csv` with:
  - step and wall time;
  - learning rate;
  - each loss;
  - λ_ic and λ_r;
  - the gradient norms, on steps where the weights are updated;
  - the minimum and mean causal weight.

  The relative L2 error is added every 1000 steps.
- **Checkpointing.** Every 5000 steps the trainer saves `<checkpoint_dir>/last.pt`, written atomically. It holds the model, optimizer, scheduler, loss weights, all RNG states (including the sampler's generator), the step count, and the wall time.
- **Resume.** A resumed run continues exactly where the checkpoint left off and rewrites any CSV rows logged after it. `tests/test_trainer.py` checks that the result matches an uninterrupted run bit for bit on CPU.
- **Outputs.** At the end of a run the trainer writes `prediction.npz` (u_θ on the reference grid) and `summary.json`. The plotting scripts read only these files and the CSVs.

## Not implemented yet

Modified MLP (EG Sec. 6.4), PirateNet and physics-informed initialization (PN Sec. 4), NTK weighting (EG eqs. 5.4–5.9), SOAP.
