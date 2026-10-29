# Handoff (updated 2026-10-29)

The Windows laptop (RTX 3070 Ti Laptop GPU) is now the only workspace. Earlier work was done on a Mac, and there is nothing to send back there.

## Goal
Reproduce the Allen–Cahn ablation from the Expert's Guide (arXiv:2308.08468, **Table 2**) in PyTorch. The target is the relative L2 error of each row; the paper's values are in the table at the end of `windows_run_guide.md` (full pipeline: 5.84e-4). The implementation is independent and MIT-licensed (see `CLAUDE.md`).

## Where things stand
- **Code:** the whole Table 2 pipeline is done. 43 tests pass.
- **Laptop:** set up per `windows_run_guide.md`: Python 3.12, torch 2.14.1+cu132, a CUDA venv in `.venv`. The first GPU run found two platform problems, both fixed:
  - resume failed on CUDA because the RNG states were loaded onto the GPU (`c079158`);
  - Windows background power throttling (EcoQoS) slowed minimized runs 2–3× once the display turned off. `Trainer` now opts its own process out (`53a52b5`).
- **Derivative mode:** `--set training.derivatives=autograd`. It measured 11.1 ms/step against 21.4 for `func`, and gives the same values up to round-off.
- **Results:** the full six-row ablation ran once, with seed 0 (2026-10-27). See below.

## Seed-0 results
| Row | Ours | Paper | Ours / paper | Wall time (min) |
|---|--:|--:|--:|--:|
| default | 3.39e-3 | 5.84e-4 | 5.8 | 91.9 (about 50 without throttling) |
| no_fourier | 5.33e-2 | 4.35e-1 | 0.12 | 43.9 |
| no_rwf | 1.39e-2 | 6.62e-3 | 2.1 | 52.1 |
| no_grad_norm | 7.28e-4 | 7.51e-3 | 0.10 | 52.0 |
| no_causal | 8.63e-3 | 1.59e-3 | 5.4 | 52.0 |
| plain | 5.25e-1 | 5.17e-1 | 1.0 | 39.5 |

- **The main open question:** `no_grad_norm` beats `default` by 4.6×, the reverse of the paper.
  - In `default`, λ_ic stays around 20–50 and λ_r around 0.5. The IC loss falls to about 1e-10, while the residual loss plateaus near 1e-7 from about 120k steps.
  - In `no_grad_norm`, both losses settle near 1–5e-8.
- `no_fourier` does about 8× better than the paper. `plain` matches it.
- In every row with causal weighting, the minimum causal weight rises to 1 and stays there (≥ 0.99). That happens from about 34–40k steps in `default`, `no_rwf` and `no_grad_norm`, and from 67k in `no_fourier`.
- **Outputs:** in `runs/allen_cahn/ablation/` (gitignored). That holds `ablation_table.md`, `ablation_rel_l2.png`, `run_notes.md`, and per row `metrics.csv`, `training.png`, `solution.png` and the checkpoints. `ablation_results.zip` in the repo root packs the same outputs without checkpoints.
- **Re-running:** to re-run or extend a row, use `windows_run_guide.md` step 7. Finished rows are skipped and an interrupted row resumes. Never pass `--no-resume` on `runs/allen_cahn/ablation/`.

## If results are off from the paper (next steps)
- **Look at `training.png` for each row.**
  - `min causal weight` should rise to about 1 by the end. The paper's Fig. 19 shows this for its best run.
  - λ_ic should settle. After the first update, D13 and D20 record λ_ic ≈ 82 on CPU; on CUDA it is 73.9.
- **Settings that are our own choices or inferences.** Check these first, in `docs/decisions.md`:
  - D14, the reading of the plain row;
  - D17, our reference grid;
  - D18, seeds. A single seed can be noisy, so consider 2–3 seeds per row: `--set training.seed=N` with a separate `--out-root`.
- **Settings taken from jaxpi** (D2–D13, D20) are already matched to the code that produced the paper. Change them only with evidence, and record any change in `docs/decisions.md`.

## What exists
- **`piratenets/`:**
  - the Allen–Cahn residual, with derivatives via `torch.func` (an `autograd` path is also available);
  - exact periodic embedding, trainable random Fourier features, random weight factorization (RWF) layers, and a plain MLP;
  - causal weighting and grad-norm loss weights;
  - a trainer with CSV logging, and checkpoints whose resume is bit-exact on CPU and on CUDA (both are tested).
- **`configs/allen_cahn/ablation/`:** one YAML per Table 2 row. Each differs from `default.yaml` only in its switch, and a test checks that.
- **`scripts/`:** `train.py`, `run_ablation.py`, `benchmark_step.py`, `plot_ablation.py`, `plot_run.py`, `pack_results.py`.
- **`docs/decisions.md`:** every detail the papers leave open (D1–D20), with its source. Read it before changing any setting. `docs/spec.md` describes what is implemented.
- **`notebooks/colab_ablation.ipynb`:** a Colab fallback, no longer needed.
- **Not implemented:** Modified MLP, PirateNet with physics-informed init, NTK weighting, SOAP.

## Settings taken from jaxpi rather than a plain reading of the paper
- 200k steps.
- Fourier features: scale σ = 1, 256 outputs, B trained.
- RWF μ = 0.5.
- Truncated-normal Glorot initialization.
- Causal chunks formed by sorting the sampled times into 32 equal-count groups.
- IC loss on the fixed 512-point reference x-grid.
- Grad-norm weights updated *after* the optimizer step, using the *mean* of the norms.

## Working conventions
- **jaxpi:** values and design decisions may come from jaxpi, but never copy its code. Where the paper and jaxpi disagree, default to jaxpi, implemented independently, and keep the paper's version as a config option. Record each such choice in `docs/decisions.md`.
- **Configs:** every component is a config switch, and each experiment is a YAML file.
- **Plots:** they read the CSV logs and never re-run training.
- **Compute:** this laptop's GPU. Launch long runs as a detached process (`windows_run_guide.md` step 7). They take about 40–50 min per 200k-step row.
- **Python:** call the venv interpreter explicitly (`.\.venv\Scripts\python.exe`); each shell command starts a fresh shell.
- **Papers:** `papers/` is gitignored. Download them from arXiv if needed: 2402.00326 (PirateNets) and 2308.08468 (Expert's Guide).
