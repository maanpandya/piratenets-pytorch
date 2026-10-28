# Handoff (2026-10-28)

The Windows laptop (RTX 3070 Ti Laptop GPU) is now the only workspace. Earlier work was done on a Mac, and there is nothing to send back there.

## Goal
Reproduce the Allen–Cahn ablation from the Expert's Guide (arXiv:2308.08468, **Table 2**) in PyTorch. The target is the relative L2 error of each row; the paper's values are in the table at the end of `windows_run_guide.md` (full pipeline: 5.84e-4). The implementation is independent and MIT-licensed (see `CLAUDE.md`).

## Where things stand
- **Code:** the whole Table 2 pipeline is done. 41 tests pass.
- **Laptop:** set up per `windows_run_guide.md`. That covers the CUDA venv in `.venv`, the tests, the GPU smoke and resume check, the benchmark (which chose the derivative mode), and the launch of `scripts/run_ablation.py`.
- **Results:** none of the full 200k-step rows had been seen when this was written.

## First: find out where the ablation is
1. Finished rows have a `summary.json`: `runs/allen_cahn/ablation/ablation_<row>/summary.json`.
2. Progress is in `runs/ablation_stdout.log`, or `runs/ablation.log` if the user ran it in their own window. Check whether a `python` process is still running.
3. If it stopped before all six rows finished, relaunch exactly as in `windows_run_guide.md` step 7, with the same derivative mode. Finished rows are skipped and the interrupted row resumes. Never pass `--no-resume`.
4. When all six are done, run `scripts/plot_ablation.py` and `scripts/plot_run.py` for each row, then compare `runs/allen_cahn/ablation/ablation_table.md` with the paper.

## If results are off from the paper
- **Look at `training.png` for each row.**
  - `min causal weight` should rise to about 1 by the end. The paper's Fig. 19 shows this for its best run.
  - λ_ic should settle. After the first update λ_ic ≈ 82; that is expected (D13, D20).
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
  - a trainer with CSV logging, and checkpoints whose resume is bit-exact on CPU.
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
- **Compute:** this laptop's GPU. The Colab line in `CLAUDE.md` predates the move.
- **Python:** call the venv interpreter explicitly (`.\.venv\Scripts\python.exe`); each shell command starts a fresh shell.
- **Papers:** `papers/` is gitignored. Download them from arXiv if needed: 2402.00326 (PirateNets) and 2308.08468 (Expert's Guide).
