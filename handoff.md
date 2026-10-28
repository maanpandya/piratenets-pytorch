# Handoff (2026-10-28)

## Goal
Reproduce the Allen–Cahn ablation from the Expert's Guide (arXiv:2308.08468, **Table 2**) in PyTorch. The target is the relative L2 error of each row: the full pipeline is 5.84e-4, and the paper lists the other rows in `windows_run_guide.md`. The implementation is independent and MIT-licensed (see `CLAUDE.md`).

## Status
- **Done:** the whole Table 2 pipeline: code, six configs, runner, plots, docs. 41 tests pass.
- **Run so far:** short CPU sanity checks only. The largest was 3,000 steps of the full row (relative L2 0.93 → about 0.6), which doesn't say much.
- **Never run:** a full 200k-step run, and anything on CUDA.
- **Next:** follow `windows_run_guide.md` to set up the laptop, verify, benchmark, run all six rows unattended, and pack the results.
- **Not implemented:** Modified MLP, PirateNet with physics-informed init, NTK weighting, SOAP.

## What exists
- **`piratenets/`:**
  - the Allen–Cahn residual, with derivatives via `torch.func` (an `autograd` path is also available);
  - exact periodic embedding, trainable random Fourier features, random weight factorization (RWF) layers, and a plain MLP;
  - causal weighting and grad-norm loss weights;
  - a trainer with CSV logging, and checkpoints whose resume is bit-exact on CPU (tested).
- **`configs/allen_cahn/ablation/`:** one YAML per Table 2 row. Each differs from `default.yaml` only in its switch, and a test checks that.
- **`scripts/`:** `train.py`, `run_ablation.py` (skips finished rows, resumes interrupted ones), `benchmark_step.py`, `plot_ablation.py`, `plot_run.py`, `pack_results.py`.
- **`notebooks/colab_ablation.ipynb`:** a Colab fallback.
- **`docs/decisions.md`:** every detail the papers leave open (D1–D20), with its source. Read this before changing any setting. `docs/spec.md` describes what is implemented.

## Decisions that differ from a plain reading of the paper (all taken from jaxpi)
- 200k steps.
- Fourier features: scale σ = 1, 256 outputs, B trained.
- RWF μ = 0.5.
- Truncated-normal Glorot initialization.
- Causal chunks formed by sorting the sampled times into 32 equal-count groups.
- IC loss on the fixed 512-point reference x-grid.
- Grad-norm weights updated *after* the optimizer step, using the *mean* of the norms.

Expected side effect: after the first update, λ_ic ≈ 82. Adam's first step briefly inflates the losses, and the update lands on that step (see D13 and D20).

## Working conventions
- **jaxpi:** values and design decisions may come from jaxpi, but never copy its code. Where the paper and jaxpi disagree, default to jaxpi, implemented independently, and keep the paper's version as a config option. Record each such choice in `docs/decisions.md`.
- **Configs:** every component is a config switch, and each experiment is a YAML file.
- **Plots:** they read the CSV logs and never re-run training.
- **Compute:** now the laptop's RTX 3070 Ti. The Colab line in `CLAUDE.md` predates the move.
- **Papers:** `papers/` is gitignored. Download them from arXiv if needed: 2402.00326 (PirateNets) and 2308.08468 (Expert's Guide).

## Commands
```
pytest -q
python scripts/benchmark_step.py
python scripts/run_ablation.py            # outputs in runs/allen_cahn/ablation/<row>/
python scripts/plot_ablation.py
```
