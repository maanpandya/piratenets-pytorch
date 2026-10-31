# Handoff (updated 2026-10-30)

The Windows laptop (RTX 3070 Ti Laptop GPU) is now the only workspace. Earlier work was done on a Mac, and there is nothing to send back there.

## Goal
Reproduce the Allen–Cahn ablation from the Expert's Guide (arXiv:2308.08468, **Table 2**) in PyTorch. The target is the relative L2 error of each row; the paper's values are in the table at the end of `windows_run_guide.md` (full pipeline: 5.84e-4). The implementation is independent and MIT-licensed (see `CLAUDE.md`).

## Where things stand
- **Code:** the whole Table 2 pipeline is done. 48 tests pass.
- **Laptop:** set up per `windows_run_guide.md`: Python 3.12, torch 2.14.1+cu132, a CUDA venv in `.venv`. The first GPU run found two platform problems, both fixed:
  - resume failed on CUDA because the RNG states were loaded onto the GPU (`c079158`);
  - Windows background power throttling (EcoQoS) slowed minimized runs 2–3× once the display turned off. `Trainer` now opts its own process out (`53a52b5`).
- **Derivative mode:** `--set training.derivatives=autograd`. It measured 11.1 ms/step against 21.4 for `func`, and gives the same values up to round-off.
- **RWF trains exp(s) itself since 2026-10-30** (`model.rwf.scale_param: direct`, jaxpi's version; D21). Runs before that trained s (`log`, EG Algorithm 2). The switch made no consistent difference to the error.
- **Results** (next section): every row has seeds 0–2, and `default` also seeds 3–4. By median, `no_grad_norm`, `no_fourier` and `plain` match the paper, and `no_rwf` is 1.8× high. `default` (3.9–8.2× high over five seeds) and `no_causal` (3.1–3.5×) are further off.
  - The paper's own logged runs (W&B) show the same loss weights and losses as ours, but lower error. The cause is not found yet; see "The paper's own runs" and "Next steps".

## Current results (`scale_param: direct`)
| Row | Seeds 0, 1, 2 (3, 4) | Median | Paper | Median / paper |
|---|---|--:|--:|--:|
| default | 2.93e-3, 2.27e-3, 2.27e-3 (3.67e-3, 4.81e-3) | 2.93e-3 | 5.84e-4 | 5.0 |
| no_causal | 5.61e-3, 4.95e-3, 5.16e-3 | 5.16e-3 | 1.59e-3 | 3.2 |
| no_rwf | 1.39e-2, 1.20e-2, 7.60e-3 | 1.20e-2 | 6.62e-3 | 1.8 |
| no_grad_norm | 2.86e-3, 6.41e-3, 6.14e-3 | 6.14e-3 | 7.51e-3 | 0.82 |
| no_fourier | 5.12e-1, 5.06e-1, 4.24e-1 | 5.06e-1 | 4.35e-1 | 1.2 |
| plain | 5.25e-1, 4.94e-1, 5.08e-1 | 5.08e-1 | 5.17e-1 | 0.98 |

- **Folders:** the RWF rows are in `runs/allen_cahn/rwf_direct_seedN/`. `no_rwf` and `plain` have no RWF, so their runs in `ablation/` (seed 0), `ablation_seed1/` and `ablation_seed2/` count unchanged. Each run folder has `metrics.csv`, `summary.json`, `training.png`, `solution.png` and its checkpoint.
- **Ordering:** `default` < `no_causal` < `no_grad_norm` < `no_rwf` < `no_fourier` ≈ `plain`. The paper has `no_rwf` before `no_grad_norm`, within 15% of each other, and `no_fourier` 16% below `plain`; ours end within 0.4%.
- **The `log` runs** in the next two sections give the same picture. Pair by pair, the two versions differ about as much as two seeds do (D21).
- **The older folders hold `log` runs.** `runs/allen_cahn/ablation/` and `ablation_seed1/`–`ablation_seed4/` trained s. `scripts/run_ablation.py` now stops on their RWF rows rather than report them under the new default. To resume or extend them, add `--set model.rwf.scale_param=log`. Put new `direct` runs in their own `--out-root`.

## Seed-0 results (`log`)
| Row | Ours | Paper | Ours / paper | Wall time (min) |
|---|--:|--:|--:|--:|
| default | 3.39e-3 | 5.84e-4 | 5.8 | 91.9 (about 50 without throttling) |
| no_fourier | 5.33e-2 | 4.35e-1 | 0.12 | 43.9 |
| no_rwf | 1.39e-2 | 6.62e-3 | 2.1 | 52.1 |
| no_grad_norm | 7.28e-4 | 7.51e-3 | 0.10 | 52.0 |
| no_causal | 8.63e-3 | 1.59e-3 | 5.4 | 52.0 |
| plain | 5.25e-1 | 5.17e-1 | 1.0 | 39.5 |

- With seed 0 alone, `no_grad_norm` beat `default` by 4.6×, the reverse of the paper. Seeds 1–2 show this was seed noise; see the next section.
  - In `default`, λ_ic stays around 20–50 and λ_r around 0.5. The IC loss falls to about 1e-10, while the residual loss plateaus near 1e-7 from about 120k steps.
  - In `no_grad_norm`, both losses settle near 1–5e-8.
- `no_fourier` does about 8× better than the paper, which seeds 1–2 also show to be seed noise. `plain` matches the paper.
- In every row with causal weighting, the minimum causal weight rises to 1 and stays there (≥ 0.99). That happens from about 34–40k steps in `default`, `no_rwf` and `no_grad_norm`, and from 67k in `no_fourier`.
- **Outputs:** in `runs/allen_cahn/ablation/` (gitignored). That holds `ablation_table.md`, `ablation_rel_l2.png`, `run_notes.md`, and per row `metrics.csv`, `training.png`, `solution.png` and the checkpoints. `ablation_results.zip` in the repo root packs the same outputs without checkpoints.
- **Re-running:** to re-run or extend a row, use `windows_run_guide.md` step 7, adding `model.rwf.scale_param=log` to the `--set` list for this folder. Finished rows are skipped and an interrupted row resumes. Never pass `--no-resume` on `runs/allen_cahn/ablation/`.

## Seeds 1–2 and the remaining gap (2026-10-29, `log`)
- **Runs:** seeds 1 and 2 of every row, one process at a time, go into `runs/allen_cahn/ablation_seed1/` and `ablation_seed2/` (gitignored). `default` seeds 3 and 4 go into `ablation_seed3/` and `ablation_seed4/`.
  - `runs\seeds_chain.cmd` ran `default` and `no_grad_norm`. A Windows Update reboot at 09:34 on 2026-10-29 stopped it.
  - `runs\seeds_chain2.cmd` ran the rest, finishing at 23:12 on 2026-10-29. Re-running a chain is safe, since finished rows are skipped and interrupted rows resume. Nothing restarts a chain after a reboot.

| Row | Fourier | RWF | Grad norm | Seeds 0, 1, 2 (3, 4) | Median | Paper | Median / paper |
|---|:-:|:-:|:-:|---|--:|--:|--:|
| default | yes | yes | yes | 3.39e-3, 2.88e-3, 2.56e-3 (5.03e-3, 4.17e-3) | 3.39e-3 | 5.84e-4 | 5.8 |
| no_causal | yes | yes | yes | 8.63e-3, 2.66e-3, 4.89e-3 | 4.89e-3 | 1.59e-3 | 3.1 |
| no_rwf | yes | no | yes | 1.39e-2, 1.20e-2, 7.60e-3 | 1.20e-2 | 6.62e-3 | 1.8 |
| no_grad_norm | yes | yes | no | 7.28e-4, 5.65e-3, 5.70e-3 | 5.65e-3 | 7.51e-3 | 0.75 |
| no_fourier | no | yes | yes | 5.33e-2, 5.09e-1, 4.08e-1 | 4.08e-1 | 4.35e-1 | 0.94 |
| plain | no | no | no | 5.25e-1, 4.94e-1, 5.08e-1 | 5.08e-1 | 5.17e-1 | 0.98 |

- Per-seed tables and plots are in each root's `ablation_table.md` and `ablation_rel_l2.png`, and each run's `training.png` and `solution.png`.
- **Both seed-0 surprises were seed noise.**
  - `no_grad_norm` seed 0 (7.28e-4, which beat `default`) was an outlier. Seeds 1–2 follow the paper's `no_grad_norm` curve in EG Fig. 4 closely: a stall near 0.4–0.5 until 30–40k steps, a sharp drop, then 8.2e-3 at 100k against the paper's ~9.5e-3.
  - `no_fourier` seed 0 (5.33e-2) escaped the wrong solution at 60–80k steps. Seeds 1–2 stay at an error of 0.4–0.5 to the end, as in the paper, and so does seed 0 under D21's `direct` (0.51). Their L_r stays near 1.5–4e-4, so the causal weights sit at about 0.98–0.995 throughout, and causal weighting never holds the later times back.
- **By the median, the paper's ordering holds** (`default` < `no_causal` < `no_grad_norm`, `no_rwf` < `no_fourier` < `plain`). The one exception is that `no_rwf` and `no_grad_norm` swap; they are within 15% of each other in the paper. Each component helps, but by less than in the paper: disabling grad norm costs 1.7×, against the paper's 13×.
- **The remaining gap is in the rows with both RWF and grad norm.** Compared with the paper's own logged runs (JW in `docs/decisions.md`):
  - `default` is at 1.3–1.6× the paper's error at 10k steps (0.47–0.59 against 0.37), and 2.0–2.6× at 40k. Between 40k and 100k the paper's error drops 7.7×, ours 3.0–3.9×. Ours ends 4.4–8.6× high, so even the best of five seeds is far from the paper's run.
  - `no_causal` behaves the same way, more weakly (1.7–5.4× high).
  - `no_rwf` (grad norm without RWF) is 1.1–2.1× high, and seed 2 tracks the paper's curve from 20k steps on. `no_grad_norm` (RWF without grad norm) matches.
  - All rows share the learning-rate schedule, so the schedule is not the cause.

## The paper's own runs (2026-10-30)
jaxpi's Allen–Cahn README links a public Weights & Biases project, `jaxpi/allen_cahn`. It holds the paper's runs from 2023-04-14, one per Table 2 row, each logged every 100 steps. `docs/decisions.md` lists them as source JW. In everything the training sees, our `default` runs (the five `log` runs) look like the paper's:

| `default` at step | 10k | 40k | 100k | 200k |
|---|--:|--:|--:|--:|
| λ_r, paper | 0.67 | 0.52 | 0.51 | 0.51 |
| λ_r, ours (5 seeds) | 0.67 | 0.52 | 0.51–0.52 | 0.50–0.51 |
| λ_ic, paper | 33 | 21 | 21 | 25 |
| λ_ic, ours | 29–42 | 18–25 | 18–35 | 51–113 |
| L_r, paper | 1.2e-2 | 5.7e-5 | 4.7e-7 | 1.5e-7 |
| L_r, ours | 0.6–1.5e-2 | 0.5–16e-5 | 2.3–6.7e-7 | 0.8–4.4e-7 |
| min causal weight, paper | 0.64 | 1.00 | 1.00 | 1.00 |
| min causal weight, ours | 0.52–0.82 | 1.00 | 1.00 | 1.00 |
| rel. L2, paper | 0.37 | 7.6e-3 | 9.9e-4 | 5.84e-4 |
| rel. L2, ours | 0.47–0.59 | 1.5–2.0e-2 | 4.1–6.6e-3 | 2.6–5.0e-3 |

- **Same losses, higher error.** Our λ_r stays within 0.008 of the paper's at all 2,000 logged steps (four seeds; 0.03 for the fifth). Our losses reach the same levels, and the causal weights open up at the same time. Only the error differs.
  - Over the last 5k steps, the paper's `default` has rel. L2 / √L_r = 1.5. Our ten `default` runs (five regular, five D21) have 7.2–12.5.
  - The same holds, less strongly, in every row: `no_causal` 3.8 against 8.0–12.8, `no_rwf` 9.4 against 13.4–15.1, `no_grad_norm` 35 against 44–51.
  - So our networks reach the paper's loss levels but sit farther from the true solution.
- **That ratio is seed-dependent.** `no_grad_norm` seed 0 has 5.5, against 44–51 for seeds 1–2, at the same L_r. Its D21 twin, from the same network and batches, has 25 (and 2.9e-3 error instead of 7.3e-4). How close a run gets at a given loss depends on which solution it settles into, and small changes to the updates can change that.
  - Nothing the losses or weights show distinguishes the paper's `default` run from ours.
  - The paper's rows share one seed (D18): `default` and `default_ntk` take the identical first step. So the two runs agreeing near 5.7e-4 is one seed, not two.
- **Confirmed by JW:** D13 (the update timing, with the same jump at step 0), D14 (the plain row is 0.5174), D20 (the mean: the paper's λ_r settles at 0.51, while the sum would keep it above 1), and D2 (200k steps).
- **Hardware:** each paper run used one NVIDIA RTX A6000, with batch 4096 per run.
- **Ruled out:**
  - Settings whose effect would show in the logged losses or weights (D10–D13, D15, D20). These follow the paper's run.
  - The RWF parametrization (D21). jaxpi's version shows no consistent effect over 14 paired runs (below).
  - IC over-weighting as such. At equal residual loss (L_r ≈ 6e-7), `default` has about half the error of `no_grad_norm`.
  - The reference solution and metric. jaxpi's own reference data (Chebfun) and ours agree to 1.5e-5 relative L2 (D17).
  - Changes in jaxpi since 1.0.0. Its library and Allen–Cahn example have changed only cosmetically since 2023-08-06 (`docs/decisions.md`, Sources). The paper's runs predate 1.0.0, though, so differences in their pre-release code cannot be ruled out.
- **Matmul precision: not the cause** (tested with jaxpi itself, below).
  - jaxpi has pinned JAX's float32 matmul precision to `"highest"` only since 2024-11 (`df13b96a`). That came after a report of hardware-dependent results (jaxpi issue #22), so the paper's runs used JAX's default.
  - jaxpi's seed 42 at JAX's default precision (TF32 on this GPU) ends at 7.31e-4, against 5.52e-4 at full precision. Our own runs keep TF32 off (rule 1 of `windows_run_guide.md`).
- **D21 test (`runs\rwf_direct_chain.cmd` and `runs\rwf_direct_chain2.cmd`, `model.rwf.scale_param=direct`, into `runs/allen_cahn/rwf_direct_seed*/`):**
  - Each pair starts from the same network and sees the same batches. The 11 pairs of `default`, `no_causal` and `no_grad_norm` are within 7% at 10k steps, then drift apart about as far as different seeds do.
  - `default`, seeds 0–4: `direct` is lower in 4 of 5 (ratios 0.73–0.89, and 1.15 for seed 4). Median 3.39 → 2.93e-3, still 5.0× the paper.
  - `no_causal` and `no_grad_norm`, seeds 0–2: `direct` is higher in 5 of 6 (ratios 0.65–3.9). `no_grad_norm` seed 0 loses its lucky 7.3e-4 and ends at 2.9e-3.
  - `no_fourier`, seeds 0–2 (`runs\rwf_direct_chain3.cmd`): seeds 1–2 within 4%. Seed 0 loses its lucky 5.3e-2 and ends at 0.51.
  - Over all 14 pairs `direct` is lower in 6, and the geometric mean of the ratios is 1.28 (0.98 without the two seed-0 outliers). The table is in D21.
  - Medians, with `no_rwf` and `plain` unchanged since they have no RWF:

    | Row | Seeds | `log` | `direct` | Paper |
    |---|---|--:|--:|--:|
    | default | 0–4 | 3.39e-3 | 2.93e-3 | 5.84e-4 |
    | no_causal | 0–2 | 4.89e-3 | 5.16e-3 | 1.59e-3 |
    | no_grad_norm | 0–2 | 5.65e-3 | 6.14e-3 | 7.51e-3 |
    | no_fourier | 0–2 | 4.08e-1 | 5.06e-1 | 4.35e-1 |

    The row ordering is the same under both, though under `direct` `no_fourier` ends within 0.4% of `plain` (5.08e-1).
- **λ_ic climbs late in every `default` seed,** to 51–113 at 200k, where the paper's stays at 25. L_ic falls to 1e-10–2e-9. By then the learning rate is below 1e-6, so this barely moves the parameters.

## jaxpi itself (2026-10-31)
jaxpi's own Allen–Cahn `default` config, run unmodified on this laptop (source JR in `docs/decisions.md`). This separates "the paper's seed was favorable" from "our implementation differs somewhere the logs don't show".

| jaxpi run | rel. L2 at 10k / 40k / 100k | Final rel. L2 | rel. L2 / √L_r |
|---|---|--:|--:|
| paper (W&B) | 0.37 / 7.6e-3 / 9.9e-4 | 5.84e-4 | 1.5 |
| seed 42, full precision | 0.38 / 7.7e-3 / 9.4e-4 | 5.52e-4 | 2.4 |
| seed 0, full precision | 0.54 / 1.1e-2 / 2.5e-3 | 1.51e-3 | 5.6 |
| seed 1, full precision | 0.55 / 1.1e-2 / 2.4e-3 | 1.39e-3 | 4.3 |
| seed 42, JAX default precision (TF32) | 0.37 / 7.2e-3 / 9.6e-4 | 7.31e-4 | 1.8 |
| ours, `default`, 10 runs | 0.47–0.59 / 1.4–2.1e-2 / 3.6–6.6e-3 | 2.3–5.0e-3 | 7.2–12.5 |

- **The paper used seed 42**, and jaxpi reproduces the paper from it. With JAX's pre-0.5 random streams, jaxpi's seed 42 logs the paper's first step, so it starts from the same network (D18). It ends at 5.52e-4, and its error stays within about 10% of the paper's run the whole way, although its losses drift up to 2× from the paper's.
- **Seed 42 was favorable.** jaxpi's seeds 0 and 1 end 2.5–2.7× above it. At 10k steps they are at 0.54–0.55, inside our range.
- **Ours are still about 2× worse than jaxpi's other seeds:** our best of ten (2.27e-3) is above both. So the gap to the paper looks like two parts: about 2.6× from the paper's favorable seed, and about 2× from something in our training.
- **Not the initialization:** jaxpi's initial weights and ours have the same statistics, per layer: Glorot spread, the truncation at 2σ, log g ≈ N(0.5, 0.1²), and B ~ N(0, 1).
- **Not the training step:** from jaxpi's seed-42 weights and one fixed batch, our `Trainer.train_step` and jaxpi's step agree.
  - Both losses are identical to 8 digits, and the chunk losses and causal weights agree to 3e-7.
  - The raw grad-norm weights are identical to 6 digits, and λ_ic after the update is 67.7440 against 67.7431.
  - After one Adam step, every parameter moves in the same direction, within 1.5e-5 on steps of 1e-3.
  - jaxpi's optimizer is plain Adam with the same exponential decay as ours, with no warmup, clipping or gradient accumulation.
  - The one remaining difference is the initial-condition grid. jaxpi uses its data's `linspace(-1, 1, 512)`, which holds x = ±1 twice; ours uses 512 periodic points on [-1, 1). The step test used jaxpi's grid.
- **λ_ic climbs late in jaxpi too** (36–67 at 200k, against the paper's 25), so that climb in our runs is not a difference from jaxpi.
- **TF32 is not the cause.** At JAX's default precision, seed 42 follows the paper's curve as closely as at full precision, and ends 25% higher (7.31e-4). This does not show which precision the paper used.
- **Running:** our trainer from jaxpi's exact initial networks for seeds 42 and 0 (`runs\from_jaxpi_init.ps1`, into `runs/allen_cahn/from_jaxpi_init/`). The jaxpi chain is paused; re-launching it runs seeds 2–3.
  - Our model reproduces jaxpi's initial predictions to 7.6e-7 from the loaded arrays.
  - If our trainer then matches jaxpi's results from those networks, the 2× comes from chance in our seeds. If not, our training loop differs. The chain (`~/jaxpi-runs/chain.sh` in Ubuntu) takes about 50 min per run. It skips finished runs, and an interrupted run starts over.
- **Setup:**
  - jaxpi `main` (`f944a0c`) is cloned at `~/jaxpi` in WSL2 Ubuntu, with a virtualenv at `~/jaxpi-venv`. It has jax 0.6.2 (the newest for Python 3.10), CPU-only torch (jaxpi's sampler imports it) and `tabulate`.
  - Our wrapper `~/jaxpi-runs/run_ac.py` replaces W&B with a JSON-lines log (`~/jaxpi-runs/*.jsonl`). Its options: `--seed`, `--precision highest|default`, and `--rng legacy`, the old random streams that reproduce the paper's initialization.
  - Launch from Windows with `Start-Process wsl.exe -ArgumentList "-d","Ubuntu","--exec","bash","/home/maan/jaxpi-runs/chain.sh" -WindowStyle Minimized`.
- **License** (read 2026-10-30): jaxpi may be used, copied and modified for non-profit research only, and not distributed to third parties without Penn's approval. Publications should acknowledge it. Keep its code, and the wrapper, out of this repository, and never imitate its code here.

## Next steps
- **Finish the jaxpi comparison** (above). If jaxpi's other seeds land near 6e-4, our implementation differs from jaxpi somewhere; if they spread like ours, the paper's seed was favorable.
- **Comparing with the paper's runs:** the W&B histories can be read without an account through `https://api.wandb.ai/graphql` (project `jaxpi/allen_cahn`; run ids in `docs/decisions.md`, JW). They log the error, both losses, both weights and the minimum causal weight every 100 steps. Prefer them over values read off EG Fig. 4.
- **When a run looks off,** check `training.png`:
  - the minimum causal weight should reach 1 by about 30k steps, as in the paper's runs;
  - after the first update, λ_ic is 49 in the paper's run and 56–81 in ours on CUDA (D13).
- **Settings taken from jaxpi** (D2–D13, D20, D21) are matched to its code, and JW confirms D2, D13 and D20. Change them only with evidence, and record any change in `docs/decisions.md`.

## What exists
- **`piratenets/`:**
  - the Allen–Cahn residual, with derivatives via `torch.func` (an `autograd` path is also available);
  - exact periodic embedding, trainable random Fourier features, random weight factorization (RWF) layers, and a plain MLP;
  - causal weighting and grad-norm loss weights;
  - a trainer with CSV logging, and checkpoints whose resume is bit-exact on CPU and on CUDA (both are tested).
- **`configs/allen_cahn/ablation/`:** one YAML per Table 2 row. Each differs from `default.yaml` only in its switch, and a test checks that.
- **`scripts/`:** `train.py`, `run_ablation.py`, `benchmark_step.py`, `plot_ablation.py`, `plot_run.py`, `pack_results.py`.
- **`docs/decisions.md`:** every detail the papers leave open (D1–D21), with its source. Read it before changing any setting. `docs/spec.md` describes what is implemented.
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
- RWF trains the scale exp(s) itself, not s.

## Working conventions
- **jaxpi:** values and design decisions may come from jaxpi, but never copy its code. Where the paper and jaxpi disagree, default to jaxpi, implemented independently, and keep the paper's version as a config option. Record each such choice in `docs/decisions.md`.
- **Configs:** every component is a config switch, and each experiment is a YAML file.
- **Plots:** they read the CSV logs and never re-run training.
- **Compute:** this laptop's GPU. Launch long runs as a detached process (`windows_run_guide.md` step 7). They take about 40–50 min per 200k-step row.
- **Python:** call the venv interpreter explicitly (`.\.venv\Scripts\python.exe`); each shell command starts a fresh shell.
- **Papers:** `papers/` is gitignored. Download them from arXiv if needed: 2402.00326 (PirateNets) and 2308.08468 (Expert's Guide).
