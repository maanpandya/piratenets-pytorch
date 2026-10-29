# Windows run guide: Allen–Cahn ablation on the local GPU

**Status: completed on the laptop.** Keep this as the reference for re-doing the setup
and for the launch and resume commands (step 7). For the current state and next steps,
see `handoff.md`.

**For the Claude Code session on the Windows laptop (RTX 3070 Ti Laptop GPU).**
The setup steps were:
1. set up this repo;
2. check that it runs correctly on the GPU;
3. measure its speed;
4. run the six-row ablation (Expert's Guide, Table 2) unattended;
5. pack the results.

Report to the user at the checkpoints marked **Report**. The run takes hours; your estimate in step 6 sets the expected length.

## Rules

These rules apply on top of `CLAUDE.md`. See also `handoff.md` for the project's status.

1. **Don't change the experiment.** Do not edit configs, hyperparameters, numerics or the precision settings. TF32 must stay off; `piratenets/trainer.py` turns it off.
   - The only override you may add to the ablation command is `training.derivatives=autograd`, and only under the rule in step 6.
2. **Fix only platform problems.** If something fails because of Windows, make the smallest platform-only fix and show the user the diff. Then re-run the tests.
3. **Commit or push only when the user asks.**
4. **Don't use jaxpi code.** This repo is an independent, MIT-licensed reimplementation. Never copy code from `PredictiveIntelligenceLab/jaxpi` or any port of it. You don't need to look at it for this task.
5. **Keep test runs separate.** Smoke and benchmark runs write only to `runs/smoke/` (or a temp dir), never to `runs/allen_cahn/ablation/`.
6. **Never delete `runs/allen_cahn/ablation/`.** Its checkpoints are what make a resume possible.
7. **Ask before changing system settings,** such as power and sleep.

### Shell notes
- Each shell command you run starts a fresh shell, so an activated venv doesn't carry over.
- So always call the venv's interpreter explicitly: `.\.venv\Scripts\python.exe` in PowerShell, `./.venv/Scripts/python.exe` in Git Bash.
- Below, **`PY`** stands for that interpreter.
- Run all commands from the repo root.

## 1. Preconditions

1. Run `nvidia-smi`. It must list the RTX 3070 Ti Laptop GPU. Note the driver version and the "CUDA Version" it reports.
2. Run `git --version` and `py --list`. You need Python 3.10 or newer; 3.12 is the safest choice for PyTorch's CUDA wheels.
3. Check the repo path. If it is inside OneDrive (for example `C:\Users\<name>\OneDrive\...`), tell the user. OneDrive locks files and can break the atomic checkpoint write. Suggest re-cloning to a path like `C:\dev\piratenets-pytorch`.

## 2. Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install the **CUDA build of PyTorch first**. The plain PyPI wheel on Windows is CPU-only.
1. Open the selector at https://pytorch.org/get-started/locally/ and choose Stable, Windows, Pip, Python, and a CUDA version no newer than the one `nvidia-smi` reports.
2. Run the command it gives, using `PY -m pip`. It looks like this, with the `cuXXX` suffix from the site:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX
   ```

Then install the project:
```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Check that PyTorch sees the GPU:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- **Expected:** a CUDA version, then `True`, then the RTX 3070 Ti name.
- **If you get `False` or `None`:** the CPU wheel was installed. Run `PY -m pip uninstall -y torch` and reinstall from the CUDA index.

## 3. Tests (CPU, a few seconds)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
- **Expected:** `43 passed` (on a machine without CUDA, one test is skipped).
- **If any test fails:** stop and report the output. Don't change test tolerances.

## 4. GPU smoke test and resume check

This checks the GPU path and resume on CUDA (the first run here found and fixed a CUDA-only resume bug). Run 200 steps, stop, then resume to step 400:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\allen_cahn\ablation\default.yaml --out-dir runs\smoke --no-resume --max-steps 200 --set training.steps=400 logging.eval_every=100
.\.venv\Scripts\python.exe scripts\train.py --config configs\allen_cahn\ablation\default.yaml --out-dir runs\smoke --set training.steps=400 logging.eval_every=100
```

Expected:
- The second command prints `resumed from step 200` and ends with `done: rel_l2 ...`.
- An error of about 0.6–1.0 is normal this early.
- `runs\smoke\summary.json` contains `"device": "cuda"`.
- `runs\smoke\metrics.csv` has rows at steps 0, 100, 200, 300 and 400, with no duplicates.

## 5. Benchmark

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_step.py
```
This prints the ms/step for the two derivative implementations, `func` and `autograd`, and the projected hours for 6 runs of 200,000 steps. They give the same values up to float32 round-off.

**Choosing the mode:**
- If `autograd` is at least 10% faster than `func`, use `--set training.derivatives=autograd` for the whole ablation.
- Otherwise use the default.
- Use the same mode for all six rows.

**Report:** GPU, driver, torch and CUDA versions, the test result, the smoke/resume result, both ms/step numbers, the chosen mode, and the projected total hours.

## 6. Prepare the laptop

Ask the user to do the following, or offer to run the `powercfg` commands, but only after they agree:
- Keep it plugged in and use the best-performance power mode.
- Never sleep or hibernate on AC power: `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`.
- If they will close the lid, set the lid-close action on AC power to "Do nothing".
- Close other GPU-heavy applications.
- Make sure it has airflow; long runs can thermal-throttle.

## 7. Run the ablation

Launch it as a separate process, so it keeps running if this Claude Code session ends. From PowerShell (from Git Bash, wrap it in `powershell -NoProfile -Command "..."`):

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$p = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "scripts\run_ablation.py" `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "runs\ablation_stdout.log" `
  -RedirectStandardError "runs\ablation_stderr.log" `
  -WindowStyle Minimized -PassThru
$p.Id
```

- If step 5 chose `autograd`, use `-ArgumentList "scripts\run_ablation.py","--set","training.derivatives=autograd"`.
- Tell the user the process ID and the estimated finish time.
- If `Start-Process` causes problems, give the user this command to run in their own PowerShell window in the repo folder, with the same `--set` if you chose `autograd`:
  `.\.venv\Scripts\python.exe scripts\run_ablation.py *>&1 | Tee-Object runs\ablation.log`

Rows run in this order: `default`, `no_fourier`, `no_rwf`, `no_grad_norm`, `no_causal`, `plain`. Outputs go to `runs\allen_cahn\ablation\ablation_<row>\`.

**Monitoring.** Check occasionally, not in a tight loop:
- `Get-Content runs\ablation_stdout.log -Tail 5` shows one line every 1,000 steps, with the loss, λ_ic, the minimum causal weight, the relative L2 error and the elapsed time.
- `Get-Process -Id <PID>` shows whether it is still running.
- `runs\ablation_stderr.log` should hold only warnings.
- In `nvidia-smi`, if GPU-Util stays below about 60%, tell the user that running two processes on disjoint rows could cut the total time. For example, `--only default no_fourier no_rwf` in one and `--only no_grad_norm no_causal plain` in the other. Start the second only if the user agrees.

**Report when `ablation_default` finishes.** `summary.json` appears and the log shows `[ablation_default] done: rel_l2 ...`.
- Give its final relative L2 error next to the paper's 5.84e-4.
- If it is more than about 10× worse (above ~6e-3), say so plainly. The user may want to stop and debug before spending hours on the other rows (see `handoff.md`). That is their call; the run continues otherwise.

**If the run is interrupted** (crash, reboot, sleep), launch the same command again:
- Finished rows are skipped.
- The interrupted row resumes from its last checkpoint, which is saved every 5,000 steps.
- Never add `--no-resume`.

## 8. Results

When all six rows are done (each has a `summary.json`):

```powershell
.\.venv\Scripts\python.exe scripts\plot_ablation.py
foreach ($r in "default","no_fourier","no_rwf","no_grad_norm","no_causal","plain") { .\.venv\Scripts\python.exe scripts\plot_run.py "runs\allen_cahn\ablation\ablation_$r" }
.\.venv\Scripts\python.exe scripts\pack_results.py
```

This writes `ablation_results.zip` in the repo root, a few MB without checkpoints.

**Report:**
- the contents of `runs\allen_cahn\ablation\ablation_table.md`;
- the wall time per row;
- anything unusual in the logs;
- any code changes you made, as a diff;
- the path of the zip, if the user wants to share the results.

Paper values for comparison (Expert's Guide Table 2, relative L2):

| Row | Fourier | RWF | Grad norm | Causal | Paper |
|---|:-:|:-:|:-:|:-:|--:|
| default | yes | yes | yes | yes | 5.84e-4 |
| no_fourier | no | yes | yes | yes | 4.35e-1 |
| no_rwf | yes | no | yes | yes | 6.62e-3 |
| no_grad_norm | yes | yes | no | yes | 7.51e-3 |
| no_causal | yes | yes | yes | no | 1.59e-3 |
| plain | no | no | no | no | ~5.2e-1 |

## Troubleshooting

- **`torch.cuda.is_available()` is False:** the CPU wheel is installed, or the driver is too old for the chosen CUDA wheel. Reinstall torch from a CUDA index no newer than `nvidia-smi`'s CUDA version.
- **`PermissionError` when replacing `last.pt`:** OneDrive or antivirus is holding the file. Move the repo out of OneDrive, or exclude the folder from scanning.
- **CUDA out of memory:** this is not expected, since a run should need well under 2 GB. Close other GPU applications and report it.
- **The run gets slower over time:** the elapsed time per 1,000 steps in the log creeps up. This is harmless to the results, but tell the user.
  - If GPU-Util in `nvidia-smi` stays high, it is likely thermal throttling.
  - If GPU-Util drops (to ~30%) while one CPU core is busy, Windows is throttling the background process (EcoQoS), typically once the display turns off. `Trainer` opts its process out of this at startup (`disable_power_throttling` in `piratenets/trainer.py`); before that was added, it slowed the first ablation's `default` row from 12 to 30–34 s per 1,000 steps.
- **`ModuleNotFoundError: piratenets`:** you are not using `PY`, or `pip install -e ".[dev]"` didn't run in the venv.
