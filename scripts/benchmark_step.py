"""Time training steps on the current device and project the ablation's run time.

    python scripts/benchmark_step.py --config configs/allen_cahn/ablation/default.yaml

Times both derivative implementations (training.derivatives = func | autograd);
use whichever is faster via --set training.derivatives=... (they agree to
float32 round-off, see tests/test_derivatives.py).
"""
import argparse
import tempfile
import time

import torch

from piratenets.config import load_config
from piratenets.trainer import Trainer


def time_steps(cfg, n_warmup: int, n_steps: int) -> float:
    trainer = Trainer(cfg)
    for _ in range(n_warmup):
        trainer.train_step()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        trainer.train_step()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/allen_cahn/ablation/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=6, help="number of runs for the projection")
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("func", "autograd"):
            cfg = load_config(a.config, a.set + [f"training.derivatives={mode}"])
            cfg.logging.out_dir = tmp
            sec = time_steps(cfg, a.warmup, a.steps)
            hours = sec * cfg.training.steps / 3600
            print(
                f"{mode:>8s}: {sec * 1e3:7.1f} ms/step -> {hours:5.2f} h per run of {cfg.training.steps} steps, "
                f"{hours * a.runs:5.1f} h for {a.runs} runs",
                flush=True,
            )
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
