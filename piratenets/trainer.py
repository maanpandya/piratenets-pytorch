"""Training loop for the Allen-Cahn PINN (Expert's Guide Algorithm 1).

Per step: sample collocation points, compute the IC loss and the causally
weighted residual loss, take one Adam step, and every f steps update the global
loss weights. Logs to <out_dir>/metrics.csv, checkpoints to
<checkpoint_dir>/last.pt, and resumes from there exactly.
"""
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import causal
from .config import config_to_dict, save_config
from .models import build_model
from .problems import allen_cahn as ac
from .weighting import LossWeighting

CSV_FIELDS = [
    "step", "wall_time", "lr", "loss", "loss_ic", "loss_r",
    "lambda_ic", "lambda_r", "grad_norm_ic", "grad_norm_r",
    "causal_w_min", "causal_w_mean", "rel_l2",
]

def _resume_relevant(cfg_dict: dict) -> dict:
    """The parts of a config that must match for a resume. Logging, device, name, the
    total step count and the derivative implementation (same values up to round-off)
    may change, e.g. to extend a run or to move it to another GPU."""
    d = {k: v for k, v in cfg_dict.items() if k not in ("logging", "device", "name")}
    d["training"] = {k: v for k, v in d["training"].items() if k not in ("steps", "derivatives")}
    return d


def set_precision() -> None:
    """Full float32 matmuls: TF32 would cap the attainable error around 1e-3."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_factor(cfg):
    """Linear warmup from 0 over warmup_steps, then exponential decay."""

    def factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        exponent = (step - cfg.warmup_steps) / cfg.decay_steps
        if cfg.staircase:
            exponent = math.floor(exponent)
        return cfg.decay_rate ** exponent

    return factor


class CSVLogger:
    def __init__(self, path: Path, keep_below_step: int):
        self.path = path
        rows = []
        if path.exists() and keep_below_step > 0:
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["step"]) < keep_below_step]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({k: row.get(k, "") for k in CSV_FIELDS})


class Trainer:
    def __init__(self, cfg):
        set_precision()
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        seed = cfg.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = build_model(cfg.model, x_period=ac.X_PERIOD).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.optim.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor(cfg.optim))
        self.weighting = LossWeighting(
            ["ic", "r"], cfg.weighting.scheme, cfg.weighting.update_every, cfg.weighting.momentum,
            cfg.weighting.grad_norm_reduction,
        )
        # Collocation points come from their own generator so that sampling is
        # reproducible independently of anything else that draws random numbers.
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed + 1)
        self.reference = ac.load_reference(cfg.problem.reference)
        if cfg.training.ic_sampling == "grid":
            # The initial condition is fitted on the fixed x-grid of the reference data.
            self.x_ic = torch.as_tensor(self.reference.x, dtype=torch.float32, device=self.device)
        elif cfg.training.ic_sampling != "random":
            raise ValueError(f"unknown ic_sampling {cfg.training.ic_sampling!r}; use 'grid' or 'random'")
        self.step = 0
        self.elapsed = 0.0  # training wall time in seconds, summed over sessions

    # ------------------------------------------------------------------ losses
    def sample_batch(self):
        cfg = self.cfg
        t, x = ac.sample_collocation(
            cfg.training.batch_size, cfg.causal.num_chunks, self.generator, self.device
        )
        if cfg.training.ic_sampling == "grid":
            x_ic = self.x_ic
        else:
            x_ic = ac.sample_initial(cfg.training.n_ic, self.generator, self.device)
        return t, x, x_ic

    def compute_losses(self, batch):
        cfg = self.cfg
        t, x, x_ic = batch
        u_ic = self.model(torch.zeros_like(x_ic), x_ic)
        loss_ic = torch.mean((u_ic - ac.initial_condition(x_ic)) ** 2)

        r = ac.pde_residual(self.model, t.reshape(-1), x.reshape(-1), cfg.training.derivatives)
        chunk_losses = torch.mean(r.reshape(t.shape) ** 2, dim=1)
        loss_r, w = causal.residual_loss(chunk_losses, cfg.causal.tol, cfg.causal.enabled)
        return {"ic": loss_ic, "r": loss_r}, w

    def train_step(self):
        """One Adam step; on update steps the loss weights are then recomputed from the
        gradients at the new parameters on the same batch (docs/decisions.md, D13)."""
        batch = self.sample_batch()
        losses, w = self.compute_losses(batch)
        loss = self.weighting.total_loss(losses)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        updated = self.weighting.due(self.step)
        if updated:
            new_losses, _ = self.compute_losses(batch)
            self.weighting.update(new_losses, self.model.parameters())
        self.step += 1
        return loss, losses, w, updated

    # -------------------------------------------------------------- evaluation
    def predict(self) -> np.ndarray:
        return ac.predict_grid(self.model, self.reference, self.device)

    def evaluate(self) -> float:
        return ac.relative_l2(self.predict(), self.reference.usol)

    # ------------------------------------------------------------- checkpoints
    def checkpoint_path(self) -> Path:
        return self.cfg.checkpoint_dir / "last.pt"

    def state_dict(self) -> dict:
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "sampler": self.generator.get_state(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "step": self.step,
            "elapsed": self.elapsed,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "weighting": self.weighting.state_dict(),
            "rng": rng,
            "config": config_to_dict(self.cfg),
        }

    def load_state_dict(self, state: dict) -> None:
        saved, current = _resume_relevant(state["config"]), _resume_relevant(config_to_dict(self.cfg))
        if saved != current:
            diff = sorted(k for k in current if saved.get(k) != current[k])
            raise ValueError(f"checkpoint config differs from the current config in: {diff}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.weighting.load_state_dict(state["weighting"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        self.generator.set_state(rng["sampler"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.step = state["step"]
        self.elapsed = state["elapsed"]

    def save_checkpoint(self) -> None:
        path = self.checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)  # atomic: a crash mid-save never corrupts last.pt

    def load_checkpoint(self) -> bool:
        path = self.checkpoint_path()
        if not path.exists():
            return False
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        return True

    # -------------------------------------------------------------------- loop
    def train(self, resume: bool = True, max_steps: int | None = None) -> dict:
        """Train to cfg.training.steps (or stop early after max_steps steps this call)."""
        cfg = self.cfg
        out = cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        if resume and self.load_checkpoint():
            print(f"[{cfg.name}] resumed from step {self.step} ({self.checkpoint_path()})", flush=True)
        save_config(cfg, out / "config.yaml")
        logger = CSVLogger(out / "metrics.csv", keep_below_step=self.step)
        total = cfg.training.steps
        stop = total if max_steps is None else min(total, self.step + max_steps)
        log = cfg.logging

        session_start = time.perf_counter() - self.elapsed
        while self.step < stop:
            step = self.step
            rel_l2 = self.evaluate() if step % log.eval_every == 0 else None
            lr = self.scheduler.get_last_lr()[0]
            loss, losses, w, updated = self.train_step()
            self.elapsed = time.perf_counter() - session_start

            if step % log.log_every == 0:
                row = {
                    "step": step, "wall_time": f"{self.elapsed:.2f}", "lr": f"{lr:.6e}",
                    "loss": f"{loss.item():.6e}", "loss_ic": f"{losses['ic'].item():.6e}",
                    "loss_r": f"{losses['r'].item():.6e}",
                    "lambda_ic": f"{self.weighting.lambdas['ic']:.6e}",
                    "lambda_r": f"{self.weighting.lambdas['r']:.6e}",
                    "causal_w_min": f"{w.min().item():.6e}", "causal_w_mean": f"{w.mean().item():.6e}",
                }
                if updated:
                    row["grad_norm_ic"] = f"{self.weighting.grad_norms['ic']:.6e}"
                    row["grad_norm_r"] = f"{self.weighting.grad_norms['r']:.6e}"
                if rel_l2 is not None:
                    row["rel_l2"] = f"{rel_l2:.6e}"
                    print(
                        f"[{cfg.name}] step {step:>7d}  loss {loss.item():.3e}  ic {losses['ic'].item():.3e}"
                        f"  r {losses['r'].item():.3e}  lam_ic {self.weighting.lambdas['ic']:.2f}"
                        f"  w_min {w.min().item():.3f}  rel_l2 {rel_l2:.3e}  {self.elapsed / 60:.1f} min",
                        flush=True,
                    )
                logger.write(row)

            if self.step % log.checkpoint_every == 0 and self.step < total:
                self.save_checkpoint()

        if self.step < total:  # stopped early by max_steps
            self.save_checkpoint()
            return {"name": cfg.name, "step": self.step, "finished": False}
        return self.finish(logger)

    def finish(self, logger: CSVLogger) -> dict:
        cfg = self.cfg
        pred = self.predict()
        rel_l2 = ac.relative_l2(pred, self.reference.usol)
        logger.write({"step": self.step, "wall_time": f"{self.elapsed:.2f}", "rel_l2": f"{rel_l2:.6e}"})
        self.save_checkpoint()
        np.savez_compressed(
            cfg.out_dir / "prediction.npz",
            t=self.reference.t, x=self.reference.x, u_pred=pred.astype(np.float32), rel_l2=rel_l2,
        )
        summary = {
            "name": cfg.name, "step": self.step, "finished": True, "rel_l2": rel_l2,
            "wall_time_min": self.elapsed / 60, "seed": cfg.training.seed,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "torch": torch.__version__,
        }
        with open(cfg.out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{cfg.name}] done: rel_l2 {rel_l2:.3e} after {self.step} steps, {self.elapsed / 60:.1f} min", flush=True)
        return summary
