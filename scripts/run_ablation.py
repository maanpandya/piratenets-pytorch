"""Run the Allen-Cahn ablation (Expert's Guide Table 2): one run per YAML, in table order.

Finished runs are skipped and interrupted runs resume from their last checkpoint,
so after a Colab disconnect just run the same command again.

    python scripts/run_ablation.py --out-root /content/drive/MyDrive/piratenets/ablation
    python scripts/run_ablation.py --only default plain
    python scripts/run_ablation.py --set training.steps=2000      # quick trial of every row
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from piratenets.config import REPO_ROOT, load_config
from piratenets.trainer import Trainer

ROWS = ["default", "no_fourier", "no_rwf", "no_grad_norm", "no_causal", "plain"]


def is_finished(out_dir: Path, steps: int) -> bool:
    summary = out_dir / "summary.json"
    if not summary.exists():
        return False
    s = json.loads(summary.read_text(encoding="utf-8"))
    return bool(s.get("finished")) and s.get("step") == steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs-dir", default=str(REPO_ROOT / "configs/allen_cahn/ablation"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "runs/allen_cahn/ablation"))
    p.add_argument("--only", nargs="*", choices=ROWS, help="subset of rows (default: all, in table order)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override applied to every row")
    a = p.parse_args()

    out_root = Path(a.out_root).expanduser()
    results = {}
    for row in a.only or ROWS:
        cfg = load_config(Path(a.configs_dir) / f"{row}.yaml", a.set)
        cfg.logging.out_dir = str(out_root / cfg.name)
        if is_finished(cfg.out_dir, cfg.training.steps):
            print(f"[{cfg.name}] already finished, skipping", flush=True)
            results[row] = json.loads((cfg.out_dir / "summary.json").read_text(encoding="utf-8"))["rel_l2"]
            continue
        trainer = Trainer(cfg)
        results[row] = trainer.train(resume=True)["rel_l2"]
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nfinal relative L2 errors:")
    for row, err in results.items():
        print(f"  {row:<14s} {err:.3e}")


if __name__ == "__main__":
    main()
