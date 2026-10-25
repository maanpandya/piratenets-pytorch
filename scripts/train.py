"""Train one configuration.

    python scripts/train.py --config configs/allen_cahn/ablation/default.yaml
    python scripts/train.py --config ... --set training.steps=300 logging.eval_every=100
    python scripts/train.py --config ... --out-dir /content/drive/MyDrive/piratenets/runs/default

Resumes from <checkpoint_dir>/last.pt if it exists (use --no-resume to start over).
"""
import argparse

from piratenets.config import load_config
from piratenets.trainer import Trainer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--out-dir", help="override logging.out_dir")
    p.add_argument("--checkpoint-dir", help="override logging.checkpoint_dir")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    p.add_argument("--max-steps", type=int, help="stop after this many steps in this session")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if a.out_dir:
        cfg.logging.out_dir = a.out_dir
    if a.checkpoint_dir:
        cfg.logging.checkpoint_dir = a.checkpoint_dir
    Trainer(cfg).train(resume=not a.no_resume, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
