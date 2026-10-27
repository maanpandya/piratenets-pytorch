"""Zip run outputs for sharing, leaving out checkpoints.

    python scripts/pack_results.py                     # runs/allen_cahn/ablation -> ablation_results.zip
    python scripts/pack_results.py --out-root runs/allen_cahn/ablation --zip results.zip

Includes metrics.csv, config.yaml, summary.json, prediction.npz, figures, the
comparison table, and any *.log files under the output root.
"""
import argparse
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="runs/allen_cahn/ablation")
    p.add_argument("--zip", default="ablation_results.zip")
    a = p.parse_args()

    root = Path(a.out_root).expanduser()
    files = sorted(f for f in root.rglob("*") if f.is_file() and "checkpoints" not in f.relative_to(root).parts)
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(root.name) / f.relative_to(root))
    size_mb = Path(a.zip).stat().st_size / 1e6
    print(f"wrote {a.zip}: {len(files)} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
