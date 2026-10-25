"""Plot one run: training curves (Expert's Guide Fig. 19 style) and the solution vs. the reference.

    python scripts/plot_run.py runs/allen_cahn/ablation/ablation_default

Writes training.png (and solution.png if prediction.npz exists) into the run directory.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import DIVERGING, SEQUENTIAL, SERIES, apply_style, read_metrics


def plot_training(run: Path):
    m = read_metrics(run / "metrics.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0))

    ax = axes[0]
    for color, key, label in [(SERIES[0], "loss_ic", "IC loss"), (SERIES[1], "loss_r", "Residual loss")]:
        if key in m:
            ax.semilogy(*zip(*m[key]), color=color, label=label)
    ax.set_title("Losses (unweighted)")
    ax.legend()

    ax = axes[1]
    for color, key, label in [(SERIES[0], "lambda_ic", "lambda IC"), (SERIES[1], "lambda_r", "lambda residual")]:
        if key in m:
            ax.plot(*zip(*m[key]), color=color, label=label)
    ax.set_title("Loss weights")
    ax.legend()

    ax = axes[2]
    if "causal_w_min" in m:
        ax.plot(*zip(*m["causal_w_min"]), color=SERIES[0])
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Minimum causal weight")

    ax = axes[3]
    if "rel_l2" in m:
        ax.semilogy(*zip(*m["rel_l2"]), color=SERIES[0])
    ax.set_title("Relative L2 error")

    for ax in axes:
        ax.set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(run / "training.png")


def plot_solution(run: Path, reference: Path):
    pred = np.load(run / "prediction.npz")
    ref = np.load(reference)
    t, x, u, u_ref = pred["t"], pred["x"], pred["u_pred"], ref["usol"]
    err = np.abs(u - u_ref)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.2))
    extent = [t[0], t[-1], x[0], x[-1]]
    for ax, field, title, cmap, lim in [
        (axes[0], u_ref, "Reference", DIVERGING, (-1, 1)),
        (axes[1], u, "Prediction", DIVERGING, (-1, 1)),
        (axes[2], err, f"Absolute error (rel. L2 {float(pred['rel_l2']):.2e})", SEQUENTIAL, (0, err.max())),
    ]:
        im = ax.imshow(field.T, origin="lower", aspect="auto", extent=extent, cmap=cmap,
                       vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.grid(False)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run / "solution.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--reference", default=str(Path(__file__).resolve().parents[1] / "data/allen_cahn_ref.npz"))
    a = p.parse_args()
    run = Path(a.run_dir).expanduser()
    apply_style()
    plot_training(run)
    if (run / "prediction.npz").exists():
        plot_solution(run, Path(a.reference))
    print(f"wrote plots to {run}")


if __name__ == "__main__":
    main()
