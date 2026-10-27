"""Plot the Allen-Cahn ablation from the runs' CSV logs (Expert's Guide Fig. 4 / Table 2).

    python scripts/plot_ablation.py --out-root runs/allen_cahn/ablation

Writes <out-root>/ablation_rel_l2.png and <out-root>/ablation_table.md.
Reads only metrics.csv / summary.json; never re-runs training.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import INK_2, SERIES, apply_style, read_metrics

# Table 2 of the Expert's Guide (arXiv:2308.08468): switches (FF, RWF, grad norm, causal) and rel. L2.
# The plain row is printed as "51.74 x 10^-1"; we read it as 5.174e-1 (docs/decisions.md, D14).
ROWS = {
    "default":      ("Default",             (1, 1, 1, 1), 5.84e-4),
    "no_fourier":   ("No Fourier features", (0, 1, 1, 1), 4.35e-1),
    "no_rwf":       ("No RWF",              (1, 0, 1, 1), 6.62e-3),
    "no_grad_norm": ("No grad norm",        (1, 1, 0, 1), 7.51e-3),
    "no_causal":    ("No causal",           (1, 1, 1, 0), 1.59e-3),
    "plain":        ("Plain",               (0, 0, 0, 0), 5.174e-1),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="runs/allen_cahn/ablation")
    a = p.parse_args()
    root = Path(a.out_root).expanduser()
    apply_style()

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    table = [
        "| Row | Fourier | RWF | Grad norm | Causal | Ours rel. L2 | Paper rel. L2 | Ours / paper | Steps | Wall time (min) |",
        "|---|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|",
    ]
    max_step = 0
    for color, (row, (label, switches, paper)) in zip(SERIES, ROWS.items()):
        run = root / f"ablation_{row}"
        marks = " | ".join("yes" if s else "no" for s in switches)
        if not (run / "metrics.csv").exists():
            table.append(f"| {label} | {marks} | not run | {paper:.2e} | | | |")
            continue
        m = read_metrics(run / "metrics.csv")
        steps, errs = zip(*m["rel_l2"])
        max_step = max(max_step, steps[-1])
        ax.semilogy(steps, errs, color=color, label=label)
        summary = run / "summary.json"
        done = summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("finished")
        final, wall = errs[-1], m["wall_time"][-1][1] / 60
        status = f"{steps[-1]}" + ("" if done else " (running)")
        table.append(
            f"| {label} | {marks} | {final:.2e} | {paper:.2e} | {final / paper:.2f} | {status} | {wall:.1f} |"
        )
        ax.plot([steps[-1]], [paper], linestyle="none", marker="o", markersize=7,
                markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.5)

    ax.plot([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=INK_2,
            label="Paper, final (Table 2)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Allen-Cahn ablation: relative L2 error during training")
    ax.set_xlim(0, 1.03 * max(max_step, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.savefig(root / "ablation_rel_l2.png")

    text = "\n".join(table) + "\n"
    (root / "ablation_table.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {root / 'ablation_rel_l2.png'} and {root / 'ablation_table.md'}")


if __name__ == "__main__":
    main()
