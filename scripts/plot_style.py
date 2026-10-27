"""Shared figure style: one categorical order, recessive chrome, and colormaps."""
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots in fixed order (validated for CVD separation on adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 1.5, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def read_metrics(path):
    """metrics.csv -> {column: list of (step, value)} with empty cells skipped."""
    series = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            for key, value in row.items():
                if key != "step" and value not in ("", None):
                    v = float(value)
                    if math.isfinite(v):
                        series.setdefault(key, []).append((step, v))
    return series
