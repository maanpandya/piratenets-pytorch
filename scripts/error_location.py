"""Where the error of finished Allen-Cahn runs sits (see handoff.md, "jaxpi itself").

    python scripts/error_location.py runs/allen_cahn/rwf_direct_seed*/ablation_default

At x = 0 the initial condition has a double zero, so the solution starts at the unstable
state u = 0 and small errors there grow roughly like exp(5 t). For each run this prints
the share of the squared error near x = 0 and late in time, and the error at x = 0 at a
few times.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--reference", default=str(Path(__file__).resolve().parents[1] / "data/allen_cahn_ref.npz"))
    a = p.parse_args()
    ref = np.load(a.reference)
    t, x, u_ref = ref["t"], ref["x"], ref["usol"]
    i_x0 = np.argmin(np.abs(x))
    times = (0.0, 0.1, 1.0)
    i_t = [np.argmin(np.abs(t - s)) for s in times]

    print(f"{'rel. L2':>9} {'|x|<0.1 at t=1':>15} {'t>0.75':>7} " + " ".join(f"{f'e(t={s:g}, x=0)':>14}" for s in times)
          + f" {'rel. L2 / e(1, 0)':>18}  run")
    for run in map(Path, a.run_dirs):
        pred = np.load(run / "prediction.npz")
        err = pred["u_pred"] - u_ref
        sq = err ** 2
        rel = np.linalg.norm(err) / np.linalg.norm(u_ref)
        near0 = sq[-1, np.abs(x) < 0.1].sum() / sq[-1].sum()
        late = sq[t > 0.75].sum() / sq.sum()
        e0 = [err[i, i_x0] for i in i_t]
        print(f"{rel:9.2e} {near0:15.1%} {late:7.1%} " + " ".join(f"{v:+14.2e}" for v in e0)
              + f" {rel / e0[-1]:18.3f}  {run}")


if __name__ == "__main__":
    main()
