"""Sweep AlphaIVEA's α parameter and report discrete-cell distortion.

Re-uses the metric machinery from probe_projection_distortion.py.
At α=1.0 the row should exactly match plain IVEA.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_projection_distortion import measure  # active_projection used inside

from mu3.projection import (
    AlphaIVEAProjection,
    AlphaSlerp,
    IVEAProjection,
)


def main():
    alphas = np.linspace(0.7, 1.5, 17)

    print(f"{'α':>6s} {'area_r':>8s} {'edge_r':>8s} "
          f"{'shape_mx':>9s} {'sh_p90':>7s} {'sh_p50':>7s} "
          f"{'ang_mx°':>8s} {'ang_p90°':>9s} {'ang_p50°':>9s}",
          flush=True)
    rows = []
    for a in alphas:
        factory = partial(AlphaIVEAProjection, alpha=float(a))
        m = measure(factory)
        rows.append((float(a), m))
        print(f"{a:6.3f} {m['area_r']:8.4f} {m['edge_r']:8.4f} "
              f"{m['shape_mx']:9.4f} {m['shape_p90']:7.4f} {m['shape_p50']:7.4f} "
              f"{m['ang_dev']:8.3f} {m['ang_p90']:9.3f} {m['ang_p50']:9.3f}",
              flush=True)

    # Reference rows
    print("\n--- references ---", flush=True)
    for name, cls in [("IVEA (α=1, by class)", IVEAProjection),
                       ("AlphaSlerp rich",       AlphaSlerp)]:
        m = measure(cls)
        print(f"{name:<24s} {m['area_r']:8.4f} {m['edge_r']:8.4f} "
              f"{m['shape_mx']:9.4f} {m['shape_p90']:7.4f} {m['shape_p50']:7.4f} "
              f"{m['ang_dev']:8.3f} {m['ang_p90']:9.3f} {m['ang_p50']:9.3f}",
              flush=True)

    # Best-α candidates
    print("\n--- best α along each axis ---", flush=True)
    best_area = min(rows, key=lambda r: r[1]['area_r'])
    best_ang_p90 = min(rows, key=lambda r: r[1]['ang_p90'])
    best_ang_max = min(rows, key=lambda r: r[1]['ang_dev'])
    best_balance = min(rows, key=lambda r:
        max((r[1]['area_r'] - 1.0) * 100.0, r[1]['ang_p90']))

    def fmt(label, a, m):
        return (f"{label}: α = {a:.3f}  "
                f"area_r {m['area_r']:.4f}  "
                f"ang_p90 {m['ang_p90']:.3f}°  "
                f"ang_max {m['ang_dev']:.3f}°  "
                f"shape_mx {m['shape_mx']:.4f}")

    print(fmt("min area_r",     best_area[0], best_area[1]))
    print(fmt("min ang_p90°",   best_ang_p90[0], best_ang_p90[1]))
    print(fmt("min ang_max°",   best_ang_max[0], best_ang_max[1]))
    print(fmt("balanced minmax", best_balance[0], best_balance[1]))


if __name__ == "__main__":
    main()
