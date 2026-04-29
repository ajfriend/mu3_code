"""Report the distribution of mu3 cell areas at each resolution.

Two tables are printed:
  1. Per-resolution summary with hex-only min/max/std (pentagons are
     reported separately since they're all equal by icosa 5-fold symmetry
     and are always the smallest cell in the grid).
  2. Hex-area percentiles relative to the hex mean — a stable shape that
     captures the intrinsic distortion of the current projection.

Sum/4π is printed as a tiling-correctness check (should be 1 to float-
precision at every resolution).

Edit ``MAX_RES`` below to change the range. Deeper resolutions cost
roughly 7× more time per step.
"""

import math
import time

import numpy as np

from mu3 import cell_area, cells_at_res, is_pentagon


MAX_RES = 5

SPHERE = 4.0 * math.pi


def main() -> None:
    rows = []
    for res in range(MAX_RES + 1):
        t0 = time.perf_counter()
        pent, hex_ = [], []
        for cell in cells_at_res(res):
            a = cell_area(cell)
            (pent if is_pentagon(cell) else hex_).append(a)
        elapsed = time.perf_counter() - t0

        h = np.array(hex_) if hex_ else None
        pent_area = float(np.mean(pent))
        total = sum(pent) + (sum(hex_) if hex_ else 0.0)
        rows.append({
            "res": res,
            "n_cells": len(pent) + len(hex_),
            "n_hex": len(hex_),
            "pent_area": pent_area,
            "hex_mean": float(h.mean()) if h is not None else None,
            "hex_min": float(h.min()) if h is not None else None,
            "hex_max": float(h.max()) if h is not None else None,
            "hex_std_over_mean": float(h.std() / h.mean()) if h is not None else None,
            "hex_max_over_min": float(h.max() / h.min()) if h is not None else None,
            "pent_over_hex_mean": pent_area / float(h.mean()) if h is not None else None,
            "sum_ratio": total / SPHERE,
            "time_s": elapsed,
        })

    hdr = (f"{'res':>3}  {'cells':>6}  {'hex':>6}  "
           f"{'pent_area':>10}  {'hex_mean':>10}  "
           f"{'hex_min':>10}  {'hex_max':>10}  "
           f"{'hex_max/min':>11}  {'σ/μ':>6}  "
           f"{'pent/hex_μ':>10}  {'sum/4π':>10}  {'time(s)':>7}")
    print(hdr)
    print("-" * len(hdr))

    def _fmt(x, w, spec):
        return f"{x:>{w}{spec}}" if x is not None else f"{'—':>{w}}"

    for r in rows:
        print(
            f"{r['res']:>3}  {r['n_cells']:>6d}  {r['n_hex']:>6d}  "
            f"{r['pent_area']:>10.4e}  "
            f"{_fmt(r['hex_mean'], 10, '.4e')}  "
            f"{_fmt(r['hex_min'], 10, '.4e')}  "
            f"{_fmt(r['hex_max'], 10, '.4e')}  "
            f"{_fmt(r['hex_max_over_min'], 11, '.4f')}  "
            f"{_fmt(r['hex_std_over_mean'], 6, '.4f')}  "
            f"{_fmt(r['pent_over_hex_mean'], 10, '.4f')}  "
            f"{r['sum_ratio']:>10.8f}  {r['time_s']:>7.2f}"
        )

    print()
    print("Hex-area percentiles (relative to hex mean):")
    pcts = (1, 10, 25, 50, 75, 90, 99)
    pct_hdr = f"{'res':>3}  " + "  ".join(f"{p:>6}%" for p in pcts)
    print(pct_hdr)
    for res in range(1, MAX_RES + 1):
        hex_areas = [cell_area(c) for c in cells_at_res(res) if not is_pentagon(c)]
        h = np.array(hex_areas)
        vals = np.percentile(h / h.mean(), list(pcts))
        print(f"{res:>3}  " + "  ".join(f"{x:>6.4f}" for x in vals))


if __name__ == "__main__":
    main()
