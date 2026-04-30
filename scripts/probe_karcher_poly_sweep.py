"""2D sweep over KarcherPoly's (κ₁, κ₂) — check if the quadratic
per-i term opens a Pareto-improvement region beyond plain Karcher.

The hypothesis: TunedKarcher (κ₁ alone) only moved along an existing
trade-off curve. With κ₂ added, we have a genuine new degree of
freedom; this might let us improve area without losing shape/angular,
or vice versa.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_projection_distortion import measure

from mu3.projection import (
    AlphaSlerp,
    KarcherPolyProjection,
    KarcherProjection,
)


def main():
    k1s = np.linspace(-0.5, 0.5, 11)
    k2s = np.linspace(-1.0, 1.0, 11)

    print(f"sweeping {len(k1s)} × {len(k2s)} = {len(k1s) * len(k2s)} "
          f"(κ₁, κ₂) combinations…", flush=True)
    grid = np.empty((len(k1s), len(k2s)), dtype=object)
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            factory = partial(KarcherPolyProjection,
                              kappas=(float(k1), float(k2)))
            grid[i, j] = measure(factory)

    print("\narea_r grid (κ₁ rows, κ₂ cols):", flush=True)
    print("        κ₂=" + "  ".join(f"{k:+.2f}" for k in k2s))
    for i, k in enumerate(k1s):
        row = "  ".join(f"{grid[i, j]['area_r']:.3f}" for j in range(len(k2s)))
        print(f"κ₁={k:+.2f}  {row}")

    print("\nshape_p50 grid:", flush=True)
    print("        κ₂=" + "  ".join(f"{k:+.2f}" for k in k2s))
    for i, k in enumerate(k1s):
        row = "  ".join(f"{grid[i, j]['shape_p50']:.3f}" for j in range(len(k2s)))
        print(f"κ₁={k:+.2f}  {row}")

    print("\nang_p50° grid:", flush=True)
    print("        κ₂=" + "  ".join(f"{k:+.2f}" for k in k2s))
    for i, k in enumerate(k1s):
        row = "  ".join(f"{grid[i, j]['ang_p50']:5.2f}" for j in range(len(k2s)))
        print(f"κ₁={k:+.2f}  {row}")

    karcher_m = measure(KarcherProjection)
    alpha_m = measure(AlphaSlerp)
    print(f"\nReference plain Karcher    "
          f"area_r={karcher_m['area_r']:.4f}  shape_p50={karcher_m['shape_p50']:.4f}  "
          f"ang_p50={karcher_m['ang_p50']:.3f}°")
    print(f"Reference AlphaSlerp rich  "
          f"area_r={alpha_m['area_r']:.4f}  shape_p50={alpha_m['shape_p50']:.4f}  "
          f"ang_p50={alpha_m['ang_p50']:.3f}°")

    rows = [(float(k1), float(k2), grid[i, j])
            for i, k1 in enumerate(k1s) for j, k2 in enumerate(k2s)]
    best_area = min(rows, key=lambda r: r[2]['area_r'])
    best_shape = min(rows, key=lambda r: r[2]['shape_p50'])
    best_ang = min(rows, key=lambda r: r[2]['ang_p50'])
    best_bal = min(rows, key=lambda r: 100 * (r[2]['area_r'] - 1.0) + r[2]['ang_p90'])

    def fmt(label, r):
        k1, k2, m = r
        return (f"{label}: κ₁={k1:+.3f}  κ₂={k2:+.3f}  "
                f"area_r={m['area_r']:.4f}  "
                f"shape_p50={m['shape_p50']:.4f}  "
                f"ang_p50={m['ang_p50']:.3f}°  "
                f"ang_p90={m['ang_p90']:.3f}°")
    print("\n--- best in each axis ---")
    print(fmt("min area_r       ", best_area))
    print(fmt("min shape_p50    ", best_shape))
    print(fmt("min ang_p50°     ", best_ang))
    print(fmt("balanced minimax ", best_bal))

    # Pareto vs plain Karcher
    k_a, k_s, k_p = (karcher_m['area_r'], karcher_m['shape_p50'],
                     karcher_m['ang_p50'])
    dominators = []
    for k1, k2, m in rows:
        if (m['area_r'] <= k_a and m['shape_p50'] <= k_s
                and m['ang_p50'] <= k_p):
            if (m['area_r'] < k_a or m['shape_p50'] < k_s
                    or m['ang_p50'] < k_p):
                dominators.append((k1, k2, m))
    print("\n--- Pareto vs plain Karcher (area, shape_p50, ang_p50) ---")
    if dominators:
        print(f"Found {len(dominators)} (κ₁, κ₂) Pareto-dominating Karcher:")
        for d in dominators[:10]:
            print("  " + fmt("", d).strip())
    else:
        print("No (κ₁, κ₂) on this grid Pareto-dominates plain Karcher.")


if __name__ == "__main__":
    main()
