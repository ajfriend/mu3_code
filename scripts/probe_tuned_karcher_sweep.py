"""2D sweep over TunedKarcher's (η, κ) — find the optimum for the
Karcher construction (which is generally different from AlphaSlerp's).
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_projection_distortion import measure

from mu3.projection import (
    AlphaSlerp,
    KarcherProjection,
    TunedKarcherProjection,
)


def main():
    etas = np.linspace(-0.4, 0.4, 9)
    kappas = np.linspace(-0.4, 0.4, 9)

    print(f"sweeping {len(etas)} × {len(kappas)} = {len(etas) * len(kappas)} "
          f"(η, κ) combinations…", flush=True)
    grid = np.empty((len(etas), len(kappas)), dtype=object)
    for i, eta in enumerate(etas):
        for j, kap in enumerate(kappas):
            factory = partial(TunedKarcherProjection, eta=float(eta), kappa=float(kap))
            grid[i, j] = measure(factory)

    print("\narea_r grid (η rows, κ cols):", flush=True)
    print("        κ=" + "  ".join(f"{k:+.2f}" for k in kappas))
    for i, e in enumerate(etas):
        row = "  ".join(f"{grid[i, j]['area_r']:.3f}" for j in range(len(kappas)))
        print(f"η={e:+.2f}  {row}")

    print("\nshape_p50 grid:", flush=True)
    print("        κ=" + "  ".join(f"{k:+.2f}" for k in kappas))
    for i, e in enumerate(etas):
        row = "  ".join(f"{grid[i, j]['shape_p50']:.3f}" for j in range(len(kappas)))
        print(f"η={e:+.2f}  {row}")

    print("\nang_p50° grid:", flush=True)
    print("        κ=" + "  ".join(f"{k:+.2f}" for k in kappas))
    for i, e in enumerate(etas):
        row = "  ".join(f"{grid[i, j]['ang_p50']:5.2f}" for j in range(len(kappas)))
        print(f"η={e:+.2f}  {row}")

    # References
    karcher_m = measure(KarcherProjection)
    alpha_m = measure(AlphaSlerp)
    print(f"\nReference plain Karcher    area_r={karcher_m['area_r']:.4f}  "
          f"shape_p50={karcher_m['shape_p50']:.4f}  "
          f"ang_p50={karcher_m['ang_p50']:.3f}°")
    print(f"Reference AlphaSlerp rich  area_r={alpha_m['area_r']:.4f}  "
          f"shape_p50={alpha_m['shape_p50']:.4f}  "
          f"ang_p50={alpha_m['ang_p50']:.3f}°")

    # Best in each axis
    rows = [(float(e), float(k), grid[i, j])
            for i, e in enumerate(etas) for j, k in enumerate(kappas)]
    best_area = min(rows, key=lambda r: r[2]['area_r'])
    best_shape = min(rows, key=lambda r: r[2]['shape_p50'])
    best_ang = min(rows, key=lambda r: r[2]['ang_p50'])
    # Balanced: minimize 100·(area_r - 1) + ang_p90
    best_bal = min(rows, key=lambda r: 100 * (r[2]['area_r'] - 1.0) + r[2]['ang_p90'])

    def fmt(label, r):
        e, k, m = r
        return (f"{label}: η={e:+.3f}  κ={k:+.3f}  "
                f"area_r={m['area_r']:.4f}  "
                f"shape_p50={m['shape_p50']:.4f}  "
                f"ang_p50={m['ang_p50']:.3f}°  "
                f"ang_p90={m['ang_p90']:.3f}°")
    print("\n--- best in each axis ---")
    print(fmt("min area_r       ", best_area))
    print(fmt("min shape_p50    ", best_shape))
    print(fmt("min ang_p50°     ", best_ang))
    print(fmt("balanced minimax ", best_bal))

    # Pareto-dominance check vs plain Karcher
    print("\n--- Pareto vs plain Karcher ---")
    k_a, k_s, k_p = karcher_m['area_r'], karcher_m['shape_p50'], karcher_m['ang_p50']
    dominators = []
    for e, k, m in rows:
        if (m['area_r'] <= k_a and m['shape_p50'] <= k_s
                and m['ang_p50'] <= k_p):
            if (m['area_r'] < k_a or m['shape_p50'] < k_s
                    or m['ang_p50'] < k_p):
                dominators.append((e, k, m))
    if dominators:
        print(f"Found {len(dominators)} (η, κ) Pareto-dominating Karcher on "
              f"(area_r, shape_p50, ang_p50):")
        for d in dominators:
            print("  " + fmt("", d).strip())
    else:
        print("No (η, κ) on this grid Pareto-dominates plain Karcher.")


if __name__ == "__main__":
    main()
