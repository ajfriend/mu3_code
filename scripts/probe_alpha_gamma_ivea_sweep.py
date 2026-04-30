"""2D sweep over AlphaIVEA's (α, γ) — radial and angular cubic params.

Reports area_r, ang_p90 for each (α, γ) on the grid. Identifies points
that Pareto-dominate both plain IVEA (α=γ=1) and AlphaSlerp rich.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_projection_distortion import measure  # uses active_projection internally

from mu3.projection import AlphaIVEAProjection, AlphaSlerp, IVEAProjection


def main():
    alphas = np.linspace(0.85, 1.15, 11)
    gammas = np.linspace(0.85, 1.15, 11)

    grid = np.empty((len(alphas), len(gammas)), dtype=object)
    print(f"sweeping {len(alphas)} × {len(gammas)} = {len(alphas) * len(gammas)} "
          f"(α, γ) combinations…", flush=True)

    for i, a in enumerate(alphas):
        for j, g in enumerate(gammas):
            factory = partial(AlphaIVEAProjection, alpha=float(a), gamma=float(g))
            grid[i, j] = measure(factory)

    print("\narea_r grid (α rows, γ cols):", flush=True)
    print("       γ=" + "  ".join(f"{g:.2f}" for g in gammas))
    for i, a in enumerate(alphas):
        row = "  ".join(f"{grid[i, j]['area_r']:.3f}" for j in range(len(gammas)))
        print(f"α={a:.2f}  {row}")

    print("\nang_p90 grid:", flush=True)
    print("       γ=" + "  ".join(f"{g:.2f}" for g in gammas))
    for i, a in enumerate(alphas):
        row = "  ".join(f"{grid[i, j]['ang_p90']:5.2f}" for j in range(len(gammas)))
        print(f"α={a:.2f}  {row}")

    # Reference baselines
    ivea_m = measure(IVEAProjection)
    rich_m = measure(AlphaSlerp)
    print(f"\nReference IVEA              area_r={ivea_m['area_r']:.4f}  "
          f"ang_p90={ivea_m['ang_p90']:.3f}  ang_max={ivea_m['ang_dev']:.3f}")
    print(f"Reference AlphaSlerp rich   area_r={rich_m['area_r']:.4f}  "
          f"ang_p90={rich_m['ang_p90']:.3f}  ang_max={rich_m['ang_dev']:.3f}")

    # Pareto candidates
    print("\n--- best in each axis ---")
    rows = [(float(a), float(g), grid[i, j]) for i, a in enumerate(alphas)
            for j, g in enumerate(gammas)]
    best_area = min(rows, key=lambda r: r[2]['area_r'])
    best_p90 = min(rows, key=lambda r: r[2]['ang_p90'])
    best_max = min(rows, key=lambda r: r[2]['ang_dev'])
    bal = min(rows, key=lambda r:
        max((r[2]['area_r'] - 1.0) * 200.0, r[2]['ang_p90']))

    def fmt(label, r):
        a, g, m = r
        return (f"{label}: α={a:.3f}  γ={g:.3f}  "
                f"area_r={m['area_r']:.4f}  "
                f"ang_p90={m['ang_p90']:.3f}°  "
                f"ang_max={m['ang_dev']:.3f}°  "
                f"shape_mx={m['shape_mx']:.4f}")
    print(fmt("min area_r       ", best_area))
    print(fmt("min ang_p90°     ", best_p90))
    print(fmt("min ang_max°     ", best_max))
    print(fmt("balanced minmax  ", bal))

    # Pareto-dominance check vs rich
    print("\n--- Pareto vs AlphaSlerp rich ---")
    rich_area = rich_m['area_r']
    rich_p90 = rich_m['ang_p90']
    dominators = []
    for r in rows:
        a, g, m = r
        if m['area_r'] <= rich_area and m['ang_p90'] <= rich_p90:
            if m['area_r'] < rich_area or m['ang_p90'] < rich_p90:
                dominators.append(r)
    if dominators:
        print(f"Found {len(dominators)} (α, γ) Pareto-dominating rich on "
              f"(area_r ≤ {rich_area:.4f}, ang_p90 ≤ {rich_p90:.3f}):")
        for r in dominators:
            print("  " + fmt("", r).strip())
    else:
        print("No (α, γ) on this grid Pareto-dominates AlphaSlerp rich.")
        # Show the closest-to-Pareto point
        closest = min(rows, key=lambda r:
            max(0, r[2]['area_r'] - rich_area) * 100.0
            + max(0, r[2]['ang_p90'] - rich_p90))
        print(fmt("closest          ", closest))


if __name__ == "__main__":
    main()
