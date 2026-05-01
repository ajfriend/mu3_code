# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Higher-order AlphaSlerp parameter sweep — does the 3-param plateau
extend to discrete-cell metrics?

Two complementary searches:

1. 2D (λ, μ) grid at fixed (α, η, κ) = (1.149, 0.121, 0.170). Confirms
   whether the higher-order knobs have any effect on our metric at the
   3-param optimum.

2. 5D Nelder-Mead optimization over (α, η, κ, λ, μ) with multiple
   seeds. Mirrors sister-repo's experiment_5param.py methodology, but
   against our discrete-cell area_r objective at res 3.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "src")  # for mu3 (matches probe_alpha_slerp_floor.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling probe

from probe_projection_distortion import measure

from mu3.projection import (
    ALPHA_SLERP_DEFAULTS,
    AlphaSlerp,
    AlphaSlerpExtended,
)


ALPHA0, ETA0, KAPPA0 = ALPHA_SLERP_DEFAULTS  # (1.149, 0.121, 0.170)


# ---------------------------------------------------------------------------
# Phase 1: 2D (λ, μ) grid at the 3-param optimum
# ---------------------------------------------------------------------------

def grid_2d():
    lambdas = np.linspace(-0.5, 0.5, 11)
    mus = np.linspace(-0.5, 0.5, 11)

    print(f"Phase 1: 2D (λ, μ) grid — {len(lambdas)}×{len(mus)} = "
          f"{len(lambdas) * len(mus)} evals at (α, η, κ) = "
          f"({ALPHA0}, {ETA0}, {KAPPA0})", flush=True)

    grid = np.empty((len(lambdas), len(mus)), dtype=object)
    for i, lambd in enumerate(lambdas):
        for j, mu in enumerate(mus):
            factory = partial(AlphaSlerpExtended,
                              alpha=ALPHA0, eta=ETA0, kappa=KAPPA0,
                              lambd=float(lambd), mu=float(mu))
            grid[i, j] = measure(factory)

    print("\narea_r grid (λ rows, μ cols):", flush=True)
    print("        μ=" + "  ".join(f"{m:+.2f}" for m in mus))
    for i, ld in enumerate(lambdas):
        row = "  ".join(f"{grid[i, j]['area_r']:.4f}" for j in range(len(mus)))
        print(f"λ={ld:+.2f}  {row}")

    rows = [(float(ld), float(m), grid[i, j])
            for i, ld in enumerate(lambdas) for j, m in enumerate(mus)]

    plain = measure(AlphaSlerp)
    print(f"\nPlain AlphaSlerp: area_r = {plain['area_r']:.6f}, "
          f"shape_p50 = {plain['shape_p50']:.4f}, "
          f"ang_p50 = {plain['ang_p50']:.3f}°")

    best_area = min(rows, key=lambda r: r[2]['area_r'])
    print(f"\nBest area_r in grid: λ={best_area[0]:+.3f}, μ={best_area[1]:+.3f}  "
          f"area_r = {best_area[2]['area_r']:.6f}  "
          f"(plain: {plain['area_r']:.6f}; "
          f"Δ = {best_area[2]['area_r'] - plain['area_r']:+.5f})")

    return grid, rows, plain


# ---------------------------------------------------------------------------
# Phase 2: 5D Nelder-Mead optimization
# ---------------------------------------------------------------------------

def objective_area_r(params):
    """Discrete-cell area_r at res 3, for use as scipy minimize target."""
    alpha, eta, kappa, lambd, mu = params
    if alpha < 0.5 or alpha > 1.5:  # safety bounds
        return 1.0e6
    factory = partial(AlphaSlerpExtended,
                      alpha=float(alpha), eta=float(eta), kappa=float(kappa),
                      lambd=float(lambd), mu=float(mu))
    try:
        m = measure(factory)
        return float(m['area_r'])
    except Exception as e:
        return 1.0e6


def nm_optimize():
    print("\n" + "=" * 60)
    print("Phase 2: 5D Nelder-Mead optimization on area_r")
    print("=" * 60, flush=True)

    seeds = [
        ("default",      np.array([ALPHA0, ETA0, KAPPA0, 0.0, 0.0])),
        ("perturbed",    np.array([ALPHA0, ETA0, KAPPA0, 0.1, 0.05])),
        ("alt-α",        np.array([1.10, 0.10, 0.15, 0.0, 0.0])),
        ("aggressive",   np.array([1.149, 0.121, 0.170, 0.3, -0.2])),
    ]

    best = None
    for name, x0 in seeds:
        print(f"\nSeed '{name}': {x0}")
        res = minimize(objective_area_r, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-7,
                                "maxiter": 200, "disp": False})
        print(f"  → α={res.x[0]:+.5f}, η={res.x[1]:+.5f}, κ={res.x[2]:+.5f}, "
              f"λ={res.x[3]:+.5f}, μ={res.x[4]:+.5f}")
        print(f"    area_r = {res.fun:.6f}  ({res.nit} iters, "
              f"{res.nfev} evals)")
        if best is None or res.fun < best.fun:
            best = res
            best.seed_name = name

    return best


# ---------------------------------------------------------------------------
# Phase 3: Verdict
# ---------------------------------------------------------------------------

def verdict(plain, best_grid_row, best_nm):
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    plain_area = plain['area_r']
    grid_best = best_grid_row[2]['area_r']
    nm_best = best_nm.fun

    print(f"\nPlain AlphaSlerp (α, η, κ) = ({ALPHA0}, {ETA0}, {KAPPA0})")
    print(f"  area_r = {plain_area:.6f}")

    print(f"\nBest in 2D (λ, μ) grid at fixed (α, η, κ):")
    print(f"  λ={best_grid_row[0]:+.3f}, μ={best_grid_row[1]:+.3f}  →  "
          f"area_r = {grid_best:.6f}")
    print(f"  Δ vs plain: {grid_best - plain_area:+.5f} "
          f"({100 * (grid_best - plain_area) / plain_area:+.3f}%)")

    print(f"\nBest from 5D Nelder-Mead (seed: {best_nm.seed_name}):")
    print(f"  α={best_nm.x[0]:+.5f}, η={best_nm.x[1]:+.5f}, "
          f"κ={best_nm.x[2]:+.5f}, λ={best_nm.x[3]:+.5f}, μ={best_nm.x[4]:+.5f}")
    print(f"  area_r = {nm_best:.6f}")
    print(f"  Δ vs plain: {nm_best - plain_area:+.5f} "
          f"({100 * (nm_best - plain_area) / plain_area:+.3f}%)")

    improvement = (plain_area - nm_best) / plain_area
    if improvement < 0.01:  # less than 1% improvement
        print(f"\n→ PLATEAU CONFIRMED. 5D NM finds <{improvement * 100:.2f}% "
              f"area_r improvement over 3-param baseline. "
              f"AlphaSlerp's 3-parameter form is at a discrete-cell plateau, "
              f"matching the sister-repo's continuous-Jacobian finding.")
    elif improvement < 0.05:
        print(f"\n→ SMALL PARETO WIN. 5D NM finds {improvement * 100:.2f}% "
              f"area_r improvement. Worth documenting; not a transformative gain.")
    else:
        print(f"\n→ PLATEAU PIERCED. 5D NM finds {improvement * 100:.2f}% "
              f"area_r improvement. Higher-order parameters substantially "
              f"improve discrete-cell area uniformity. Worth promoting.")


def main():
    grid, rows, plain = grid_2d()
    best_grid_row = min(rows, key=lambda r: r[2]['area_r'])
    best_nm = nm_optimize()
    verdict(plain, best_grid_row, best_nm)


if __name__ == "__main__":
    main()
