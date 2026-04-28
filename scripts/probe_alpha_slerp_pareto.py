# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Sweep the area-vs-aspect Pareto frontier in the 3-param α-slerp family.

Objective: minimize  area_excess + λ · aspect_excess
  area_excess   = max(area_ratio - 1, 0)
  aspect_excess = max(aspect_max - 1.2584, 0)

λ → 0 recovers pure-area optimum.
λ → ∞ recovers pure-aspect optimum.

For each λ in a sweep, run NM and report (area_ratio, aspect_max,
worst hex angle).
"""

import sys
sys.path.insert(0, "src")

import math
import numpy as np
from scipy.optimize import minimize

from probe_alpha_slerp_aspect import aspect_grid, angle_dev_from_aspect
from probe_alpha_slerp_floor import jac_dets


def objective(params, lam_weight):
    a, e, k_ = params
    dets = jac_dets(a, e, k_, n_grid=20)
    ar = float(dets.max() / dets.min())
    mx, _, _ = aspect_grid(a, e, k_, n_grid=20)
    return (ar - 1.0) + lam_weight * (mx - 1.0)


def main():
    print(f"{'lambda':>10}  {'alpha':>10}  {'eta':>10}  {'kappa':>10}  "
          f"{'area_r':>10}  {'asp_max':>8}  {'angle':>7}")
    x0 = [1.149, 0.121, 0.170]
    for lam in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        res = minimize(objective, x0, args=(lam,), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-8, "maxiter": 2000})
        a, e, k_ = res.x
        dets = jac_dets(a, e, k_, n_grid=20)
        ar = float(dets.max() / dets.min())
        mx, _, _ = aspect_grid(a, e, k_, n_grid=20)
        ang = angle_dev_from_aspect(mx)
        print(f"{lam:>10.2f}  {a:>10.6f}  {e:>10.6f}  {k_:>10.6f}  "
              f"{ar:>10.6f}  {mx:>8.4f}  {ang:>6.2f}°")
        # Use this as warm-start for next lambda (continuation).
        x0 = list(res.x)


if __name__ == "__main__":
    main()
