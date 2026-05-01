"""Compare AlphaSlerp parameter sets at res 5, 7, 10, 15, 20 using a
structural cell subset that captures both area_r extremes.

The user's hypothesis (verified empirically at res 3, 4, 5):

  - Smallest hexes are pentagon-adjacent cells at every hierarchy
    level (NOT just at the same resolution as the pentagon).
  - Largest hexes are at face centroids (= dodec vertices = icosa
    face centers), reached via zigzag digit walks like (d, d-1, d, d-1).

Structural subset: for each level k = 1..res, the 60 cells with a
single non-zero digit at level k (= pentagon ring-1 at hierarchy
level k); plus 12·5·2 = 120 zigzag walkers. Total: 60·N + 121 cells
at res N. At res 20: 1309 cells vs ~10^17 cells for full enumeration.

Validated against full enumeration at res 3-5: structural slightly
underestimates (≤2e-4) because some deeper-pattern extreme cells
aren't included, but the underestimate is the same order across
parameter sets — relative comparisons are reliable.

NOTE: at res ~20, ``cell_area`` (spherical polygon via van Oosterom)
hits f64 precision floor — hex cells are ~3 cm on a unit sphere,
cell-vertex separations are ~5e-9 rad, the ``1 - V_i·V_j``
cancellation kills 16+ digits in the denominator. Need a more
numerically stable area formula for high-res work; separate fix.
"""
import sys
import time
from functools import partial

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from probe_projection_distortion import active_projection
from mu3 import cell_area, is_pentagon, is_valid_cell
from mu3.projection import AlphaSlerpExtended


def structural_cells(res):
    cells = []
    cells.append(tuple([0] + [0] * res))
    for k in range(1, res + 1):
        for base in range(12):
            for d in (2, 3, 4, 5, 6):
                digits = [0] * res
                digits[k - 1] = d
                cell = tuple([base] + digits)
                if is_valid_cell(cell):
                    cells.append(cell)
    for base in range(12):
        for d in (2, 3, 4, 5, 6):
            for start_offset in (0, 1):
                digits = [d if (k + start_offset) % 2 == 0 else (d - 1)
                          for k in range(res)]
                first_nz = next((x for x in digits if x != 0), None)
                if first_nz == 1:
                    continue
                cell = tuple([base] + digits)
                if is_valid_cell(cell):
                    cells.append(cell)
    return cells


def measure_area_r_structural(factory, res):
    with active_projection(factory):
        cells = structural_cells(res)
        hex_areas = []
        for cell in cells:
            if is_pentagon(cell):
                continue
            try:
                hex_areas.append(abs(cell_area(cell)))
            except Exception:
                pass
        return max(hex_areas) / min(hex_areas) if hex_areas else float('nan')


PARAM_SETS = {
    "literature":  (1.149, 0.121, 0.170, 0.0, 0.0),
    "discrete-3":  (1.14952, 0.10836, 0.18578, 0.00020, -0.00002),
}


print(f"{'name':<12s} {'res':>4s} {'area_r':>10s} {'time_s':>8s} {'n_cells':>9s}")
for name, params in PARAM_SETS.items():
    a, e, k, l, m = params
    factory = partial(AlphaSlerpExtended, alpha=a, eta=e, kappa=k, lambd=l, mu=m)
    for res in [5, 7, 10, 15, 20]:
        t0 = time.time()
        ar = measure_area_r_structural(factory, res)
        dt = time.time() - t0
        n_cells = len(structural_cells(res))
        print(f"{name:<12s} {res:>4d} {ar:>10.6f} {dt:>8.3f} {n_cells:>9d}",
              flush=True)
