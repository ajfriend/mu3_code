"""Side-by-side comparison: H3 Cagnoli (lat/lng) vs VOS-with-chord-identities (3D).

Two implementations of the same per-edge spherical polygon area
formula, applied to mu3 hex cells at res 5..20. Cagnoli is the
current production implementation in ``cell.py``; VOS-chord is the
candidate replacement. Both should match to numerical noise.

Validation steps:

1. Cell-by-cell comparison at res 5: max absolute / relative
   diff across ~100k hex cells.
2. ``area_r`` agreement at res 5, 10, 15, 18, 20 for both
   parameter sets (literature, discrete-3).
3. Antipodal stress: tiny hex polygon centered exactly at the north
   pole (the antipode of our fixed fan reference at south pole).
   Both formulas should agree, both should match the analytic
   spherical-cap area to f64 precision.
"""
import math
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from probe_projection_distortion import active_projection
from probe_alpha_slerp_high_res_area_r import structural_cells
from mu3 import cell_boundary, is_pentagon, is_valid_cell, cells_at_res
from mu3.projection import AlphaSlerpExtended


def cagnoli_area(V: np.ndarray) -> float:
    """Current production: H3 Cagnoli per-edge via lat/lng."""
    n = len(V)
    lat = np.empty(n)
    lng = np.empty(n)
    for i in range(n):
        v = V[i]
        lat[i] = math.asin(max(-1.0, min(1.0, float(v[2]))))
        lng[i] = math.atan2(float(v[1]), float(v[0]))
    phi = lat * 0.5 + math.pi / 4.0
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sum_val = 0.0
    c = 0.0
    for i in range(n):
        j = (i + 1) % n
        sa = float(sin_phi[i] * sin_phi[j])
        ca = float(cos_phi[i] * cos_phi[j])
        d = float(lng[j] - lng[i])
        sd = math.sin(d)
        cd = math.cos(d)
        contribution = -2.0 * math.atan2(sa * sd, sa * cd + ca)
        y = contribution - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


def vos_chord_area(V: np.ndarray) -> float:
    """Candidate A: per-edge VOS with chord identities, FIXED south
    pole reference. Has antipodal failure mode for polygons near +N.
    """
    PX, PY, PZ = 0.0, 0.0, -1.0
    n = len(V)
    sum_val = 0.0
    c = 0.0
    for i in range(n):
        j = (i + 1) % n
        ax, ay, az = float(V[i][0]), float(V[i][1]), float(V[i][2])
        bx, by, bz = float(V[j][0]), float(V[j][1]), float(V[j][2])
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        cx = ay * dz - az * dy
        cy = az * dx - ax * dz
        cz = ax * dy - ay * dx
        num = PX * cx + PY * cy + PZ * cz
        pa = PX * ax + PY * ay + PZ * az
        pb = PX * bx + PY * by + PZ * bz
        d2 = dx * dx + dy * dy + dz * dz
        den = (1.0 + pa) + (1.0 + pb) - 0.5 * d2
        contribution = 2.0 * math.atan2(num, den)
        y = contribution - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


def vos_chord_stable_fixed_area(V: np.ndarray) -> float:
    """Candidate C: per-edge VOS with chord identities + the
    `onePlusDotStable` rewrite for (1 + P·V), with FIXED south pole P.

    Does the stable rewrite alone rescue the fixed-pole version?
    (1 + P·V) ≡ 0.5 · |P + V|² for unit P, V — algebraically exact,
    avoids the 1 + (near -1) cancellation when V is near -P.
    """
    PX, PY, PZ = 0.0, 0.0, -1.0
    n = len(V)

    def one_plus_dot(vx, vy, vz):
        x = PX + vx; y = PY + vy; z = PZ + vz
        return 0.5 * (x * x + y * y + z * z)

    sum_val = 0.0
    c = 0.0
    for i in range(n):
        j = (i + 1) % n
        ax, ay, az = float(V[i][0]), float(V[i][1]), float(V[i][2])
        bx, by, bz = float(V[j][0]), float(V[j][1]), float(V[j][2])
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        cx = ay * dz - az * dy
        cy = az * dx - ax * dz
        cz = ax * dy - ay * dx
        num = PX * cx + PY * cy + PZ * cz
        opa = one_plus_dot(ax, ay, az)
        opb = one_plus_dot(bx, by, bz)
        d2 = dx * dx + dy * dy + dz * dz
        den = opa + opb - 0.5 * d2
        contribution = 2.0 * math.atan2(num, den)
        y = contribution - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


def vos_chord_v0_area(V: np.ndarray) -> float:
    """Candidate D: VOS with chord identities, P = V[0] (first vertex).

    For polygons where no vertex is near the antipode of V[0]
    (always true for small mu3 cells), this works and saves the
    centroid normalize. Edges (V[0], V[1]) and (V[n-1], V[0])
    contribute 0 because num = V[0]·(V[0] × D) = 0 exactly. Net
    effect: per-fan triangulation around V[0] with the chord-
    identity denominator.
    """
    PX, PY, PZ = float(V[0][0]), float(V[0][1]), float(V[0][2])
    n = len(V)
    sum_val = 0.0
    c = 0.0
    for i in range(n):
        j = (i + 1) % n
        ax, ay, az = float(V[i][0]), float(V[i][1]), float(V[i][2])
        bx, by, bz = float(V[j][0]), float(V[j][1]), float(V[j][2])
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        cx = ay * dz - az * dy
        cy = az * dx - ax * dz
        cz = ax * dy - ay * dx
        num = PX * cx + PY * cy + PZ * cz
        pa = PX * ax + PY * ay + PZ * az
        pb = PX * bx + PY * by + PZ * bz
        d2 = dx * dx + dy * dy + dz * dz
        den = (1.0 + pa) + (1.0 + pb) - 0.5 * d2
        contribution = 2.0 * math.atan2(num, den)
        y = contribution - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


def vos_chord_two_pole_area(V: np.ndarray) -> float:
    """Candidate E (the user's idea): two fixed-pole references with
    per-edge selection plus a lune correction at switches.

    For each edge, pick P_N or P_S so that neither vertex is near
    antipode of the chosen pole. When P_S is used, add 2·Δλ
    (longitude difference along the edge) to convert the contribution
    back to the "P_N convention". Δλ is computed directly from 3D
    coords as atan2(A_x B_y − A_y B_x, A_x B_x + A_y B_y).

    Kahan-feeds the per-edge VOS contribution AND the per-edge
    lune correction as separate terms — adding them together
    pre-Kahan would cancel away most of the residual precision when
    the two have similar magnitudes with opposite signs (high-res
    cells far from the equator).
    """
    n = len(V)
    # Two independent Kahan accumulators: one for the per-edge VOS
    # contributions, one for the lune corrections. Adding them
    # together pre-Kahan would mix scales and grow intermediate
    # values; keeping them separate bounds each running sum by its
    # own natural scale (per-edge term magnitude, not their sum).
    vos_sum = 0.0
    c_vos = 0.0
    lune_sum = 0.0
    c_lune = 0.0
    for i in range(n):
        j = (i + 1) % n
        ax, ay, az = float(V[i][0]), float(V[i][1]), float(V[i][2])
        bx, by, bz = float(V[j][0]), float(V[j][1]), float(V[j][2])

        if az + bz >= 0.0:
            PX, PY, PZ = 0.0, 0.0, 1.0
            used_S = False
        else:
            PX, PY, PZ = 0.0, 0.0, -1.0
            used_S = True

        dx = bx - ax
        dy = by - ay
        dz = bz - az
        cx = ay * dz - az * dy
        cy = az * dx - ax * dz
        cz = ax * dy - ay * dx
        num = PX * cx + PY * cy + PZ * cz
        pa = PX * ax + PY * ay + PZ * az
        pb = PX * bx + PY * by + PZ * bz
        d2 = dx * dx + dy * dy + dz * dz
        den = (1.0 + pa) + (1.0 + pb) - 0.5 * d2
        vos_term = 2.0 * math.atan2(num, den)

        # Kahan into VOS accumulator.
        y = vos_term - c_vos
        t = vos_sum + y
        c_vos = (t - vos_sum) - y
        vos_sum = t

        if used_S:
            d_lambda = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
            lune_term = 2.0 * d_lambda
            # Kahan into lune accumulator.
            y = lune_term - c_lune
            t = lune_sum + y
            c_lune = (t - lune_sum) - y
            lune_sum = t

    # Combine the two Kahan-summed values. Both are ~polygon_area-
    # scale (vos_sum) or ~0 (lune_sum) for closed non-pole-enclosing
    # polygons; this final add doesn't introduce new cancellation.
    sum_val = vos_sum + lune_sum
    c = c_vos + c_lune

    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


def vos_chord_centroid_area(V: np.ndarray) -> float:
    """Candidate B: per-edge VOS with chord identities + polygon-centroid
    fan reference. Per-polygon overhead: one normalize. Avoids the
    antipodal precision loss in (1 + P·V) by ensuring P is inside the
    polygon, so all (1 + P·V) ≈ 2.
    """
    n = len(V)
    # Centroid of the polygon (sum of vertices, normalized).
    sx = sy = sz = 0.0
    for v in V:
        sx += float(v[0]); sy += float(v[1]); sz += float(v[2])
    norm = math.sqrt(sx * sx + sy * sy + sz * sz)
    PX, PY, PZ = sx / norm, sy / norm, sz / norm

    sum_val = 0.0
    c = 0.0
    for i in range(n):
        j = (i + 1) % n
        ax, ay, az = float(V[i][0]), float(V[i][1]), float(V[i][2])
        bx, by, bz = float(V[j][0]), float(V[j][1]), float(V[j][2])
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        cx = ay * dz - az * dy
        cy = az * dx - ax * dz
        cz = ax * dy - ay * dx
        num = PX * cx + PY * cy + PZ * cz
        pa = PX * ax + PY * ay + PZ * az
        pb = PX * bx + PY * by + PZ * bz
        d2 = dx * dx + dy * dy + dz * dz
        den = (1.0 + pa) + (1.0 + pb) - 0.5 * d2
        contribution = 2.0 * math.atan2(num, den)
        y = contribution - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    if sum_val < 0.0:
        y = 4.0 * math.pi - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val


# ----- Step 1: cell-by-cell at res 5 -----

def step_1_cell_by_cell(factory, res=5):
    print(f"\n=== Step 1: cell-by-cell comparison at res {res} (vs cagnoli) ===")
    with active_projection(factory):
        results = {"vos-chord (fixed P)": (vos_chord_area, [0.0, 0.0, 0]),
                   "vos-chord (fixed P, stable 1+P·V)": (vos_chord_stable_fixed_area, [0.0, 0.0, 0]),
                   "vos-chord (P = V[0])": (vos_chord_v0_area, [0.0, 0.0, 0]),
                   "vos-chord (centroid P)": (vos_chord_centroid_area, [0.0, 0.0, 0]),
                   "vos-chord (two-pole + lune)": (vos_chord_two_pole_area, [0.0, 0.0, 0])}
        n_total = 0
        for cell in cells_at_res(res):
            if is_pentagon(cell):
                continue
            V = cell_boundary(cell, closed=False)
            a_old = cagnoli_area(V)
            for label, (fn, stats) in results.items():
                a_new = fn(V)
                diff = abs(a_old - a_new)
                rel = diff / a_old if a_old > 0 else 0.0
                stats[0] = max(stats[0], diff)
                stats[1] = max(stats[1], rel)
                if rel > 1e-12:
                    stats[2] += 1
            n_total += 1
        print(f"  cells compared: {n_total}")
        for label, (_, stats) in results.items():
            print(f"  {label:<26s} max_abs={stats[0]:.3e}  max_rel={stats[1]:.3e}  rel>1e-12: {stats[2]}")
        return results["vos-chord (centroid P)"][1][1]


# ----- Step 2: area_r at multiple resolutions -----

def step_2_area_r_table(factories):
    print("\n=== Step 2: area_r at res 5, 10, 15, 18, 20 ===")
    for name, factory in factories.items():
        print(f"\nparameter set: {name}")
        print(f"  {'formula':<14s} {'res 5':>10s} {'res 10':>10s} {'res 15':>10s} {'res 18':>10s} {'res 20':>10s}")
        with active_projection(factory):
            for label, area_fn in [("cagnoli",                cagnoli_area),
                                   ("vos-chord (fixed)",      vos_chord_area),
                                   ("vos-chord (fix+stab)",   vos_chord_stable_fixed_area),
                                   ("vos-chord (P=V[0])",     vos_chord_v0_area),
                                   ("vos-chord (cent.)",      vos_chord_centroid_area),
                                   ("vos-chord (2pole+lune)", vos_chord_two_pole_area)]:
                row = [f"  {label:<14s}"]
                for res in [5, 10, 15, 18, 20]:
                    cells = structural_cells(res)
                    areas = []
                    for cell in cells:
                        if is_pentagon(cell):
                            continue
                        try:
                            V = cell_boundary(cell, closed=False)
                            areas.append(abs(area_fn(V)))
                        except Exception:
                            pass
                    ar = max(areas) / min(areas) if areas else float('nan')
                    row.append(f" {ar:>9.6f}")
                print("".join(row), flush=True)


# ----- Step 3: antipodal stress -----

def hex_polygon_at_north_pole(angular_radius: float) -> np.ndarray:
    """Equilateral hex on the unit sphere centered at the north pole,
    angular radius ``r`` (rad). CCW from outside the sphere (looking
    down along +z).
    """
    r = angular_radius
    pts = []
    for k in range(6):
        theta = 2.0 * math.pi * k / 6.0
        x = math.sin(r) * math.cos(theta)
        y = math.sin(r) * math.sin(theta)
        z = math.cos(r)
        pts.append((x, y, z))
    return np.array(pts)


def step_3_antipodal_stress():
    print("\n=== Step 3: antipodal stress (hex centered at north pole = -P_fixed) ===")
    print(f"  {'r (rad)':>10s} {'analytic':>10s} {'cagnoli':>10s} {'fix':>10s} {'fix+stab':>10s} {'P=V[0]':>10s} {'centroid':>10s}"
          f"  {'r_cag':>7s} {'r_fix':>7s} {'r_stab':>7s} {'r_v0':>7s} {'r_cen':>7s}")
    for r in [1e-3, 1e-6, 1e-9, 1e-12]:
        V = hex_polygon_at_north_pole(r)
        analytic = 1.5 * math.sqrt(3.0) * (r ** 2)
        a_cag  = cagnoli_area(V)
        a_fix  = vos_chord_area(V)
        a_stab = vos_chord_stable_fixed_area(V)
        a_v0   = vos_chord_v0_area(V)
        a_cen  = vos_chord_centroid_area(V)
        r_cag  = abs(a_cag - analytic) / analytic
        r_fix  = abs(a_fix - analytic) / analytic
        r_stab = abs(a_stab - analytic) / analytic
        r_v0   = abs(a_v0 - analytic) / analytic
        r_cen  = abs(a_cen - analytic) / analytic
        print(f"  {r:>10.0e} {analytic:>10.3e} {a_cag:>10.3e} {a_fix:>10.3e} {a_stab:>10.3e} {a_v0:>10.3e} {a_cen:>10.3e}"
              f"  {r_cag:>7.1e} {r_fix:>7.1e} {r_stab:>7.1e} {r_v0:>7.1e} {r_cen:>7.1e}", flush=True)


def step_4_pole_spanning_test():
    """A 'tall' polygon spanning both pole regions — V[0] is far from
    some other vertex, so V[0] reference loses precision. The two-pole
    + lune correction should still work."""
    print("\n=== Step 4: tall pole-spanning polygon (V[0] reference is challenged) ===")
    # 4-vertex CCW polygon: two near north pole, two near south pole.
    # Forms a thin "strap" from pole to pole, of small longitudinal width.
    eps = 1e-3
    dlng = 0.1  # rad
    V_tall = np.array([
        # near north pole, λ = 0
        [eps,                0.0,                math.sqrt(1 - eps**2)],
        # near south pole, λ = 0
        [eps,                0.0,               -math.sqrt(1 - eps**2)],
        # near south pole, λ = dlng
        [eps * math.cos(dlng), eps * math.sin(dlng), -math.sqrt(1 - eps**2)],
        # near north pole, λ = dlng
        [eps * math.cos(dlng), eps * math.sin(dlng),  math.sqrt(1 - eps**2)],
    ])
    # Analytic: this is approximately a (very thin) lune of dihedral
    # angle dlng = 0.1 rad → area ≈ 2·dlng = 0.2 sr (minus end caps).
    # For eps small, the end-caps shrink as eps² so the lune dominates.
    analytic_lune = 2.0 * dlng
    print(f"  analytic lune (eps→0): {analytic_lune:.6f} sr")
    for label, fn in [("cagnoli",      cagnoli_area),
                      ("V[0]",         vos_chord_v0_area),
                      ("centroid",     vos_chord_centroid_area),
                      ("two-pole+lune", vos_chord_two_pole_area)]:
        try:
            a = fn(V_tall)
        except Exception as e:
            a = float('nan')
            print(f"  {label:<14s}: FAILED ({e})")
            continue
        rel = abs(a - analytic_lune) / analytic_lune
        print(f"  {label:<14s}: {a:.10f}  (rel diff vs analytic lune: {rel:.2e})")


PARAM_SETS = {
    "literature":  (1.149, 0.121, 0.170, 0.0, 0.0),
    "discrete-3":  (1.14952, 0.10836, 0.18578, 0.00020, -0.00002),
}

factories = {}
for name, params in PARAM_SETS.items():
    a, e, k, l, m = params
    factories[name] = partial(AlphaSlerpExtended, alpha=a, eta=e,
                              kappa=k, lambd=l, mu=m)


t0 = time.time()
literature_factory = factories["literature"]
max_rel = step_1_cell_by_cell(literature_factory, res=5)
step_2_area_r_table(factories)
step_3_antipodal_stress()
step_4_pole_spanning_test()
print(f"\nTotal time: {time.time() - t0:.1f}s")
print(f"\n=> max relative diff (cell-by-cell at res 5): {max_rel:.3e}")
print("   Threshold for swap: max rel diff < 1e-10.")
