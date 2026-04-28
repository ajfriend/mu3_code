# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Probe how low max-SVD-aspect (≈ angular distortion) can go in the
α-slerp family.

Aspect = σ_max / σ_min of the Jacobian. Geometric floor at the 12
pentagon corners is 1.2584 (independent of α, η, κ). Question: is the
INTERIOR max aspect close to this corner floor, or is there room to
push it down with smarter parameter choices?

Also report angular deviation: for a regular hexagon's interior angle
of 120° in the plane, the worst-case angle deviation under a linear
map with SVD ratio r (worst hex orientation) is

    delta(r) = | 120° - 2·arctan(sqrt(3)/r) |

At r = 1 → 0°. At r = 1.2584 → about 11.5°. So aspect_max maps
directly to a worst-case hex-vertex angle deviation.
"""

import sys
sys.path.insert(0, "src")

import math
import numpy as np
from scipy.optimize import minimize

from mu3 import icosahedron
from mu3.projection import AlphaSlerp


F = icosahedron.faces()
V = icosahedron.vertices()
v0, v1, v2 = V[F[0, 0]], V[F[0, 1]], V[F[0, 2]]
omega = math.acos(np.clip(np.dot(v0, v1), -1.0, 1.0))
sin_omega = math.sin(omega)
n_face = np.cross(v1 - v0, v2 - v0); n_face /= np.linalg.norm(n_face)
Vmat = np.stack([v0, v1, v2], axis=0)


def forward(alpha, eta, kappa, lam, mu, beta):
    w0 = omega - alpha
    b = np.asarray(beta, dtype=float)
    P = float(b[0] * b[1] * b[2])
    S = eta + kappa * (b - 1.0 / 3.0)
    if lam != 0.0 or mu != 0.0:
        cyc = np.array([b[1] * b[2], b[2] * b[0], b[0] * b[1]])
        S = S + lam * (cyc - 1.0 / 9.0) + mu * (b - 1.0 / 3.0) ** 2
    f = alpha * b + 3.0 * w0 * b * b - 2.0 * w0 * b * b * b
    weights = (1.0 + P * S) * np.sin(f) / sin_omega
    v_star = Vmat.T @ weights
    n_dot = float(np.dot(v_star, n_face))
    disc = 1.0 + n_dot * n_dot - float(np.dot(v_star, v_star))
    p = -n_dot + math.sqrt(max(disc, 0.0))
    out = v_star + p * n_face
    return out / np.linalg.norm(out)


def aspect_grid(alpha, eta, kappa, lam=0.0, mu=0.0, n_grid=30, h=1e-4):
    """Max and median Jacobian SVD aspect over interior grid.

    Computes ∂v/∂(x, y) where (x, y) are ORTHONORMAL Euclidean
    coordinates on the planar reference triangle (V₀ at origin, V₁ at
    (1, 0), V₂ at (0.5, √3/2)). Step purely in x: db₁ = h, db₂ = 0.
    Step purely in y: db₁ = -h/√3, db₂ = 2h/√3.
    """
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    aspects = []
    for i in range(2, n_grid - 1):
        for j in range(2, n_grid - i - 1):
            k = n_grid - i - j
            if k < 2:
                continue
            b1, b2 = i / n_grid, j / n_grid
            def F_(x, y):
                return forward(alpha, eta, kappa, lam, mu,
                               np.array([1.0 - x - y, x, y]))
            # d/dx in orthonormal (x, y): step (db1, db2) = (h, 0)
            dvx = (F_(b1 + h, b2) - F_(b1 - h, b2)) / (2 * h)
            # d/dy: step (db1, db2) = (-h/√3, 2h/√3)
            dvy = (F_(b1 - h * inv_sqrt3, b2 + 2 * h * inv_sqrt3)
                   - F_(b1 + h * inv_sqrt3, b2 - 2 * h * inv_sqrt3)) / (2 * h)
            J = np.stack([dvx, dvy], axis=1)  # 3x2
            sv = np.linalg.svd(J, compute_uv=False)
            aspects.append(sv[0] / sv[1])
    arr = np.array(aspects)
    return arr.max(), float(np.median(arr)), arr


def angle_dev_from_aspect(r):
    """Worst-case 120° hex vertex angle deviation under SVD aspect r."""
    return abs(120.0 - 2.0 * math.degrees(math.atan(math.sqrt(3) / r)))


def aspect_max(params, ansatz="3p"):
    if ansatz == "1p":
        a, e, k_, l_, m_ = params[0], 0.0, 0.0, 0.0, 0.0
    elif ansatz == "3p":
        a, e, k_ = params; l_, m_ = 0.0, 0.0
    elif ansatz == "4p":
        a, e, k_, l_ = params; m_ = 0.0
    elif ansatz == "5p":
        a, e, k_, l_, m_ = params
    mx, _, _ = aspect_grid(a, e, k_, l_, m_)
    return mx


def main():
    floor = 1.2584
    print(f"Geometric corner-aspect floor: {floor:.4f}")
    print(f"  → worst hex-vertex angle deviation at floor: "
          f"{angle_dev_from_aspect(floor):.2f}°")
    print()

    # Sanity at recommended params.
    a, e, k = 1.149, 0.121, 0.170
    mx, med, _ = aspect_grid(a, e, k)
    print(f"At published (1.149, 0.121, 0.170):")
    print(f"  aspect_max  = {mx:.4f}  (sampled grid)")
    print(f"  aspect_p50  = {med:.4f}")
    print(f"  worst angle = {angle_dev_from_aspect(mx):.2f}°")
    print()

    # 1-param.
    print("Optimizing 1-param (alpha) for aspect_max...")
    r1 = minimize(aspect_max, [1.142], args=("1p",), method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 500})
    print(f"  alpha={r1.x[0]:.10f}  aspect_max={r1.fun:.6f}  "
          f"angle={angle_dev_from_aspect(r1.fun):.2f}°")

    # 3-param.
    print("Optimizing 3-param for aspect_max...")
    r3 = minimize(aspect_max, [1.149, 0.121, 0.170], args=("3p",),
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 3000})
    print(f"  alpha={r3.x[0]:.6f} eta={r3.x[1]:.6f} kappa={r3.x[2]:.6f}")
    print(f"  aspect_max={r3.fun:.6f}  angle={angle_dev_from_aspect(r3.fun):.2f}°")

    # 4-param.
    print("Optimizing 4-param (+lambda) for aspect_max...")
    r4 = minimize(aspect_max, [*r3.x, 0.0], args=("4p",),
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 5000})
    print(f"  alpha={r4.x[0]:.6f} eta={r4.x[1]:.6f} kappa={r4.x[2]:.6f} "
          f"lam={r4.x[3]:.6f}")
    print(f"  aspect_max={r4.fun:.6f}  angle={angle_dev_from_aspect(r4.fun):.2f}°")

    # 5-param.
    print("Optimizing 5-param (+mu) for aspect_max...")
    r5 = minimize(aspect_max, [*r4.x, 0.0], args=("5p",),
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000})
    print(f"  alpha={r5.x[0]:.6f} eta={r5.x[1]:.6f} kappa={r5.x[2]:.6f}")
    print(f"  lam={r5.x[3]:.6f} mu={r5.x[4]:.6f}")
    print(f"  aspect_max={r5.fun:.6f}  angle={angle_dev_from_aspect(r5.fun):.2f}°")
    print()

    # Combined area + aspect: see if there's a better Pareto point than
    # the published one.
    def combined(params):
        a, e, k_ = params
        mx, _, _ = aspect_grid(a, e, k_)
        # Reuse the area-probe logic via Jacobian determinants
        from probe_alpha_slerp_floor import jac_dets
        dets = jac_dets(a, e, k_, n_grid=30)
        ar = float(dets.max() / dets.min())
        return 3.0 * (ar - 1.0) + max(mx - 1.253, 0.0)
    print("Optimizing 3-param under combined (3·area + (aspect-1.253)+) objective...")
    rc = minimize(combined, [1.149, 0.121, 0.170], method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 3000})
    a, e, k_ = rc.x
    from probe_alpha_slerp_floor import jac_dets
    dets = jac_dets(a, e, k_, n_grid=30)
    ar = float(dets.max() / dets.min())
    mx, _, _ = aspect_grid(a, e, k_)
    print(f"  alpha={a:.6f} eta={e:.6f} kappa={k_:.6f}")
    print(f"  area_ratio={ar:.6f}  aspect_max={mx:.6f}  "
          f"angle={angle_dev_from_aspect(mx):.2f}°")


if __name__ == "__main__":
    main()
