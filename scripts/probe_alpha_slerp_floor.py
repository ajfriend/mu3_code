# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Probe whether the 3-param α-slerp family contains an exact equal-area member.

Uses mu3.projection.AlphaSlerp directly so the Jacobian we measure is
exactly the map mu3 ships. The "Jacobian determinant" is computed as
|∂v/∂β₁ × ∂v/∂β₂| via central finite differences on a fixed (β₁, β₂)
grid covering the face interior. The area_ratio metric is max/min over
that grid.

If the family's true minimum is 1.0 (achievable in closed form),
Nelder-Mead will find a point with area_ratio - 1 below numerical
noise. If 1.001 is a genuine in-family floor, we'll see it stop there
no matter how many parameters we add.
"""

import sys
sys.path.insert(0, "src")

import math
import numpy as np
from scipy.optimize import minimize

from mu3 import icosahedron
from mu3.projection import AlphaSlerp


# Standard icosa face vertices: pick face 0 from the canonical table.
F = icosahedron.faces()
V = icosahedron.vertices()
v0, v1, v2 = V[F[0, 0]], V[F[0, 1]], V[F[0, 2]]


def make_alpha_slerp(alpha, eta, kappa):
    return AlphaSlerp(v0, v1, v2, alpha=alpha, eta=eta, kappa=kappa)


# A custom forward that adds optional 4th/5th params (lam, mu) on top
# of the 3-param ansatz. Identical to AlphaSlerp.forward_barycentric
# when lam = mu = 0.
def forward_extended(alpha, eta, kappa, lam, mu, beta):
    omega = math.acos(np.clip(np.dot(v0, v1), -1.0, 1.0))
    sin_omega = math.sin(omega)
    w0 = omega - alpha
    n = np.cross(v1 - v0, v2 - v0)
    n = n / np.linalg.norm(n)
    Vmat = np.stack([v0, v1, v2], axis=0)

    b = np.asarray(beta, dtype=float)
    P = float(b[0] * b[1] * b[2])
    S = eta + kappa * (b - 1.0 / 3.0)
    if lam != 0.0 or mu != 0.0:
        cyc = np.array([b[1] * b[2], b[2] * b[0], b[0] * b[1]])
        S = S + lam * (cyc - 1.0 / 9.0) + mu * (b - 1.0 / 3.0) ** 2
    f = alpha * b + 3.0 * w0 * b * b - 2.0 * w0 * b * b * b
    weights = (1.0 + P * S) * np.sin(f) / sin_omega
    v_star = Vmat.T @ weights
    n_dot = float(np.dot(v_star, n))
    disc = 1.0 + n_dot * n_dot - float(np.dot(v_star, v_star))
    p = -n_dot + math.sqrt(max(disc, 0.0))
    out = v_star + p * n
    return out / np.linalg.norm(out)


def jac_dets(alpha, eta, kappa, lam=0.0, mu=0.0, n_grid=40, h=1e-4):
    """|∂v/∂β₁ × ∂v/∂β₂| at interior barycentric grid points."""
    dets = []
    for i in range(2, n_grid - 1):
        for j in range(2, n_grid - i - 1):
            k = n_grid - i - j
            if k < 2:
                continue
            b1, b2 = i / n_grid, j / n_grid
            # Central FD
            def F(x, y):
                return forward_extended(alpha, eta, kappa, lam, mu,
                                        np.array([1.0 - x - y, x, y]))
            v_p1 = F(b1 + h, b2)
            v_m1 = F(b1 - h, b2)
            v_p2 = F(b1, b2 + h)
            v_m2 = F(b1, b2 - h)
            d1 = (v_p1 - v_m1) / (2 * h)
            d2 = (v_p2 - v_m2) / (2 * h)
            dets.append(float(np.linalg.norm(np.cross(d1, d2))))
    return np.array(dets)


def area_ratio(params, ansatz="3p"):
    if ansatz == "1p":
        a, e, k_, l_, m_ = params[0], 0.0, 0.0, 0.0, 0.0
    elif ansatz == "3p":
        a, e, k_ = params; l_, m_ = 0.0, 0.0
    elif ansatz == "4p":
        a, e, k_, l_ = params; m_ = 0.0
    elif ansatz == "5p":
        a, e, k_, l_, m_ = params
    dets = jac_dets(a, e, k_, l_, m_, n_grid=30)
    return float(dets.max() / dets.min())


def main():
    omega = math.acos(np.clip(np.dot(v0, v1), -1.0, 1.0))
    print(f"omega = {omega:.10f}")
    print()

    # Sanity at recommended.
    r = area_ratio([1.149, 0.121, 0.170], "3p")
    print(f"At (1.149, 0.121, 0.170): area_ratio = {r:.6f}  (target ~1.002)")
    if r > 1.05:
        print("  -- sanity FAILS, Jacobian computation is broken")
        return
    print()

    # 1-param.
    print("Optimizing 1-param (alpha)...")
    r1 = minimize(area_ratio, [1.142], args=("1p",), method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 500})
    print(f"  alpha={r1.x[0]:.10f}  area_ratio={r1.fun:.10f}")

    # 3-param.
    print("Optimizing 3-param...")
    r3 = minimize(area_ratio, [1.149, 0.121, 0.170], args=("3p",),
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 3000})
    print(f"  alpha={r3.x[0]:.10f} eta={r3.x[1]:.10f} kappa={r3.x[2]:.10f}")
    print(f"  area_ratio={r3.fun:.10f}")

    # 4-param.
    print("Optimizing 4-param (+lambda)...")
    r4 = minimize(area_ratio, [*r3.x, 0.0], args=("4p",),
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 5000})
    print(f"  alpha={r4.x[0]:.8f} eta={r4.x[1]:.8f} kappa={r4.x[2]:.8f} lam={r4.x[3]:.8f}")
    print(f"  area_ratio={r4.fun:.10f}")

    # 5-param.
    print("Optimizing 5-param (+mu)...")
    r5 = minimize(area_ratio, [*r4.x, 0.0], args=("5p",),
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000})
    print(f"  alpha={r5.x[0]:.8f} eta={r5.x[1]:.8f} kappa={r5.x[2]:.8f}")
    print(f"  lam={r5.x[3]:.8f} mu={r5.x[4]:.8f}")
    print(f"  area_ratio={r5.fun:.10f}")
    print()

    print("Floor scaling (log10(area_ratio - 1)):")
    for label, fun in [("1p", r1.fun), ("3p", r3.fun), ("4p", r4.fun), ("5p", r5.fun)]:
        excess = fun - 1.0
        print(f"  {label}: excess={excess:.3e}  log10={math.log10(max(excess, 1e-15)):+.2f}")


if __name__ == "__main__":
    main()
