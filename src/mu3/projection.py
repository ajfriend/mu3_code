"""Face <-> sphere projections.

A :class:`Projection` maps between points on a single icosahedron face
(planar, centered at the origin) and points on the unit sphere. The
interface is kept minimal so alternative maps — α-slerp, D_3-corrected,
equal-area tweaks — can be dropped in without touching the indexing layer.
"""

import math
from typing import Protocol

import numpy as np


class Projection(Protocol):
    """Swappable face/sphere projection.

    Implementations take the face-center unit vector ``center`` (and, if
    needed, an orientation — see :class:`Gnomonic`) and expose ``forward``
    and ``inverse`` methods. Inputs and outputs are plain ``numpy`` arrays
    with a trailing axis of length 2 (planar) or 3 (sphere).
    """

    def forward(self, xy: np.ndarray) -> np.ndarray:
        """Planar face coordinates -> unit-sphere points (..., 3)."""
        ...

    def inverse(self, p: np.ndarray) -> np.ndarray:
        """Unit-sphere points -> planar face coordinates (..., 2)."""
        ...


class Gnomonic:
    """Gnomonic projection tangent to the sphere at ``center``.

    Straight lines on the plane map to great-circle arcs on the sphere.
    Area is badly distorted toward face corners — this is a starting
    point, not the final map.
    """

    def __init__(self, center: np.ndarray, up: np.ndarray | None = None) -> None:
        c = np.asarray(center, dtype=float)
        c = c / np.linalg.norm(c)
        if up is None:
            # arbitrary axis not parallel to c
            ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = ref - np.dot(ref, c) * c
        u = np.asarray(up, dtype=float)
        u = u - np.dot(u, c) * c
        u = u / np.linalg.norm(u)
        v = np.cross(c, u)
        self.center = c
        self.u = u  # planar x-axis, tangent to sphere at center
        self.v = v  # planar y-axis

    def forward(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x = xy[..., 0:1]
        y = xy[..., 1:2]
        p = self.center + x * self.u + y * self.v
        return p / np.linalg.norm(p, axis=-1, keepdims=True)

    def inverse(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        # scale each ray so it lies on the tangent plane at center
        denom = p @ self.center
        q = p / denom[..., None]
        x = q @ self.u
        y = q @ self.v
        return np.stack([x, y], axis=-1)


# Recommended parameters from the constrained-DGGS distortion study
# (Strategy 1b, area-optimised 3-parameter form):
# see /Users/aj/work/2026-04-18_distort/constrained_plan.md, line 633.
ALPHA_SLERP_DEFAULTS = (1.149, 0.121, 0.170)  # (alpha, eta, kappa)


class AlphaSlerp:
    """3-parameter α-slerp projection for a single icosahedron face.

    Forward only (for now). Takes barycentric coordinates ``β = (β0, β1, β2)``
    on the spherical triangle ``(V0, V1, V2)`` and returns a unit 3-vector.
    Edges map onto great-circle arcs exactly, and with the recommended
    ``(α, η, κ) = (1.149, 0.121, 0.170)`` the Jacobian area ratio is
    1.0014 at subdivision level k=32 — effectively equal-area.

    Mathematical form from the cubic arc-length family:

        f(β)     = α·β + 3·(ω−α)·β² − 2·(ω−α)·β³          # cubic edge param
        S_i      = η + κ·(β_i − 1/3)                        # interior scale
        weights  = (1 + β0·β1·β2 · S) · sin(f) / sin(ω)
        v_star   = Σ weights_i · V_i
        p        = −v_star·n + √(1 + (v_star·n)² − |v_star|²)
        v        = v_star + p · n
        return   v / ‖v‖

    where ``n`` is the face's outward normal and ``ω`` is the edge angular
    length ``arccos(V0·V1)``.
    """

    def __init__(self, v0, v1, v2,
                 alpha: float = ALPHA_SLERP_DEFAULTS[0],
                 eta:   float = ALPHA_SLERP_DEFAULTS[1],
                 kappa: float = ALPHA_SLERP_DEFAULTS[2]) -> None:
        self.V = np.stack([np.asarray(v, dtype=float) for v in (v0, v1, v2)], axis=0)
        self.omega = float(np.arccos(np.clip(np.dot(self.V[0], self.V[1]), -1.0, 1.0)))
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.kappa = float(kappa)
        self._w0 = self.omega - self.alpha
        self._sin_omega = math.sin(self.omega)
        e1 = self.V[1] - self.V[0]
        e2 = self.V[2] - self.V[0]
        n = np.cross(e1, e2)
        self.n = n / np.linalg.norm(n)

        # Face-plane geometry, cached for the inverse solver.
        u_raw = e1 - np.dot(e1, self.n) * self.n
        self._u = u_raw / np.linalg.norm(u_raw)
        self._v = np.cross(self.n, self._u)
        self._e1 = e1
        self._e2 = e2
        self._g11 = float(e1 @ e1)
        self._g12 = float(e1 @ e2)
        self._g22 = float(e2 @ e2)
        self._det_g = self._g11 * self._g22 - self._g12 * self._g12

    def forward_barycentric(self, beta) -> np.ndarray:
        b = np.asarray(beta, dtype=float)
        P = float(b[0] * b[1] * b[2])
        S = self.eta + self.kappa * (b - 1.0 / 3.0)
        f = self.alpha * b + 3.0 * self._w0 * b ** 2 - 2.0 * self._w0 * b ** 3
        weights = (1.0 + P * S) * np.sin(f) / self._sin_omega
        v_star = self.V.T @ weights
        n_dot = float(np.dot(v_star, self.n))
        disc = 1.0 + n_dot ** 2 - float(np.dot(v_star, v_star))
        p = -n_dot + math.sqrt(max(disc, 0.0))
        v = v_star + p * self.n
        return v / np.linalg.norm(v)

    def _warm_start_barycentric(self, q: np.ndarray) -> np.ndarray:
        """Euclidean barycentric of q orthogonally projected onto the face plane."""
        d = q - np.dot(q - self.V[0], self.n) * self.n - self.V[0]
        d1 = float(d @ self._e1)
        d2 = float(d @ self._e2)
        b1 = (self._g22 * d1 - self._g12 * d2) / self._det_g
        b2 = (self._g11 * d2 - self._g12 * d1) / self._det_g
        return np.array([1.0 - b1 - b2, b1, b2])

    def _residual_tangent(self, beta: np.ndarray, q: np.ndarray) -> np.ndarray:
        diff = self.forward_barycentric(beta) - q
        return np.array([diff @ self._u, diff @ self._v])

    def inverse_barycentric(self, p: np.ndarray,
                            tol_residual: float = 1e-14,
                            tol_step: float = 1e-13,
                            max_iters: int = 20,
                            fd_step: float = 1e-7) -> np.ndarray:
        """Unit sphere point ``p`` (on the face) -> barycentric ``β`` on (V0, V1, V2).

        Finite-difference Newton in two unknowns ``(β0, β1)`` with
        ``β2 = 1 − β0 − β1``. Warm-started from a Euclidean barycentric
        read-off after orthogonally projecting ``p`` onto the face plane.

        Raises :class:`RuntimeError` if the solver does not converge within
        ``max_iters`` — which empirically never happens inside the face
        (benchmark converges in 3 iters across interior/edge/corner strata).
        """
        q = np.asarray(p, dtype=float)
        beta = self._warm_start_barycentric(q)

        for _ in range(max_iters):
            r = self._residual_tangent(beta, q)
            if float(r @ r) < tol_residual ** 2:
                return beta

            # Forward-difference 2x2 Jacobian (two extra forward evals per iter).
            beta_h0 = np.array([beta[0] + fd_step, beta[1], beta[2] - fd_step])
            beta_h1 = np.array([beta[0], beta[1] + fd_step, beta[2] - fd_step])
            r0 = self._residual_tangent(beta_h0, q)
            r1 = self._residual_tangent(beta_h1, q)
            J = np.column_stack([(r0 - r) / fd_step, (r1 - r) / fd_step])

            step = np.linalg.solve(J, -r)
            beta = np.array([beta[0] + step[0], beta[1] + step[1],
                             beta[2] - step[0] - step[1]])
            if float(step @ step) < tol_step ** 2:
                return beta

        raise RuntimeError(
            f"AlphaSlerp.inverse_barycentric did not converge in {max_iters} iters "
            f"(residual² = {float(r @ r):.3e})"
        )
