"""Face <-> sphere projections.

A :class:`Projection` maps between barycentric coordinates on a single
spherical triangle ``(V0, V1, V2)`` and points on the unit sphere. The
triangle is bound at construction; the interface is barycentric in
both directions so alternative maps — α-slerp, gnomonic, Snyder ISEA,
D₃-corrected, equal-area tweaks — can be dropped in without touching
the indexing layer.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Projection(Protocol):
    """Swappable spherical-triangle / barycentric projection.

    Implementations take three unit 3-vectors ``(V0, V1, V2)`` (CCW
    around the triangle) and expose ``to_sphere`` / ``to_bary``. Both
    directions consume and emit plain ``numpy`` arrays of shape ``(3,)``
    — barycentric ``β = (β0, β1, β2)`` summing to 1, or unit 3-vectors.
    """

    def __init__(self, v0, v1, v2) -> None: ...

    def to_sphere(self, beta: np.ndarray) -> np.ndarray:
        """Barycentric β -> unit-sphere point."""
        ...

    def to_bary(self, p: np.ndarray) -> np.ndarray:
        """Unit-sphere point -> barycentric β on (V0, V1, V2)."""
        ...


@dataclass(frozen=True, eq=False)
class _TriGeom:
    """Cached per-triangle geometry shared across projections."""
    V: np.ndarray   # (3, 3) stacked vertices
    e1: np.ndarray  # V1 − V0
    e2: np.ndarray  # V2 − V0
    n: np.ndarray   # outward unit normal
    g11: float      # gram matrix entries: e_i · e_j
    g12: float
    g22: float
    det_g: float


def _triangle_geom(v0, v1, v2) -> _TriGeom:
    V = np.stack([np.asarray(v, dtype=float) for v in (v0, v1, v2)], axis=0)
    e1 = V[1] - V[0]
    e2 = V[2] - V[0]
    n_raw = np.cross(e1, e2)
    n = n_raw / np.linalg.norm(n_raw)
    g11 = float(e1 @ e1)
    g12 = float(e1 @ e2)
    g22 = float(e2 @ e2)
    return _TriGeom(V, e1, e2, n, g11, g12, g22, g11 * g22 - g12 * g12)


def _bary_from_offset(g: _TriGeom, d: np.ndarray) -> np.ndarray:
    """In-plane offset ``d = q − V0`` -> barycentric ``(b0, b1, b2)``."""
    d1 = float(d @ g.e1)
    d2 = float(d @ g.e2)
    b1 = (g.g22 * d1 - g.g12 * d2) / g.det_g
    b2 = (g.g11 * d2 - g.g12 * d1) / g.det_g
    return np.array([1.0 - b1 - b2, b1, b2])


class Gnomonic:
    """Triangle-bound gnomonic projection.

    ``to_sphere(β)`` builds the planar point ``β0·V0 + β1·V1 + β2·V2``
    and renormalizes to the unit sphere. ``to_bary(p)`` scales ``p``
    along its ray to land on the triangle's plane, then reads off
    barycentric coordinates. Edges map to great-circle arcs; area is
    badly distorted toward face corners.
    """

    def __init__(self, v0, v1, v2) -> None:
        self._g = _triangle_geom(v0, v1, v2)
        self.V = self._g.V
        self.n = self._g.n
        self._d_plane = float(self.V[0] @ self.n)

    def to_sphere(self, beta: np.ndarray) -> np.ndarray:
        b = np.asarray(beta, dtype=float)
        v = b[0] * self.V[0] + b[1] * self.V[1] + b[2] * self.V[2]
        return v / np.linalg.norm(v)

    def to_bary(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        q = p * (self._d_plane / float(p @ self.n))
        return _bary_from_offset(self._g, q - self.V[0])


# Recommended parameters from the constrained-DGGS distortion study
# (Strategy 1b, area-optimised 3-parameter form):
# see /Users/aj/work/2026-04-18_distort/constrained_plan.md, line 633.
ALPHA_SLERP_DEFAULTS = (1.149, 0.121, 0.170)  # (alpha, eta, kappa)


class AlphaSlerp:
    """3-parameter α-slerp projection for a single icosahedron face.

    Takes barycentric coordinates ``β = (β0, β1, β2)`` on the spherical
    triangle ``(V0, V1, V2)`` and returns a unit 3-vector. Edges map
    onto great-circle arcs exactly, and with the recommended
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
        self._g = _triangle_geom(v0, v1, v2)
        self.V = self._g.V
        self.n = self._g.n
        self.omega = float(np.arccos(np.clip(np.dot(self.V[0], self.V[1]), -1.0, 1.0)))
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.kappa = float(kappa)
        self._w0 = self.omega - self.alpha
        self._sin_omega = math.sin(self.omega)

        # Tangent-plane basis at V0, used by the inverse solver's residual.
        e1 = self._g.e1
        u_raw = e1 - np.dot(e1, self.n) * self.n
        self._u = u_raw / np.linalg.norm(u_raw)
        self._v = np.cross(self.n, self._u)

    def to_sphere(self, beta) -> np.ndarray:
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
        offset = q - self.V[0]
        d = offset - np.dot(offset, self.n) * self.n
        return _bary_from_offset(self._g, d)

    def _residual_tangent(self, beta: np.ndarray, q: np.ndarray) -> np.ndarray:
        diff = self.to_sphere(beta) - q
        return np.array([diff @ self._u, diff @ self._v])

    def to_bary(self, p: np.ndarray,
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
            f"AlphaSlerp.to_bary did not converge in {max_iters} iters "
            f"(residual² = {float(r @ r):.3e})"
        )


def _cubic_inverse_unit(alpha: float, w0: float, target: float) -> float:
    """Solve ``f(β) = α·β + 3·w0·β² − 2·w0·β³ = target`` for ``β ∈ [0, 1]``.

    f maps [0, 1] monotonically onto [0, ω = α + w0]. 1D Newton with
    a uniform-rate warm start (β = target/ω) is enough — typically 3-4
    iters to machine precision.
    """
    omega = alpha + w0
    if omega <= 0.0:
        return 0.0
    if abs(w0) < 1e-15:
        return max(0.0, min(1.0, target / alpha))

    beta = max(0.0, min(1.0, target / omega))
    for _ in range(20):
        f = alpha * beta + 3.0 * w0 * beta * beta - 2.0 * w0 * beta ** 3
        df = alpha + 6.0 * w0 * beta * (1.0 - beta)
        if abs(df) < 1e-15:
            break
        step = (f - target) / df
        beta_new = max(0.0, min(1.0, beta - step))
        if abs(beta_new - beta) < 1e-14:
            return beta_new
        beta = beta_new
    return beta


class AlphaOnlySlerp(AlphaSlerp):
    """1-parameter α-only slerp: the (α, 0, 0) sub-family of α-slerp.

    Cubic edge reparameterization preserves edges-on-arcs; with η=κ=0
    the interior ``(1 + P · S)`` correction vanishes, so weights
    ``w_i = sin(f(β_i))/sin(ω)`` decouple per coordinate. Forward is
    identical to ``AlphaSlerp.to_sphere`` at η=κ=0.

    At α ≈ 1.149 the hex-cell area ratio at k=32 is 1.023 — about a
    third of α-slerp rich's area-uniformity win, with a cheaper
    *and* more robust inverse: a 1D Newton on the slerp-weight sum
    ``W`` followed by a per-coord cubic-inverse, instead of 2D FD-Newton.

    The 1D path also avoids the singular-Jacobian failure mode that
    afflicts the inherited 2D FD-Newton at triangle corners (where
    the absent ``(1+P·S)`` correction makes the 2D forward nearly
    rank-deficient). See ``alpha_slerp_followups.md`` in the sibling
    distortion repo at ``/Users/aj/work/2026-04-18_distort/`` for the
    full cost/quality analysis.
    """

    def __init__(self, v0, v1, v2,
                 alpha: float = ALPHA_SLERP_DEFAULTS[0]) -> None:
        super().__init__(v0, v1, v2, alpha=alpha, eta=0.0, kappa=0.0)
        # Fast-path inverse uses the identity d_plane·n = (V0+V1+V2)/3,
        # which holds only on equilateral spherical triangles. mu3's icosa
        # faces always satisfy this; guard against accidental misuse.
        V = self._g.V
        c01 = float(V[0] @ V[1])
        c12 = float(V[1] @ V[2])
        c20 = float(V[2] @ V[0])
        if not (abs(c01 - c12) < 1e-12 and abs(c12 - c20) < 1e-12):
            raise ValueError(
                "AlphaOnlySlerp requires an equilateral spherical triangle "
                f"(pairwise V·V = {c01:.6g}, {c12:.6g}, {c20:.6g}). "
                "The fast-path to_bary identity b_i = w_i + (1−W)/3 fails on "
                "non-equilateral triangles; use AlphaSlerp for those."
            )

    def to_bary(self, p: np.ndarray,
                tol: float = 1e-13,
                max_iters: int = 20) -> np.ndarray:
        """Sphere → barycentric via face-plane projection + 1D Newton on W.

        For an equilateral spherical triangle on the unit sphere,
        ``d_plane·n = (V0+V1+V2)/3``, so the Euclidean barycentric ``b``
        of ``p`` orthogonally projected onto the face plane is related
        to the slerp weights ``w`` by ``b_i = w_i + (1−W)/3`` where
        ``W = Σ w_i``. Knowing ``b``, we solve for the scalar ``W``
        such that ``Σ β_i(W) = 1``, then read off each ``β_i`` from a
        per-coord cubic inverse.
        """
        p = np.asarray(p, dtype=float)
        b = self._warm_start_barycentric(p)
        sin_omega = self._sin_omega
        alpha = self.alpha
        w0 = self._w0

        def betas(W: float) -> np.ndarray:
            w = b - (1.0 - W) / 3.0
            target = np.clip(w * sin_omega, -1.0 + 1e-15, 1.0 - 1e-15)
            f_beta = np.arcsin(target)
            return np.array([_cubic_inverse_unit(alpha, w0, float(f)) for f in f_beta])

        # Σ β_i(W) is monotonically increasing; warm-start at W=1
        # (corresponds to v on the face plane, the gnomonic case).
        W = 1.0
        h = 1e-6
        for _ in range(max_iters):
            beta = betas(W)
            residual = float(beta.sum()) - 1.0
            if abs(residual) < tol:
                return beta
            beta_h = betas(W + h)
            slope = (float(beta_h.sum()) - float(beta.sum())) / h
            if abs(slope) < 1e-15:
                break
            W -= residual / slope
        return betas(W)
