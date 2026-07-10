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

# The package-wide currency for 3D positions: a point on the unit
# sphere as a plain numpy array — shape (3,), float64, unit norm.
# An annotation, not a wrapper: values stay ordinary ndarrays. It
# annotates the point-passing surface (single points only — batched
# (M, 3) returns like cell_boundary stay np.ndarray); it lives here
# because projection is where 3D enters the system.
Vec3 = np.ndarray


class Projection(Protocol):
    """Swappable spherical-triangle / barycentric projection.

    Implementations take three unit 3-vectors ``(V0, V1, V2)`` (CCW
    around the triangle) and expose ``to_sphere`` / ``to_bary``. Both
    directions consume and emit plain ``numpy`` arrays of shape ``(3,)``
    — barycentric ``β = (β0, β1, β2)`` summing to 1, or unit
    :data:`Vec3` sphere points.
    """

    def __init__(self, v0, v1, v2) -> None: ...

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        """Barycentric β -> unit-sphere point."""
        ...

    def to_bary(self, p: Vec3) -> np.ndarray:
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

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        b = np.asarray(beta, dtype=float)
        v = b[0] * self.V[0] + b[1] * self.V[1] + b[2] * self.V[2]
        return v / np.linalg.norm(v)

    def to_bary(self, p: Vec3) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        q = p * (self._d_plane / float(p @ self.n))
        return _bary_from_offset(self._g, q - self.V[0])


_GNOMONIC_PSI_DEFAULTS = (-0.34499, 0.17899, -0.05624)  # degree-4 ψ (c₂,c₃,c₄)


class GnomonicPsiWarp(Gnomonic):
    """Gnomonic projection pre-warped by a per-coordinate polynomial ψ.

    ``to_sphere(β) = normalize(Σ ψ(βᵢ) Vᵢ)`` with
    ``ψ(x) = x + c₂x² + c₃x³ + c₄x⁴`` (monotone on [0,1], ψ(0)=0), applied
    per coordinate before gnomonic normalization. The warp cancels gnomonic's
    area distortion: at the fitted coefficients the map is equal-area to the
    icosahedral floor (area_r ≈ 1.002, aspect median ≈ 1.158). Edges stay on
    great-circle arcs (ψ fixes the simplex boundary), and the whole map is
    **transcendental-free** in both directions.

    ``to_bary`` inverts in closed-form + a 1-D solve: gnomonic⁻¹ (the parent)
    recovers ``gᵢ = ψ(βᵢ)/W`` (normalized); a 1-D monotone solve recovers the
    scale ``W`` from ``Σ βᵢ = 1``; then ``βᵢ = ψ⁻¹(gᵢW)`` per coordinate. No
    transcendentals, no 2-D Newton.

    The ψ coefficients are fit for the equilateral icosahedron face; a guard
    rejects other triangles.
    """

    def __init__(self, v0, v1, v2, coeffs=_GNOMONIC_PSI_DEFAULTS) -> None:
        super().__init__(v0, v1, v2)
        self.coeffs = tuple(float(c) for c in coeffs)
        c01 = float(self.V[0] @ self.V[1])
        c12 = float(self.V[1] @ self.V[2])
        c20 = float(self.V[2] @ self.V[0])
        if not (abs(c01 - c12) < 1e-12 and abs(c12 - c20) < 1e-12):
            raise ValueError(
                'GnomonicPsiWarp is fit for equilateral (icosa) faces; '
                f'pairwise V·V = {c01:.6g}, {c12:.6g}, {c20:.6g}.')
        self._psi1 = self._psi(1.0)

    def _psi(self, x):
        c = self.coeffs
        return x + c[0] * x ** 2 + c[1] * x ** 3 + c[2] * x ** 4

    def _dpsi(self, x):
        c = self.coeffs
        return 1.0 + 2 * c[0] * x + 3 * c[1] * x ** 2 + 4 * c[2] * x ** 3

    def _psi_inv(self, y: float) -> float:
        if y <= 0.0:
            return 0.0
        if y >= self._psi1:
            return 1.0
        x = y                                    # ψ ≈ x near 0 → good start
        for _ in range(40):
            step = (self._psi(x) - y) / self._dpsi(x)
            x -= step
            if abs(step) < 1e-15:
                break
        return x

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        w = self._psi(np.asarray(beta, dtype=float))
        v = w @ self.V
        return v / np.linalg.norm(v)

    def to_bary(self, p: Vec3) -> np.ndarray:
        g = Gnomonic.to_bary(self, p)            # gᵢ = ψ(βᵢ)/W  (closed form)
        # 1-D monotone solve for W: Σ ψ⁻¹(gᵢW) − 1 = 0 (bisection-safeguarded).
        lo, hi = 1e-9, self._psi1 / max(float(g.max()), 1e-12)
        W = 0.5 * (lo + hi)
        for _ in range(80):
            b = [self._psi_inv(gi * W) for gi in g]
            f = b[0] + b[1] + b[2] - 1.0
            if abs(f) < 1e-14:
                break
            if f > 0.0:
                hi = W
            else:
                lo = W
            fp = g[0] / self._dpsi(b[0]) + g[1] / self._dpsi(b[1]) \
                + g[2] / self._dpsi(b[2])
            W = W - f / fp if fp > 0.0 else W
            if not (lo < W < hi):
                W = 0.5 * (lo + hi)
        return np.array([self._psi_inv(gi * W) for gi in g])


# Recommended parameters from the constrained-DGGS distortion study
# (Strategy 1b, area-optimised 3-parameter form):
# see 2026-04-18_distort/constrained_plan.md, line 633.
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

    def to_sphere(self, beta) -> Vec3:
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

    def to_bary(self, p: Vec3,
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


_WARPED_KARCHER_DEFAULTS = (0.0344, 0.0107, 0.0087)


class WarpedKarcher:
    """Reweighted Karcher-mean face map with a closed-form inverse.

    The point ``p`` for barycentric ``β`` is defined implicitly by the
    reweighted Riemannian-centroid condition::

        Σ βᵢ · cᵢ(p) · log_p(Vᵢ) = 0,     Σ βᵢ = 1

    with a symmetric per-vertex reweight
    ``cᵢ(p) = exp(Σₖ λₖ (dᵢᵏ − mean dᵏ))``, ``dᵢ = θᵢ − mean θ``,
    ``θᵢ = arccos(p·Vᵢ)``. Because ``cᵢ`` depends only on ``p``:

    - ``to_bary`` (sphere → β, the point-location direction) is a single
      3×3 linear solve — no Newton, no iteration;
    - edges map to great-circle arcs exactly for any positive ``cᵢ``;
    - the reweight ``λ`` tunes cell area toward equal-area.

    ``to_sphere`` (β → sphere) is the implicit direction and uses Picard
    iteration — the dual of :class:`AlphaSlerp` (closed-form forward,
    iterative inverse). The reweight is fit for the equilateral icosahedron
    face; a guard rejects non-equilateral triangles.
    """

    def __init__(self, v0, v1, v2, lam=_WARPED_KARCHER_DEFAULTS) -> None:
        self.V = np.stack([np.asarray(v, dtype=float) for v in (v0, v1, v2)])
        self.lam = tuple(float(x) for x in lam)
        c01 = float(self.V[0] @ self.V[1])
        c12 = float(self.V[1] @ self.V[2])
        c20 = float(self.V[2] @ self.V[0])
        if not (abs(c01 - c12) < 1e-12 and abs(c12 - c20) < 1e-12):
            raise ValueError(
                'WarpedKarcher reweight is fit for equilateral (icosa) faces; '
                f'pairwise V·V = {c01:.6g}, {c12:.6g}, {c20:.6g}.')

    def _logs(self, p):
        ts, ths = [], []
        for Vi in self.V:
            c = float(np.clip(p @ Vi, -1.0, 1.0))
            u = Vi - c * p
            nu = np.linalg.norm(u)
            if nu < 1e-14:
                ts.append(np.zeros(3)); ths.append(0.0)
            else:
                # Compiled ports: prefer Kahan's 2·atan2(‖p−Vi‖, ‖p+Vi‖),
                # accurate as θ → 0 where acos(dot) loses digits.
                th = math.acos(c)
                ts.append(th * u / nu); ths.append(th)
        return ts, ths

    def _reweight(self, ths):
        d = np.asarray(ths) - float(np.mean(ths))
        poly = np.zeros(3)
        for k, lk in enumerate(self.lam, start=1):
            poly = poly + lk * (d ** k - float(np.mean(d ** k)))
        return np.exp(poly)

    def to_sphere(self, beta, iters: int = 24) -> Vec3:
        b = np.asarray(beta, dtype=float)
        p = self.V.T @ b
        p = p / np.linalg.norm(p)
        for _ in range(iters):
            ts, ths = self._logs(p)
            c = self._reweight(ths)
            w = b[0] * c[0] * ts[0] + b[1] * c[1] * ts[1] + b[2] * c[2] * ts[2]
            nw = np.linalg.norm(w)
            if nw < 1e-15:
                break
            p = math.cos(nw) * p + math.sin(nw) * (w / nw)
        return p / np.linalg.norm(p)

    def to_bary(self, p: Vec3) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        ts, ths = self._logs(p)
        c = self._reweight(ths)
        a = [c[i] * ts[i] for i in range(3)]
        ref = (np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        e1 = ref - (ref @ p) * p
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(p, e1)
        A = np.array([[a[i] @ e1 for i in range(3)],
                      [a[i] @ e2 for i in range(3)],
                      [1.0, 1.0, 1.0]])
        return np.linalg.solve(A, np.array([0.0, 0.0, 1.0]))


class AlphaSlerpExtended(AlphaSlerp):
    """AlphaSlerp + higher-order interior polynomial correction.

    Extends ``Sᵢ`` from linear to cubic in ``δᵢ = βᵢ − 1/3``::

        Sᵢ      = η + κ·δᵢ + λ·δᵢ² + μ·δᵢ³
        weights = (1 + P · Sᵢ) · sin(f(βᵢ)) / sin(ω),  P = β₀β₁β₂

    Edge-on-arc preserved automatically (P = 0 on edges, so the entire
    Sᵢ correction vanishes regardless of polynomial order). At
    ``(λ, μ) = (0, 0)`` reduces to plain AlphaSlerp.

    Used to test whether AlphaSlerp's 3-parameter optimum is at a true
    plateau under our discrete-cell metric. The sister-repo's
    continuous-Jacobian experiments show diminishing returns past 3
    params; this class lets us check the same against discrete cells.
    """

    def __init__(self, v0, v1, v2,
                 alpha: float = ALPHA_SLERP_DEFAULTS[0],
                 eta:   float = ALPHA_SLERP_DEFAULTS[1],
                 kappa: float = ALPHA_SLERP_DEFAULTS[2],
                 lambd: float = 0.0,
                 mu:    float = 0.0) -> None:
        super().__init__(v0, v1, v2, alpha=alpha, eta=eta, kappa=kappa)
        self.lambd = float(lambd)
        self.mu = float(mu)

    def to_sphere(self, beta) -> Vec3:
        b = np.asarray(beta, dtype=float)
        P = float(b[0] * b[1] * b[2])
        delta = b - 1.0 / 3.0
        S = (self.eta
             + self.kappa * delta
             + self.lambd * delta ** 2
             + self.mu    * delta ** 3)
        f = (self.alpha * b
             + 3.0 * self._w0 * b ** 2
             - 2.0 * self._w0 * b ** 3)
        weights = (1.0 + P * S) * np.sin(f) / self._sin_omega
        v_star = self.V.T @ weights
        n_dot = float(np.dot(v_star, self.n))
        disc = 1.0 + n_dot ** 2 - float(np.dot(v_star, v_star))
        p = -n_dot + math.sqrt(max(disc, 0.0))
        v = v_star + p * self.n
        return v / np.linalg.norm(v)


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
    distortion repo at ``2026-04-18_distort/`` for the
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

    def to_bary(self, p: Vec3,
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


# ---------------------------------------------------------------------------
# IVEA — Slice & Dice icosahedral vertex-oriented equal-area projection.
# Port of dggal's IVEA (https://github.com/ecere/dggal,
# src/projections/icoVertexGreatCircle.ec). Reference: Slice & Dice (2006),
# https://doi.org/10.1559/152304006779500687.
#
# Each icosa face is divided into 6 fundamental sub-triangles by its three
# medians. Each sub-tri has vertices (face_vertex, edge_mid, face_centroid),
# all on the unit sphere. Equal-area mapping per sub-tri (Snyder 1992
# generalized to a chosen radial vertex; IVEA puts the radial at the icosa
# face vertex).
# ---------------------------------------------------------------------------

_ICOSA_COS = 1.0 / math.sqrt(5.0)
_PHI = (1.0 + math.sqrt(5.0)) / 2.0

_IVEA_AREA = math.radians(6.0)  # fundamental sub-tri area, π/30
_IVEA_PARALLELEPIPED_V = math.sqrt((5.0 - 2.0 * math.sqrt(5.0)) / 15.0)
_IVEA_D_VM = math.atan(1.0 / _PHI)                     # vertex ↔ mid arc
_IVEA_D_MC = math.acos(math.sqrt((_PHI + 1.0) / 3.0))  # mid ↔ centroid arc
_IVEA_D_VC = math.atan(2.0 / (_PHI * _PHI))            # vertex ↔ centroid arc
_IVEA_COS_VM = math.cos(_IVEA_D_VM)
_IVEA_COS_MC = math.cos(_IVEA_D_MC)
_IVEA_SIN_MC = math.sin(_IVEA_D_MC)
_IVEA_COS_VC = math.cos(_IVEA_D_VC)
_IVEA_BETA = math.radians(36.0)
_IVEA_GAMMA = math.radians(60.0)
_IVEA_ALPHA = math.radians(90.0)


def _slerp_angle(p0: np.ndarray, p1: np.ndarray,
                 distance: float, movement: float) -> np.ndarray:
    s = math.sin(distance)
    return (math.sin(distance - movement) * p0 + math.sin(movement) * p1) / s


def _spherical_tri_area(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """Signed spherical triangle area / excess (steradians on unit sphere).

    ``sin(Ω/2) = det₃(α, β, γ)`` where α, β, γ are the unit midpoints of
    the triangle's three sides. From Tuynman 2013, "Areas of spherical
    and hyperbolic triangles in terms of their midpoints"
    (arxiv:1307.2567). Numerically stable for small triangles —
    midpoint norms are ``2·cos(arc/2)`` so they're well-conditioned at
    any non-antipodal pair, unlike the van Oosterom-Strackee denominator
    ``1 + V·V + V·V + V·V`` which loses precision when vertices are
    very close.
    """
    midAB = A + B; midAB = midAB / np.linalg.norm(midAB)
    midBC = B + C; midBC = midBC / np.linalg.norm(midBC)
    midCA = C + A; midCA = midCA / np.linalg.norm(midCA)
    return 2.0 * math.asin(max(-1.0, min(1.0,
        float(midAB @ np.cross(midBC, midCA)))))


def _sqrt_one_minus_dot_over_2(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable ``sqrt((1 − a·b) / 2)`` for unit vectors a, b.

    Avoids catastrophic cancellation when a ≈ b. From Felix Palmer's
    a5geo formulation referenced in dggal.
    """
    midAB = (a + b) / 2.0
    n = np.linalg.norm(midAB)
    if n < 1e-15:
        return 1.0
    midAB = midAB / n
    c = np.cross(a, midAB)
    D = float(np.linalg.norm(c))
    if D < 1e-8:
        D = float(np.linalg.norm(a - b)) / 2.0
    return D


def _bary_in_subtri(beta: np.ndarray, Pa: np.ndarray, Pb: np.ndarray,
                    Pc: np.ndarray) -> np.ndarray:
    """face-bary β -> sub-tri bary (b0, b1, b2) where β = b0·Pa + b1·Pb + b2·Pc.

    Pa/Pb/Pc are face-barycentric of the sub-triangle's vertices (each
    summing to 1), so the system has rank 2; solve any 2 of 3 equations.
    """
    M = np.array([
        [Pb[0] - Pa[0], Pc[0] - Pa[0]],
        [Pb[1] - Pa[1], Pc[1] - Pa[1]],
    ])
    rhs = np.array([beta[0] - Pa[0], beta[1] - Pa[1]])
    sol = np.linalg.solve(M, rhs)
    return np.array([1.0 - sol[0] - sol[1], float(sol[0]), float(sol[1])])


class IVEAProjection:
    """Slice & Dice icosahedral vertex-oriented equal-area projection.

    Forward (``to_sphere``) and inverse (``to_bary``) work face-globally:
    any input/output is in face barycentric ``β = (β0, β1, β2)`` on
    ``(V0, V1, V2)``. Internally classifies into one of six fundamental
    sub-triangles and applies the per-sub-tri Slice & Dice formulas.

    Requires the input triangle to be an icosahedron face (equilateral
    spherical with edge cosine ``1/√5``); the Slice & Dice constants
    encoded here are specific to that geometry. Use ``AlphaSlerp`` /
    ``Gnomonic`` / ``AlphaOnlySlerp`` for arbitrary triangles.
    """

    def __init__(self, v0, v1, v2) -> None:
        self._g = _triangle_geom(v0, v1, v2)
        self.V = self._g.V
        self.n = self._g.n

        c01 = float(self.V[0] @ self.V[1])
        c12 = float(self.V[1] @ self.V[2])
        c20 = float(self.V[2] @ self.V[0])
        if not (abs(c01 - _ICOSA_COS) < 1e-10
                and abs(c12 - _ICOSA_COS) < 1e-10
                and abs(c20 - _ICOSA_COS) < 1e-10):
            raise ValueError(
                "IVEAProjection requires an icosahedron face triangle "
                f"(pairwise V·V = {c01:.6g}, {c12:.6g}, {c20:.6g}; "
                f"expected 1/√5 ≈ {_ICOSA_COS:.6g})."
            )

        # Edge midpoints on the sphere; mids[i] is opposite V[i].
        V = self.V
        mids = np.stack([V[1] + V[2], V[0] + V[2], V[0] + V[1]])
        for k in range(3):
            mids[k] = mids[k] / np.linalg.norm(mids[k])
        self.mids = mids

        # Spherical centroid of the face.
        c = V[0] + V[1] + V[2]
        self.centroid = c / np.linalg.norm(c)

        # Pre-build face-barycentric of (mid, centroid) for the conversion
        # between face-bary and sub-tri-bary.
        self._mid_bary = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        self._centroid_bary = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # Per-sub-triangle parallelepiped sign — depends on the actual
        # icosa-vertex orientation, which can flip handedness between mu3
        # and dggal's reference frame. Compute once.
        self._sub_signs = np.empty(6)
        for subTri in range(6):
            A, B, C, *_ = self._subtri_struct(subTri)
            det = float(A @ np.cross(B, C))
            self._sub_signs[subTri] = math.copysign(1.0, det)

    @staticmethod
    def _classify_subtri(b: np.ndarray) -> int:
        """face-bary -> sub-triangle index 0..5 (dggal convention)."""
        b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
        if b0 <= b1 and b0 <= b2:
            return 0 if b1 < b2 else 1
        if b1 <= b0 and b1 <= b2:
            return 2 if b0 < b2 else 3
        return 4 if b0 < b1 else 5

    def _subtri_struct(self, subTri: int):
        """Resolve sub-tri ``subTri`` to (A, B, C, Pa, Pb, Pc, bIsA).

        A is the radial face vertex; (B, C) are (mid, centroid) when
        bIsA=True and (centroid, mid) when False.
        """
        tri3rd = subTri >> 1
        if subTri == 0 or subTri == 2: fv = 2
        elif subTri == 1 or subTri == 4: fv = 1
        else: fv = 0  # 3 or 5
        bIsA = subTri in (0, 3, 4)

        A_sph = self.V[fv]
        mid_sph = self.mids[tri3rd]
        cen_sph = self.centroid

        Pa = np.zeros(3); Pa[fv] = 1.0
        Pmid = self._mid_bary[tri3rd]
        Pcen = self._centroid_bary

        if bIsA:
            return A_sph, mid_sph, cen_sph, Pa, Pmid, Pcen, True
        return A_sph, cen_sph, mid_sph, Pa, Pcen, Pmid, False

    @staticmethod
    def _subtri_to_sphere(b: np.ndarray, A: np.ndarray, B: np.ndarray,
                          C: np.ndarray, bIsA: bool, ppV: float) -> np.ndarray:
        """Sub-tri bary -> sphere point. Mirrors dggal ``inverseVector``.

        ``ppV`` is the signed parallelepiped det ``A·(B×C)`` for this
        specific sub-tri orientation.
        """
        if b[0] > 1.0 - 1e-15: return A.copy()
        if b[1] > 1.0 - 1e-15: return B.copy()
        if b[2] > 1.0 - 1e-15: return C.copy()

        h = 1.0 - float(b[0])
        b2oh = float(b[2]) / h
        ang = b2oh * _IVEA_AREA
        halfC = math.sin(ang / 2.0)
        halfC2 = halfC * halfC
        CC = 2.0 * halfC2  # = 1 − cos(ang)
        S = 2.0 * halfC * math.sqrt(max(0.0, 1.0 - halfC2))  # = sin(ang)

        c01 = _IVEA_COS_VM if bIsA else _IVEA_COS_VC
        c12 = _IVEA_COS_MC
        c20 = _IVEA_COS_VC if bIsA else _IVEA_COS_VM
        s12 = _IVEA_SIN_MC

        f = S * ppV + CC * (c01 * c12 - c20)
        g = CC * s12 * (1.0 + c01)
        f2, g2, gf = f * f, g * g, g * f
        numerator = s12 * (f2 - g2) - 2.0 * gf * c12
        divisor = s12 * (f2 + g2)

        if abs(numerator) > 1e-9 and abs(divisor) > 1e-9:
            inv = 1.0 / divisor
            ap = max(0.0, numerator * inv)
            bp = min(1.0, 2.0 * gf * inv)
            p = ap * B + bp * C
            av = float(A @ p)
            bv = 1.0 + h * h * (av - 1.0)
            bvp = h * math.sqrt(max(0.0, (1.0 + bv) / max(1e-30, 1.0 + av)))
            avp = bv - av * bvp
            return avp * A + bvp * p

        # Fallback: 2 SLERPs through D = a point on great circle BC.
        b1pb2 = float(b[1]) + float(b[2])
        # In dggal: upOverupPvp = b[bIsA ? 1 : 2] / (b[1] + b[2])
        # → fraction of "mid-edge weight" within the BC distribution.
        if b1pb2 < 1e-11:
            up_frac = 0.0
        else:
            up_frac = (float(b[1]) if bIsA else float(b[2])) / b1pb2
        rhoPlusDelta = _IVEA_BETA + _IVEA_GAMMA - up_frac * _IVEA_AREA
        areaABD = rhoPlusDelta + _IVEA_ALPHA - math.pi

        if abs(areaABD) < 1e-11:
            D = B if bIsA else C
            BD = _IVEA_D_VM
        elif abs(areaABD - _IVEA_AREA) < 1e-13:
            D = C if bIsA else B
            BD = _IVEA_D_VC
        else:
            AD = 2.0 * math.atan2(g, f)
            D = _slerp_angle(B, C, _IVEA_D_MC, AD)
            BD = math.acos(max(-1.0, min(1.0, float(A @ D))))

        x = 2.0 * math.asin(max(-1.0, min(1.0, h * math.sin(BD / 2.0))))
        return _slerp_angle(A, D, BD, x)

    def _resolve_subtri(self, subTri: int):
        """Sub-tri vertices/face-bary, B↔C swapped if needed to align with
        dggal's positive-determinant convention (handles mu3 vs dggal
        icosa-orientation differences)."""
        A, B, C, Pa, Pb, Pc, bIsA = self._subtri_struct(subTri)
        if self._sub_signs[subTri] < 0:
            B, C = C, B
            Pb, Pc = Pc, Pb
            bIsA = not bIsA
        return A, B, C, Pa, Pb, Pc, bIsA

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        beta = np.asarray(beta, dtype=float)
        subTri = self._classify_subtri(beta)
        A, B, C, Pa, Pb, Pc, bIsA = self._resolve_subtri(subTri)
        b_sub = _bary_in_subtri(beta, Pa, Pb, Pc)
        return self._subtri_to_sphere(b_sub, A, B, C, bIsA,
                                       _IVEA_PARALLELEPIPED_V)

    @staticmethod
    def _subtri_to_bary(p: np.ndarray, A: np.ndarray, B: np.ndarray,
                        C: np.ndarray) -> np.ndarray:
        """Sphere point -> sub-tri bary. Mirrors dggal ``forwardVector``.

        Uses ``abs`` on the spherical-triangle-area term — sign depends
        on local orientation, but the fraction ``area(A,B,q) / area(A,B,C)``
        we want is unsigned (it's the swept fraction of edge BC).
        """
        c1 = np.cross(A, p)
        c2 = np.cross(B, C)
        q = np.cross(c1, c2)
        n = np.linalg.norm(q)
        if n < 1e-15:
            q = B.copy()
        else:
            q = q / n
            if float(q @ (B + C)) < 0:
                q = -q

        areaABp = abs(_spherical_tri_area(A, B, q))
        h_num = _sqrt_one_minus_dot_over_2(A, p)
        h_den = _sqrt_one_minus_dot_over_2(A, q)
        h = 0.0 if h_den < 1e-15 else h_num / h_den

        b0 = 1.0 - h
        b2 = min(h, h * areaABp / _IVEA_AREA)
        b1 = h - b2
        return np.array([b0, b1, b2])

    def to_bary(self, p: Vec3) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        # Classify by face-bary of the orthogonal projection to face plane.
        face_bary = self._warm_start_face_bary(p)
        subTri = self._classify_subtri(face_bary)
        A, B, C, Pa, Pb, Pc, bIsA = self._resolve_subtri(subTri)
        b_sub = self._subtri_to_bary(p, A, B, C)
        # Sub-tri bary -> face bary via the linear map.
        return b_sub[0] * Pa + b_sub[1] * Pb + b_sub[2] * Pc

    def _warm_start_face_bary(self, p: np.ndarray) -> np.ndarray:
        """Euclidean barycentric of p orthogonally projected onto the face."""
        offset = p - self.V[0]
        d = offset - float(offset @ self.n) * self.n
        return _bary_from_offset(self._g, d)


class AlphaIVEAProjection(IVEAProjection):
    """IVEA + cubic reparameterization on the sub-tri's two intrinsic axes.

    Each cubic ``f(t; p) = p·t + 3·(1−p)·t² − 2·(1−p)·t³`` maps [0, 1]
    monotonically onto [0, 1] for ``p ∈ (0, 3)``. We apply one to:

    - the **radial** coord ``h = 1 − b0`` (parameter ``α``) — the
      face-vertex → opposite-edge fraction, and
    - the **angular** coord ``b2oh = b2/h`` (parameter ``γ``) — the
      fraction-along-BC at distance ``h``.

    ``α = γ = 1`` recovers IVEA exactly (both cubics collapse to identity).
    Other values trade exact equal-area for tunable shape/angular behavior
    while preserving sub-tri continuity and great-circle face boundaries.
    The closed-form inverse is preserved — two extra 1D Newton steps on
    top of IVEA's per-sub-tri inverse.
    """

    def __init__(self, v0, v1, v2,
                 alpha: float = 1.0, gamma: float = 1.0) -> None:
        super().__init__(v0, v1, v2)
        for name, val in (("α", alpha), ("γ", gamma)):
            if not 0.0 < val < 3.0:
                raise ValueError(
                    f"AlphaIVEA {name} must be in (0, 3) for monotonicity; got {val}"
                )
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._w0_a = 1.0 - self.alpha
        self._w0_g = 1.0 - self.gamma

    @staticmethod
    def _cubic(t: float, p: float, w0: float) -> float:
        return p * t + 3.0 * w0 * t * t - 2.0 * w0 * t ** 3

    def _warp(self, b_sub: np.ndarray) -> np.ndarray:
        h = 1.0 - float(b_sub[0])
        if h < 1e-15:
            return b_sub
        b2oh = float(b_sub[2]) / h
        h_w = self._cubic(h, self.alpha, self._w0_a)
        b2oh_w = self._cubic(b2oh, self.gamma, self._w0_g)
        return np.array([1.0 - h_w, h_w * (1.0 - b2oh_w), h_w * b2oh_w])

    def _unwarp(self, b_sub_w: np.ndarray) -> np.ndarray:
        h_w = 1.0 - float(b_sub_w[0])
        if h_w < 1e-15:
            return b_sub_w
        b2oh_w = float(b_sub_w[2]) / h_w
        h = _cubic_inverse_unit(self.alpha, self._w0_a, h_w)
        b2oh = _cubic_inverse_unit(self.gamma, self._w0_g, b2oh_w)
        return np.array([1.0 - h, h * (1.0 - b2oh), h * b2oh])

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        beta = np.asarray(beta, dtype=float)
        subTri = self._classify_subtri(beta)
        A, B, C, Pa, Pb, Pc, bIsA = self._resolve_subtri(subTri)
        b_sub = _bary_in_subtri(beta, Pa, Pb, Pc)
        return self._subtri_to_sphere(self._warp(b_sub), A, B, C, bIsA,
                                       _IVEA_PARALLELEPIPED_V)

    def to_bary(self, p: Vec3) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        face_bary = self._warm_start_face_bary(p)
        subTri = self._classify_subtri(face_bary)
        A, B, C, Pa, Pb, Pc, bIsA = self._resolve_subtri(subTri)
        b_sub_w = self._subtri_to_bary(p, A, B, C)
        b_sub = self._unwarp(b_sub_w)
        return b_sub[0] * Pa + b_sub[1] * Pb + b_sub[2] * Pc


class LambertBaryProjection(IVEAProjection):
    """Smooth Lambert + planar-barycentric projection. **DEAD END** — kept
    as a documented null result.

    Map: face barycentric β → planar Lambert point (linear combination
    of the Lambert images of the face vertices) → sphere via inverse
    Lambert. C∞ smooth and fully closed-form, intended as a simplified
    variant of "Schwarz-Christoffel + Lambert + radial correction".

    The simplification skips the Schwarz-Christoffel step. **This breaks
    at the face boundary**: Lambert maps the spherical face edges to
    *curved* lines in the plane, but linear barycentric draws *straight*
    lines. Cells near corners are squeezed between the true (curved)
    boundary and the straight-barycentric-line, producing severely
    distorted cells. The distortion is bimodal: median cells are nearly
    perfect (shape ≈ 1.02, ang ≈ 1°) but worst-case cells are
    catastrophic (shape ≈ 3.5, area max/min ≈ 2.5, ang ≈ 32°).

    Empirical confirmation that the Schwarz-Christoffel step in the
    Option B construction is not optional. Either implement the full SC
    correction (substantial — needs complex hypergeometrics, Newton in
    complex space, boundary handling) or accept that AlphaSlerp already
    occupies the smooth + closed-form + approximate-equal-area corner.

    Class kept in the codebase so the test suite and distortion probe
    document the failure mode for future readers.
    """

    def __init__(self, v0, v1, v2) -> None:
        super().__init__(v0, v1, v2)
        c = self.centroid
        e1_raw = self.V[0] - float(self.V[0] @ c) * c
        self._lam_e1 = e1_raw / np.linalg.norm(e1_raw)
        self._lam_e2 = np.cross(c, self._lam_e1)
        self._VL = np.array([self._sphere_to_lambert(self.V[i]) for i in range(3)])
        M = np.column_stack([self._VL[1] - self._VL[0], self._VL[2] - self._VL[0]])
        self._lam_M_inv = np.linalg.inv(M)

    def _sphere_to_lambert(self, p: np.ndarray) -> np.ndarray:
        """Sphere unit vector -> 2D Lambert chord coords. r = 2·sin(angular/2)."""
        c = self.centroid
        cos_d = max(-1.0, min(1.0, float(p @ c)))
        r = math.sqrt(max(0.0, 2.0 * (1.0 - cos_d)))
        t = p - cos_d * c
        n = float(np.linalg.norm(t))
        if n < 1e-15:
            return np.array([0.0, 0.0])
        t_hat = t / n
        return np.array([float(t_hat @ self._lam_e1) * r,
                         float(t_hat @ self._lam_e2) * r])

    def _lambert_to_sphere(self, P_L: np.ndarray) -> np.ndarray:
        c = self.centroid
        x, y = float(P_L[0]), float(P_L[1])
        r = math.sqrt(x * x + y * y)
        if r < 1e-15:
            return c.copy()
        # r = 2·sin(angular/2) → cos(angular) = 1 − r²/2
        cos_d = 1.0 - 0.5 * r * r
        sin_d = math.sqrt(max(0.0, 1.0 - cos_d * cos_d))
        t_hat = (x * self._lam_e1 + y * self._lam_e2) / r
        return cos_d * c + sin_d * t_hat

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        b = np.asarray(beta, dtype=float)
        P_L = b[0] * self._VL[0] + b[1] * self._VL[1] + b[2] * self._VL[2]
        return self._lambert_to_sphere(P_L)

    def to_bary(self, p: Vec3) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        P_L = self._sphere_to_lambert(p)
        sol = self._lam_M_inv @ (P_L - self._VL[0])
        b1, b2 = float(sol[0]), float(sol[1])
        return np.array([1.0 - b1 - b2, b1, b2])


def _log_sphere(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Tangent vector at ``base`` pointing toward ``point``,
    magnitude = arc length on the unit sphere."""
    d = max(-1.0, min(1.0, float(base @ point)))
    # Compiled ports: prefer Kahan's 2·atan2(‖base−point‖, ‖base+point‖),
    # accurate as θ → 0 where acos(dot) loses digits.
    angle = math.acos(d)
    if angle < 1e-15:
        return np.zeros(3)
    direction = point - d * base
    n = float(np.linalg.norm(direction))
    if n < 1e-15:
        return np.zeros(3)
    return (angle / n) * direction


def _exp_sphere(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Geodesic from ``base`` in direction ``tangent`` (magnitude = arc length)."""
    r = float(np.linalg.norm(tangent))
    if r < 1e-15:
        return base.copy()
    return math.cos(r) * base + (math.sin(r) / r) * tangent


class KarcherProjection(IVEAProjection):
    """Smooth approximate-equal-area via the Riemannian center of mass.

    Result point ``p`` is the geodesic mean defined implicitly by
    ``Σ βᵢ · log_p(Vᵢ) = 0``. Edge-on-arc preservation is automatic
    (β with one zero coordinate puts ``p`` on the corresponding
    great-circle arc, parameterized as a slerp). Zero parameters;
    geometrically canonical.

    Forward is iterative (Picard fixed-point, ~5 iters typical).
    Inverse is **closed-form** via a 3×3 linear solve in the tangent
    plane at ``p`` — the opposite iteration profile from AlphaSlerp.
    Useful for DGGS-style inverse-heavy workloads.

    Inherits from IVEAProjection only for the equilateral-face guard
    and centroid/mids precomputation; the projection math is independent.
    """

    def __init__(self, v0, v1, v2, max_iter: int = 10,
                 tol_step: float = 1e-12) -> None:
        super().__init__(v0, v1, v2)
        self._max_iter = max_iter
        self._tol_step = tol_step

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        b = np.asarray(beta, dtype=float)
        # Warm start: normalize the linear combo (gnomonic-ish).
        p = b[0] * self.V[0] + b[1] * self.V[1] + b[2] * self.V[2]
        n = float(np.linalg.norm(p))
        p = self.centroid.copy() if n < 1e-15 else p / n
        for _ in range(self._max_iter):
            tangent = (b[0] * _log_sphere(p, self.V[0])
                     + b[1] * _log_sphere(p, self.V[1])
                     + b[2] * _log_sphere(p, self.V[2]))
            if float(np.linalg.norm(tangent)) < self._tol_step:
                return p
            p = _exp_sphere(p, tangent)
            p = p / float(np.linalg.norm(p))
        return p

    def to_bary(self, p: Vec3) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        for i in range(3):
            if float(p @ self.V[i]) > 1.0 - 1e-15:
                out = np.zeros(3); out[i] = 1.0; return out
        logs = [_log_sphere(p, self.V[i]) for i in range(3)]
        # 2D basis in the tangent plane at p
        u = np.array([1.0, 0.0, 0.0])
        if abs(float(p @ u)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        e1 = u - float(u @ p) * p
        e1 = e1 / float(np.linalg.norm(e1))
        e2 = np.cross(p, e1)
        A = np.array([
            [float(logs[0] @ e1), float(logs[1] @ e1), float(logs[2] @ e1)],
            [float(logs[0] @ e2), float(logs[1] @ e2), float(logs[2] @ e2)],
            [1.0, 1.0, 1.0],
        ])
        return np.linalg.solve(A, np.array([0.0, 0.0, 1.0]))


class TunedKarcherProjection(KarcherProjection):
    """Karcher with AlphaSlerp's edge-vanishing weight correction.

    Result point ``p`` defined implicitly by ``Σ wᵢ(β) · log_p(Vᵢ) = 0``
    with ``wᵢ(β) = βᵢ · (1 + P · Sᵢ)``, ``P = β₀β₁β₂``,
    ``Sᵢ = η + κ·(βᵢ − 1/3)``. On edges (P = 0), wᵢ collapses to βᵢ
    and Karcher reduces to slerp, so edge-on-arc is preserved
    automatically. (η, κ) = (0, 0) recovers plain Karcher exactly.

    **Empirical finding** (see ``scripts/probe_tuned_karcher_sweep.py``):
    η is **structurally degenerate** because Karcher's implicit
    equation ``Σ wᵢ log_p(Vᵢ) = 0`` is scale-invariant in the weights.
    A uniform-in-i factor like ``(1 + P·η)`` multiplies all wᵢ by the
    same scalar, which doesn't change the equation's solution. Only
    the i-dependent κ·(βᵢ − 1/3) term has any effect. Tuned Karcher
    is therefore effectively a 1-parameter family in κ that moves
    slightly along an existing trade-off curve. No (η, κ) on the
    swept grid Pareto-dominates plain Karcher; best area_r reachable
    is ~1.072 (vs plain Karcher's 1.083, AlphaSlerp's 1.002).

    Useful negative result: the AlphaSlerp-style edge-vanishing trick
    that works for fixed-base exp-map constructions does **not**
    transfer cleanly to Karcher's variable-base construction. To get
    real dials on Tuned Karcher, you'd need higher-order *per-i*
    terms like ``κ₂·(βᵢ² − 1/9)``, not uniform-in-i additive terms.

    Forward: Picard fixed-point with adjusted weights (~10 iters).
    Inverse: fixed-point iteration around Karcher's closed-form 3×3
    linear solve (~3-5 outer iters; one linear solve per iter).
    """

    def __init__(self, v0, v1, v2,
                 eta: float = 0.121, kappa: float = 0.170,
                 max_iter: int = 10, tol_step: float = 1e-12) -> None:
        super().__init__(v0, v1, v2, max_iter=max_iter, tol_step=tol_step)
        self.eta = float(eta)
        self.kappa = float(kappa)

    def _adjusted_weights(self, beta: np.ndarray) -> np.ndarray:
        b = np.asarray(beta, dtype=float)
        P = float(b[0] * b[1] * b[2])
        S = self.eta + self.kappa * (b - 1.0 / 3.0)
        return b * (1.0 + P * S)

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        b = np.asarray(beta, dtype=float)
        w = self._adjusted_weights(b)
        p = w[0] * self.V[0] + w[1] * self.V[1] + w[2] * self.V[2]
        n = float(np.linalg.norm(p))
        p = self.centroid.copy() if n < 1e-15 else p / n
        for _ in range(self._max_iter):
            tangent = (w[0] * _log_sphere(p, self.V[0])
                     + w[1] * _log_sphere(p, self.V[1])
                     + w[2] * _log_sphere(p, self.V[2]))
            if float(np.linalg.norm(tangent)) < self._tol_step:
                return p
            p = _exp_sphere(p, tangent)
            p = p / float(np.linalg.norm(p))
        return p

    def to_bary(self, p: Vec3, max_outer: int = 10,
                tol_outer: float = 1e-13) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        for i in range(3):
            if float(p @ self.V[i]) > 1.0 - 1e-15:
                out = np.zeros(3); out[i] = 1.0; return out
        logs = [_log_sphere(p, self.V[i]) for i in range(3)]
        u = np.array([1.0, 0.0, 0.0])
        if abs(float(p @ u)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        e1 = u - float(u @ p) * p
        e1 = e1 / float(np.linalg.norm(e1))
        e2 = np.cross(p, e1)
        x = np.array([float(g @ e1) for g in logs])
        y = np.array([float(g @ e2) for g in logs])
        rhs = np.array([0.0, 0.0, 1.0])

        # Initial guess: linear-weight Karcher (cᵢ = 1)
        beta = np.linalg.solve(np.array([x, y, np.ones(3)]), rhs)

        # Fixed-point: rebuild with cᵢ from current β, re-solve.
        for _ in range(max_outer):
            P = float(beta[0] * beta[1] * beta[2])
            S = self.eta + self.kappa * (beta - 1.0 / 3.0)
            c = 1.0 + P * S
            A = np.array([c * x, c * y, np.ones(3)])
            beta_new = np.linalg.solve(A, rhs)
            if float(np.max(np.abs(beta_new - beta))) < tol_outer:
                return beta_new
            beta = beta_new
        return beta


class KarcherPolyProjection(KarcherProjection):
    """Karcher with arbitrary-order per-i polynomial weight correction.

    Result point ``p`` defined implicitly by ``Σ wᵢ(β) · log_p(Vᵢ) = 0``
    with::

        Sᵢ(β) = Σ_n κ_n · (βᵢ − 1/3)^n     for n = 1, 2, …, len(kappas)
        wᵢ(β) = βᵢ · (1 + P · Sᵢ)
        P     = β₀ · β₁ · β₂                ← edge-vanishing factor

    On edges (P = 0), wᵢ collapses to βᵢ; Karcher reduces to slerp;
    edge-on-arc preserved automatically.

    Compared to ``TunedKarcherProjection``: this drops the structurally
    degenerate η term and adds higher-order per-i polynomials. ``kappas
    = (κ₁,)`` reproduces TunedKarcher's κ-only family. ``kappas =
    (κ₁, κ₂)`` adds a quadratic dial — the genuinely new degree of
    freedom for Karcher's variable-base construction. Default
    ``kappas = ()`` recovers plain Karcher exactly.
    """

    def __init__(self, v0, v1, v2,
                 kappas=(),
                 max_iter: int = 10, tol_step: float = 1e-12) -> None:
        super().__init__(v0, v1, v2, max_iter=max_iter, tol_step=tol_step)
        self.kappas = tuple(float(k) for k in kappas)

    def _S(self, beta: np.ndarray) -> np.ndarray:
        """Σ_n κ_n · (βᵢ − 1/3)^n for n=1..len(kappas)."""
        b = np.asarray(beta, dtype=float)
        delta = b - 1.0 / 3.0
        S = np.zeros(3)
        d_pow = np.ones(3)
        for k in self.kappas:
            d_pow = d_pow * delta  # δ^n at iteration n
            S = S + k * d_pow
        return S

    def _adjusted_weights(self, beta: np.ndarray) -> np.ndarray:
        b = np.asarray(beta, dtype=float)
        P = float(b[0] * b[1] * b[2])
        return b * (1.0 + P * self._S(b))

    def to_sphere(self, beta: np.ndarray) -> Vec3:
        b = np.asarray(beta, dtype=float)
        w = self._adjusted_weights(b)
        p = w[0] * self.V[0] + w[1] * self.V[1] + w[2] * self.V[2]
        n = float(np.linalg.norm(p))
        p = self.centroid.copy() if n < 1e-15 else p / n
        for _ in range(self._max_iter):
            tangent = (w[0] * _log_sphere(p, self.V[0])
                     + w[1] * _log_sphere(p, self.V[1])
                     + w[2] * _log_sphere(p, self.V[2]))
            if float(np.linalg.norm(tangent)) < self._tol_step:
                return p
            p = _exp_sphere(p, tangent)
            p = p / float(np.linalg.norm(p))
        return p

    def to_bary(self, p: Vec3, max_outer: int = 10,
                tol_outer: float = 1e-13) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        for i in range(3):
            if float(p @ self.V[i]) > 1.0 - 1e-15:
                out = np.zeros(3); out[i] = 1.0; return out
        logs = [_log_sphere(p, self.V[i]) for i in range(3)]
        u = np.array([1.0, 0.0, 0.0])
        if abs(float(p @ u)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        e1 = u - float(u @ p) * p
        e1 = e1 / float(np.linalg.norm(e1))
        e2 = np.cross(p, e1)
        x = np.array([float(g @ e1) for g in logs])
        y = np.array([float(g @ e2) for g in logs])
        rhs = np.array([0.0, 0.0, 1.0])

        beta = np.linalg.solve(np.array([x, y, np.ones(3)]), rhs)
        for _ in range(max_outer):
            P = float(beta[0] * beta[1] * beta[2])
            S = self._S(beta)
            c = 1.0 + P * S
            A = np.array([c * x, c * y, np.ones(3)])
            beta_new = np.linalg.solve(A, rhs)
            if float(np.max(np.abs(beta_new - beta))) < tol_outer:
                return beta_new
            beta = beta_new
        return beta
