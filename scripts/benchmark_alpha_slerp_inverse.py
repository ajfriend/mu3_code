"""Benchmark candidate inversion methods for AlphaSlerp.

Forward map is in ``src/mu3/projection.py`` (``AlphaSlerp.forward_barycentric``).
The inverse maps a unit sphere point ``q`` on a face back to its barycentric
``β`` on that face. This script compares five candidate solvers on three
strata (interior / edge-heavy / corner-heavy) and reports iteration counts,
wall time, accuracy, and failure counts.

Edit the variables at the top of ``main()`` to change seed, sample size, or
tolerance.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mu3 import icosahedron
from mu3.projection import ALPHA_SLERP_DEFAULTS, AlphaSlerp


# ---------------------------------------------------------------------------
# Face geometry helpers (closure over the AlphaSlerp instance)
# ---------------------------------------------------------------------------


@dataclass
class FaceGeom:
    """Per-face cached geometry shared across solvers."""
    slerp: AlphaSlerp
    u: np.ndarray          # face-plane orthonormal basis vector 1
    v: np.ndarray          # face-plane orthonormal basis vector 2
    e1: np.ndarray         # V1 - V0
    e2: np.ndarray         # V2 - V0
    g11: float
    g12: float
    g22: float
    det_g: float


def make_face_geom(face_idx: int) -> FaceGeom:
    V = icosahedron.vertices()
    F = icosahedron.faces()
    v0, v1, v2 = V[F[face_idx, 0]], V[F[face_idx, 1]], V[F[face_idx, 2]]
    slerp = AlphaSlerp(v0, v1, v2)

    u_raw = v1 - v0 - np.dot(v1 - v0, slerp.n) * slerp.n
    u = u_raw / np.linalg.norm(u_raw)
    v = np.cross(slerp.n, u)

    e1 = v1 - v0
    e2 = v2 - v0
    g11 = float(e1 @ e1)
    g12 = float(e1 @ e2)
    g22 = float(e2 @ e2)
    det_g = g11 * g22 - g12 * g12
    return FaceGeom(slerp, u, v, e1, e2, g11, g12, g22, det_g)


# ---------------------------------------------------------------------------
# Forward + residual
# ---------------------------------------------------------------------------


def forward(fg: FaceGeom, beta: np.ndarray) -> np.ndarray:
    return fg.slerp.forward_barycentric(beta)


def residual_2d(fg: FaceGeom, beta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(F(β) - q) projected onto the face-plane (u, v) basis."""
    diff = forward(fg, beta) - q
    return np.array([diff @ fg.u, diff @ fg.v])


def warm_start(fg: FaceGeom, q: np.ndarray) -> np.ndarray:
    """Project q orthogonally onto the face plane and read off Euclidean barycentric."""
    V = fg.slerp.V
    n = fg.slerp.n
    r = q - np.dot(q - V[0], n) * n
    d = r - V[0]
    d1 = float(d @ fg.e1)
    d2 = float(d @ fg.e2)
    b1 = (fg.g22 * d1 - fg.g12 * d2) / fg.det_g
    b2 = (fg.g11 * d2 - fg.g12 * d1) / fg.det_g
    b0 = 1.0 - b1 - b2
    return np.array([b0, b1, b2])


# ---------------------------------------------------------------------------
# Analytic Jacobian of F w.r.t. (β0, β1)
# ---------------------------------------------------------------------------

# Reduction matrix: ∂β_i/∂β_j for j ∈ {0,1}. Row i, column j.
_REDUCE = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])


def analytic_jacobian(fg: FaceGeom, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (F(β), dF/d(β0,β1)) as (3,) and (3,2) arrays.

    β is (3,); the two independent unknowns are β0 and β1 with β2 = 1 − β0 − β1.
    """
    s = fg.slerp
    V = s.V          # (3, 3)
    alpha = s.alpha
    omega = s.omega
    w0 = s._w0       # ω − α
    sin_w = s._sin_omega
    eta = s.eta
    kappa = s.kappa

    b = beta
    P = float(b[0] * b[1] * b[2])
    dP = np.array([b[1] * b[2], b[0] * b[2], b[0] * b[1]])           # ∂P/∂β_i
    S = eta + kappa * (b - 1.0 / 3.0)
    f = alpha * b + 3.0 * w0 * b ** 2 - 2.0 * w0 * b ** 3
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    fprime = alpha + 6.0 * w0 * b * (1.0 - b)                        # f_i'(β_i)
    weights = (1.0 + P * S) * sin_f / sin_w
    v_star = V.T @ weights                                           # (3,)

    n = s.n
    n_dot = float(n @ v_star)
    disc = 1.0 + n_dot ** 2 - float(v_star @ v_star)
    sqrt_disc = math.sqrt(max(disc, 0.0))
    p = -n_dot + sqrt_disc
    v_vec = v_star + p * n
    v_norm = float(np.linalg.norm(v_vec))
    F_val = v_vec / v_norm

    # ∂β_i/∂β_j — use _REDUCE
    # dP/dβ_j = Σ_i dP/dβ_i · _REDUCE[i, j]  (shape (2,))
    dP_dj = dP @ _REDUCE  # (2,)
    dS_dj = kappa * _REDUCE  # (3, 2)  — ∂S_i/∂β_j
    df_dj = (fprime[:, None]) * _REDUCE  # (3, 2)  — ∂f_i/∂β_j

    # dw_i/dβ_j =
    #   [dP/dβ_j * S_i + P * dS_i/dβ_j] * sin(f_i)/sinω
    # + (1 + P*S_i) * cos(f_i) * df_i/dβ_j / sinω
    term_amp = (dP_dj[None, :] * S[:, None] + P * dS_dj) * (sin_f / sin_w)[:, None]
    term_phase = ((1.0 + P * S) * cos_f)[:, None] * df_dj / sin_w
    dw_dj = term_amp + term_phase                                    # (3, 2)

    dv_star_dj = V.T @ dw_dj                                         # (3, 2)
    dn_dot_dj = n @ dv_star_dj                                       # (2,)
    ddisc_dj = 2.0 * n_dot * dn_dot_dj - 2.0 * (v_star @ dv_star_dj) # (2,)
    dp_dj = dn_dot_dj * -1.0 + ddisc_dj / (2.0 * sqrt_disc)          # (2,)
    dv_dj = dv_star_dj + n[:, None] * dp_dj[None, :]                 # (3, 2)

    # F = v / |v|
    vv = v_vec @ dv_dj                                               # (2,)
    dF_dj = (dv_dj - F_val[:, None] * vv[None, :]) / v_norm          # (3, 2)

    return F_val, dF_dj


# ---------------------------------------------------------------------------
# Solvers — each returns (beta, iters, converged)
# ---------------------------------------------------------------------------


TOL_BETA = 1e-13
TOL_RES = 1e-14
MAX_ITERS = 20


def _apply_step_3d(beta: np.ndarray, db2: np.ndarray) -> np.ndarray:
    """Apply a (β0, β1) step with β2 = 1 - β0 - β1."""
    new = beta.copy()
    new[0] += db2[0]
    new[1] += db2[1]
    new[2] = 1.0 - new[0] - new[1]
    return new


def solve_fd_newton(fg: FaceGeom, q: np.ndarray, h: float = 1e-7) -> tuple[np.ndarray, int, bool]:
    beta = warm_start(fg, q)
    for k in range(1, MAX_ITERS + 1):
        r = residual_2d(fg, beta, q)
        if float(r @ r) < TOL_RES ** 2:
            return beta, k - 1, True
        # Forward-difference Jacobian, 2 extra forward evals
        beta_h0 = _apply_step_3d(beta, np.array([h, 0.0]))
        beta_h1 = _apply_step_3d(beta, np.array([0.0, h]))
        r0 = residual_2d(fg, beta_h0, q)
        r1 = residual_2d(fg, beta_h1, q)
        J = np.column_stack([(r0 - r) / h, (r1 - r) / h])
        try:
            db2 = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return beta, k, False
        beta = _apply_step_3d(beta, db2)
        if float(db2 @ db2) < TOL_BETA ** 2:
            return beta, k, True
    return beta, MAX_ITERS, False


def solve_central_newton(fg: FaceGeom, q: np.ndarray, h: float = 1e-6) -> tuple[np.ndarray, int, bool]:
    beta = warm_start(fg, q)
    for k in range(1, MAX_ITERS + 1):
        r = residual_2d(fg, beta, q)
        if float(r @ r) < TOL_RES ** 2:
            return beta, k - 1, True
        b_p0 = _apply_step_3d(beta, np.array([h, 0.0]))
        b_m0 = _apply_step_3d(beta, np.array([-h, 0.0]))
        b_p1 = _apply_step_3d(beta, np.array([0.0, h]))
        b_m1 = _apply_step_3d(beta, np.array([0.0, -h]))
        r_p0 = residual_2d(fg, b_p0, q)
        r_m0 = residual_2d(fg, b_m0, q)
        r_p1 = residual_2d(fg, b_p1, q)
        r_m1 = residual_2d(fg, b_m1, q)
        J = np.column_stack([(r_p0 - r_m0) / (2 * h), (r_p1 - r_m1) / (2 * h)])
        try:
            db2 = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return beta, k, False
        beta = _apply_step_3d(beta, db2)
        if float(db2 @ db2) < TOL_BETA ** 2:
            return beta, k, True
    return beta, MAX_ITERS, False


def solve_analytic_newton(fg: FaceGeom, q: np.ndarray) -> tuple[np.ndarray, int, bool]:
    beta = warm_start(fg, q)
    for k in range(1, MAX_ITERS + 1):
        F_val, dF_dj = analytic_jacobian(fg, beta)
        diff = F_val - q
        r = np.array([diff @ fg.u, diff @ fg.v])
        if float(r @ r) < TOL_RES ** 2:
            return beta, k - 1, True
        # 2D Jacobian of residual = [u · dF/dβ_j, v · dF/dβ_j]
        J = np.array([fg.u @ dF_dj, fg.v @ dF_dj])
        try:
            db2 = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return beta, k, False
        beta = _apply_step_3d(beta, db2)
        if float(db2 @ db2) < TOL_BETA ** 2:
            return beta, k, True
    return beta, MAX_ITERS, False


def _precompute_centroid_jacobian(fg: FaceGeom) -> np.ndarray:
    """Analytic 2x2 residual Jacobian evaluated at the face centroid β=(1/3,1/3,1/3)."""
    centroid = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    _, dF_dj = analytic_jacobian(fg, centroid)
    return np.array([fg.u @ dF_dj, fg.v @ dF_dj])


def solve_picard(fg: FaceGeom, q: np.ndarray, J_inv_cached: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Fixed-point with precomputed Jacobian inverse at face centroid."""
    beta = warm_start(fg, q)
    for k in range(1, MAX_ITERS + 1):
        r = residual_2d(fg, beta, q)
        if float(r @ r) < TOL_RES ** 2:
            return beta, k - 1, True
        db2 = -J_inv_cached @ r
        beta = _apply_step_3d(beta, db2)
        if float(db2 @ db2) < TOL_BETA ** 2:
            return beta, k, True
    return beta, MAX_ITERS, False


def solve_hybrid(fg: FaceGeom, q: np.ndarray, J_inv_cached: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """One Picard step to polish warm start, then analytic Newton."""
    beta = warm_start(fg, q)
    r = residual_2d(fg, beta, q)
    db2 = -J_inv_cached @ r
    beta = _apply_step_3d(beta, db2)
    # Continue with analytic Newton
    for k in range(1, MAX_ITERS):
        F_val, dF_dj = analytic_jacobian(fg, beta)
        diff = F_val - q
        r = np.array([diff @ fg.u, diff @ fg.v])
        if float(r @ r) < TOL_RES ** 2:
            return beta, k, True  # +1 Picard step accounted for? We report Newton iters only + 1 prelude
        J = np.array([fg.u @ dF_dj, fg.v @ dF_dj])
        try:
            db2 = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return beta, k + 1, False
        beta = _apply_step_3d(beta, db2)
        if float(db2 @ db2) < TOL_BETA ** 2:
            return beta, k + 1, True
    return beta, MAX_ITERS, False


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def sample_strata(seed: int, n: int) -> dict[str, np.ndarray]:
    """Return dict stratum_name → (n, 3) array of β samples on the simplex."""
    rng = np.random.default_rng(seed)
    strata = {}
    strata["interior"] = rng.dirichlet([2.0, 2.0, 2.0], size=n)
    strata["edge_heavy"] = rng.dirichlet([5.0, 5.0, 0.2], size=n)
    strata["corner_heavy"] = rng.dirichlet([10.0, 0.3, 0.3], size=n)
    return strata


@dataclass
class MethodResult:
    name: str
    stratum: str
    iters_mean: float
    iters_p50: float
    iters_p99: float
    iters_max: int
    fail_count: int
    wall_us_per_call: float
    beta_err_max: float
    residual_max: float


def run_single(
    solver: Callable, fg: FaceGeom, betas_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run solver on each sample. Returns (iters, beta_errs, residuals, wall_s)."""
    n = betas_true.shape[0]
    iters = np.zeros(n, dtype=np.int64)
    beta_errs = np.zeros(n)
    residuals = np.zeros(n)
    conv = np.zeros(n, dtype=bool)

    # Pre-generate q's via forward map
    qs = np.array([forward(fg, betas_true[i]) for i in range(n)])

    t0 = time.perf_counter()
    for i in range(n):
        b, it, ok = solver(fg, qs[i])
        iters[i] = it
        conv[i] = ok
        beta_errs[i] = float(np.linalg.norm(b - betas_true[i]))
        residuals[i] = float(np.linalg.norm(forward(fg, b) - qs[i]))
    wall = time.perf_counter() - t0
    return iters, beta_errs, residuals, wall, conv


def summarize(name: str, stratum: str, iters, beta_errs, residuals, wall, conv) -> MethodResult:
    n = len(iters)
    fail = int((~conv).sum())
    return MethodResult(
        name=name,
        stratum=stratum,
        iters_mean=float(iters.mean()),
        iters_p50=float(np.percentile(iters, 50)),
        iters_p99=float(np.percentile(iters, 99)),
        iters_max=int(iters.max()),
        fail_count=fail,
        wall_us_per_call=wall / n * 1e6,
        beta_err_max=float(beta_errs.max()),
        residual_max=float(residuals.max()),
    )


def print_table(results: list[MethodResult]) -> None:
    hdr = (f"{'method':<22} {'stratum':<14} "
           f"{'iters_μ':>8} {'p50':>4} {'p99':>4} {'max':>4} "
           f"{'fail':>5} {'μs/call':>8} "
           f"{'β_err_max':>10} {'res_max':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.name:<22} {r.stratum:<14} "
            f"{r.iters_mean:>8.2f} {r.iters_p50:>4.0f} {r.iters_p99:>4.0f} {r.iters_max:>4d} "
            f"{r.fail_count:>5d} {r.wall_us_per_call:>8.2f} "
            f"{r.beta_err_max:>10.2e} {r.residual_max:>10.2e}"
        )


def main() -> None:
    # ---- knobs ----
    SEED = 42
    N = 2000
    FACE_IDX = 0          # any face (map is similar across faces)
    # ---------------

    fg = make_face_geom(FACE_IDX)
    strata = sample_strata(SEED, N)

    J_centroid = _precompute_centroid_jacobian(fg)
    J_inv_centroid = np.linalg.inv(J_centroid)

    methods = [
        ("fd_newton", lambda fg, q: solve_fd_newton(fg, q)),
        ("central_newton", lambda fg, q: solve_central_newton(fg, q)),
        ("analytic_newton", lambda fg, q: solve_analytic_newton(fg, q)),
        ("picard", lambda fg, q: solve_picard(fg, q, J_inv_centroid)),
        ("hybrid_picard+newton", lambda fg, q: solve_hybrid(fg, q, J_inv_centroid)),
    ]

    print(f"α-slerp inverse benchmark — seed={SEED}, N={N}, face={FACE_IDX}, "
          f"(α,η,κ)={ALPHA_SLERP_DEFAULTS}")
    print(f"tol_res={TOL_RES:.0e}, tol_β={TOL_BETA:.0e}, max_iters={MAX_ITERS}\n")

    results = []
    for stratum_name, betas in strata.items():
        for name, solver in methods:
            iters, beta_errs, residuals, wall, conv = run_single(solver, fg, betas)
            results.append(summarize(name, stratum_name, iters, beta_errs, residuals, wall, conv))

    print_table(results)

    print("\nLegend:")
    print("  iters = solver iterations to meet tol_res or tol_β (not counting the warm start)")
    print("  μs/call = wall-clock time per inversion, including forward eval overhead")
    print("  β_err_max = max ‖β_recovered − β_true‖ over samples in stratum")
    print("  res_max = max ‖F(β_recovered) − q‖ over samples in stratum")


if __name__ == "__main__":
    main()
