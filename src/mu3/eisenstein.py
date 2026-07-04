"""Exact integer Eisenstein arithmetic: ``Z[omega]`` as pairs of ints.

Counterpart of the float-``complex`` arithmetic in
:mod:`mu3.face_lattice`, for code paths that must be exact (the
holonomy neighbor walk). ``z = a + b*omega`` with ``a, b`` Python
ints; ``omega = exp(2*pi*i/3)``, ``omega**2 + omega + 1 = 0``.

The ``complex`` versions stay the tool for plotting and projection;
this module is for lattice combinatorics where float rounding is a
correctness hazard, not a nuisance.

Tie behavior of :func:`divmod_eis` vs the float ``round_ei``
-----------------------------------------------------------

``_round_frac`` rounds half-up while float rounding is half-even (and
in practice FP-noise-dependent). This cannot matter in the walk's
domain:

- Component half-ties need ``x/n = 1/2 (mod 1)`` with ``n = norm(d)``;
  the walk only divides by ``s7a``/``s7b`` (norm 7, odd), so exact
  ties are impossible there.
- The min-norm candidate step makes the remainder choice unique
  whenever a remainder of norm <= 1 exists: any competing remainder
  differs by a nonzero multiple of ``d``, so its norm is at least
  ``(sqrt(7) - 1)**2 ~ 2.7``. Every division the walk performs has a
  digit-offset remainder (norm <= 1), so exact and float agree on
  everything the walk can reach.

Outside that domain (arbitrary points equidistant between lattice
points) the two implementations may legitimately differ. Documented,
not fixed.
"""

import cmath
from functools import lru_cache


class Eis:
    """Eisenstein integer ``a + b*omega``, exact."""

    __slots__ = ('a', 'b')

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def __add__(self, other: 'Eis') -> 'Eis':
        return Eis(self.a + other.a, self.b + other.b)

    def __sub__(self, other: 'Eis') -> 'Eis':
        return Eis(self.a - other.a, self.b - other.b)

    def __neg__(self) -> 'Eis':
        return Eis(-self.a, -self.b)

    def __mul__(self, other: 'Eis') -> 'Eis':
        # omega**2 = -1 - omega
        a1, b1, a2, b2 = self.a, self.b, other.a, other.b
        return Eis(a1 * a2 - b1 * b2, a1 * b2 + b1 * a2 - b1 * b2)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Eis):
            return NotImplemented
        return self.a == other.a and self.b == other.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __repr__(self) -> str:
        return f'Eis({self.a}, {self.b})'

    def conj(self) -> 'Eis':
        """Complex conjugate: ``conj(omega) = omega**2 = -1 - omega``."""
        return Eis(self.a - self.b, -self.b)

    def norm(self) -> int:
        """Field norm ``|z|**2 = a**2 - a*b + b**2`` (a nonnegative int)."""
        return self.a * self.a - self.a * self.b + self.b * self.b

    def to_complex(self) -> complex:
        """Float embedding, for plotting/tests only."""
        return self.a + self.b * _OMEGA_C


_OMEGA_C = cmath.exp(2j * cmath.pi / 3)

ZERO = Eis(0, 0)
ONE = Eis(1, 0)

# UNITS[i] == zeta**i, zeta = 1 + omega = exp(i*pi/3); the source of
# truth for the six units (float tables derive from it via
# to_complex — see cross_pentagon.EISENSTEIN_UNITS). Angle i*60 deg
# CCW from +x.
UNITS: tuple[Eis, ...] = (
    Eis(1, 0),     # zeta**0 =  1
    Eis(1, 1),     # zeta**1 =  1 + omega
    Eis(0, 1),     # zeta**2 =  omega
    Eis(-1, 0),    # zeta**3 = -1
    Eis(-1, -1),   # zeta**4 = -1 - omega
    Eis(0, -1),    # zeta**5 = -omega
)

ZETA = UNITS[1]        # the +60 deg rotation
ZETA_INV = UNITS[5]    # zeta**-1 = -omega

# unit -> exponent i with UNITS[i] == unit
UNIT_EXP: dict[Eis, int] = {u: i for i, u in enumerate(UNITS)}

S7A = Eis(2, -1)   # 2 - omega, norm 7
S7B = Eis(3, 1)    # 3 + omega, norm 7

# 1 - omega = sqrt(3) * exp(-i*30deg): the corner-offset scale (see
# scaled_corner).
S3 = Eis(1, -1)

# Exact counterpart of face_lattice.digit_offset: digit d's offset IS
# the unit at d's angle (digit 6 along the primary direction at 0 deg,
# 1..5 marching CCW from 60 deg; 1 is pentagon-deleted).
DIGIT_OFFSET: dict[int, Eis] = {
    0: ZERO,
    1: UNITS[1],   #  60 deg (pentagon-deleted direction)
    2: UNITS[2],   # 120 deg
    3: UNITS[3],   # 180 deg
    4: UNITS[4],   # 240 deg
    5: UNITS[5],   # 300 deg
    6: UNITS[0],   #   0 deg (primary direction)
}

DIGIT_FOR_OFFSET: dict[Eis, int] = {z: d for d, z in DIGIT_OFFSET.items()}

# Digits in unit-angle order: DIGIT_OFFSET[UNIT_DIGITS[k]] == UNITS[k].
# The corner-row order of cell.cell_boundary and vertex.vertex_directions.
UNIT_DIGITS = (6, 1, 2, 3, 4, 5)


_CANDIDATES = (ZERO, *UNITS)


def from_complex(w: complex) -> Eis:
    """Nearest Eisenstein integer to ``w`` (honeycomb rounding) —
    the exact-snap counterpart of ``face_lattice.round_ei``.

    Component rounding lands within one unit hex of the true nearest
    lattice point; the min-distance search over ``q0 + {0, six
    units}`` finishes the job (same envelope as :func:`divmod_eis`).
    A ``w`` exactly equidistant between lattice points resolves
    arbitrarily — callers own that boundary (point location follows
    with a spherical polish step that absorbs it).
    """
    b = w.imag / _OMEGA_C.imag
    a = w.real - b * _OMEGA_C.real
    q0 = Eis(round(a), round(b))
    return min(
        (q0 + c for c in _CANDIDATES),
        key=lambda q: abs(w - q.to_complex()),
    )


def ratio(k: int) -> Eis:
    """Aperture-7 level ratio at level ``k``: ``s7b`` odd, ``s7a`` even."""
    return S7B if k % 2 == 1 else S7A


@lru_cache(maxsize=None)
def get_rot_eis(res: int) -> Eis:
    """Exact counterpart of ``face_lattice.get_rot``.

    Product of ``ratio(k)`` for ``k = 1..res``; since
    ``s7a * s7b == 7``, this equals ``7**(res//2)`` times ``s7b`` for
    odd ``res``. Cached — it sits inside the walk primitive
    (``neighbor._resolve``), one call per walk.
    """
    out = ONE
    for k in range(1, res + 1):
        out = out * ratio(k)
    return out


def in_deleted_wedge(z: Eis, res: int) -> bool:
    """True iff a position in the S3-scaled res-``res`` frame develops
    into the pentagon-frame deleted wedge ``[0deg, 60deg)`` — the exact
    counterpart of the float ``cell._stitch`` angle test.

    Multiplying by ``conj(S3 * get_rot_eis(res))`` rescales by a
    positive real onto the pentagon-frame position; the wedge is then
    the exact cone ``b >= 0 and a > b`` (angle-0 ray included, angle-60
    ray and the origin excluded, matching ``_stitch``).
    """
    w = z * (S3 * get_rot_eis(res)).conj()
    return w.b >= 0 and w.a > w.b


def _round_frac(x: int, n: int) -> int:
    """Nearest integer to ``x / n`` for ``n > 0`` (half-up; see module
    docstring for why the tie rule is immaterial in-domain)."""
    return (2 * x + n) // (2 * n)


def divmod_eis(z: Eis, d: Eis) -> tuple[Eis, Eis]:
    """Exact Euclidean division in ``Z[omega]``: ``(q, r)`` with
    ``z == q*d + r`` and ``r`` of minimal norm (honeycomb rounding).

    ``z/d = z*conj(d)/norm(d)`` exactly; component rounding gives a
    quotient within one unit hex of the honeycomb-nearest point, so
    the min-norm search over ``q0 + {0, six units}`` is sufficient
    (same envelope the float ``round_ei`` two-stage correction
    handles).

    Early exit: when ``4*norm(r0) < norm(d)`` the rounded quotient's
    remainder is UNIQUELY minimal — any competitor differs by a
    nonzero multiple of ``d``, so ``|r'| >= |d| - |r0| > |d|/2 >
    |r0|`` — and the search is skipped. In the walk's domain (norm-7
    divisors, digit-offset remainders of norm <= 1) this always
    fires; the search survives for boundary/out-of-domain inputs.
    """
    n = d.norm()
    w = z * d.conj()
    q0 = Eis(_round_frac(w.a, n), _round_frac(w.b, n))
    best_q, best_r = q0, z - q0 * d
    best_norm = best_r.norm()
    if 4 * best_norm < n:
        return best_q, best_r
    for c in UNITS:
        q = q0 + c
        r = z - q * d
        rn = r.norm()
        if rn < best_norm:
            best_q, best_r, best_norm = q, r, rn
    return best_q, best_r


def scaled_center(digits) -> Eis:
    """Exact cell-center position in res-N lattice units,
    ``N = len(digits)``: the exact counterpart of
    ``cell._eisenstein_center(digits) * get_rot(N)``."""
    out = ZERO
    factor = ONE   # product of ratio(j) for j = k+1 .. N
    for k in range(len(digits), 0, -1):
        out = out + DIGIT_OFFSET[digits[k - 1]] * factor
        factor = factor * ratio(k)
    return out


def scaled_corner(digits, d: int) -> Eis:
    """Exact corner position in the S3-scaled res-N frame,
    ``N = len(digits)``: cell corners sit at ``center + unit/s3``, so
    S3-scaling lands them on an exact lattice (the aperture-3
    refinement of the cell lattice). The single home of the corner
    formula — ``cell.cell_boundary`` rows and ``mu3.vertex``
    positions/identity keys both build on it."""
    return S3 * scaled_center(digits) + DIGIT_OFFSET[d]


def extract_digits(z: Eis, res: int) -> tuple[list[int], Eis]:
    """Inverse of :func:`scaled_center`, with residue: iterated exact
    divmod on a scaled position, finest level first. Returns
    ``(digits, root)`` — ``root`` is zero when ``z`` is a canonical
    cell-center position, and a unit (a root-level direction) when it
    lies beyond the root cell. Exact counterpart of the float
    ``cross_pentagon.z_to_cell``."""
    digits = [0] * res
    cur = z
    for k in range(res, 0, -1):
        cur, r = divmod_eis(cur, ratio(k))
        d = DIGIT_FOR_OFFSET.get(r)
        if d is None:
            raise RuntimeError(
                f'extract_digits: bad remainder at level {k}: {r}'
            )
        digits[k - 1] = d
    return digits, cur


def _build_carry_table(ratio_p: Eis) -> tuple:
    """Digit-carry transitions for one level parity: entry ``[d][e]``
    is ``(d_new, e_carry | None)`` with

        DIGIT_OFFSET[d] + UNITS[e]
            == ratio_p * UNITS[e_carry] + DIGIT_OFFSET[d_new]

    (``None`` = quotient zero: the carry absorbs). One exact divmod
    per entry — derived, never hand-entered — with the closure facts
    (quotient is zero or a unit; remainder is a digit offset) asserted
    here at import, cocycle-style."""
    table = []
    for d in range(7):
        row = []
        for e in range(6):
            q, r = divmod_eis(DIGIT_OFFSET[d] + UNITS[e], ratio_p)
            assert r in DIGIT_FOR_OFFSET, (d, e, r)
            assert q == ZERO or q in UNIT_EXP, (d, e, q)
            row.append((DIGIT_FOR_OFFSET[r],
                        None if q == ZERO else UNIT_EXP[q]))
        table.append(tuple(row))
    return tuple(table)


# CARRY[k % 2][digit][unit_exp] — the per-level transition of a
# bottom-up single-unit step (level k divides by ratio(k): S7B odd,
# S7A even).
CARRY = (_build_carry_table(S7A), _build_carry_table(S7B))

# carry_digits' hot-path shortcut, checked where the tables are built:
# the digit numbering aligns with the unit exponents.
assert all(DIGIT_OFFSET[d] == UNITS[d % 6] for d in range(1, 7))


def carry_digits(digits, d: int) -> tuple[int, ...] | None:
    """Digit string of ``center(digits) + unit(d)`` by bottom-up
    carry, or ``None`` if the carry escapes the root (the position
    lies beyond the digit tree; the caller owns what that means).

    Ring-1 only: one unit step at the finest level. The carry absorbs
    with probability ~5/7 per level, so the amortized cost is O(1)
    table lookups — the H3-style fast path, with the tables derived
    from :func:`divmod_eis` instead of hand-entered. The result may
    carry a leading nonzero digit 1 — a non-canonical form; callers
    decide (see ``neighbor._step_fast``).
    """
    out = list(digits)
    e = d % 6                     # == UNIT_EXP[DIGIT_OFFSET[d]]
    for k in range(len(out), 0, -1):
        d_new, e = CARRY[k % 2][out[k - 1]][e]
        out[k - 1] = d_new
        if e is None:
            return tuple(out)
    return None


def first_nonzero_digit(digits) -> int | None:
    """The leading nonzero digit of a digit sequence (``None`` if all
    zero). The single expression behind the leading-digit rule: a
    leading 1 is the pentagon-deleted form — excluded from canonical
    cells (``cell.is_valid_cell``/``cells_at_res``) and twin-fixed by
    the walk (``neighbor._resolve``/``_step_fast``)."""
    return next((d for d in digits if d != 0), None)


def hex_dist(z: Eis) -> int:
    """Hex-lattice graph distance from the origin to ``z``, where the
    unit steps are the six units: ``(|a| + |b| + |a - b|) // 2``.

    (Same-sign components combine along the ``1 + omega`` diagonal —
    ``max(|a|, |b|)`` steps; opposite signs cannot combine —
    ``|a| + |b|``; the closed form covers both.)
    """
    return (abs(z.a) + abs(z.b) + abs(z.a - z.b)) // 2


def line_digits(delta: Eis) -> list[int]:
    """Digit steps of an evenly interleaved hex geodesic from the
    origin to ``delta`` — the constructive counterpart of
    :func:`hex_dist` (``len == hex_dist(delta)``, and the offsets sum
    to ``delta``). Even interleaving keeps the walked line within
    about one lattice unit of the straight segment, a contract the
    traversal pair-query guard budgets for.
    """
    a, b = delta.a, delta.b
    legs: list[tuple[int, int]] = []
    if a * b > 0:
        s = 1 if a > 0 else -1
        n = min(abs(a), abs(b))
        legs.append((DIGIT_FOR_OFFSET[Eis(s, s)], n))
        a -= s * n
        b -= s * n
    if a:
        legs.append((DIGIT_FOR_OFFSET[Eis(1 if a > 0 else -1, 0)], abs(a)))
    if b:
        legs.append((DIGIT_FOR_OFFSET[Eis(0, 1 if b > 0 else -1)], abs(b)))
    if not legs:
        return []
    if len(legs) == 1:
        d, n = legs[0]
        return [d] * n
    (d1, n1), (d2, n2) = legs
    if n1 < n2:
        (d1, n1), (d2, n2) = (d2, n2), (d1, n1)
    out = []
    acc = 0
    for _ in range(n1):
        out.append(d1)
        acc += n2
        if acc >= n1:
            out.append(d2)
            acc -= n1
    return out
