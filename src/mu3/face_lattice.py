"""Per-face Eisenstein-integer hex lattice arithmetic (aperture 7).

Adapted from prototype code in ``../2026-04-02_eisint/src/eisint/eisint.py``.
Math stays in the 2D complex plane of a single icosahedron face; cross-face
bookkeeping lives in the indexing layer.

Digit labeling: digits 1..6 march strictly CCW around the parent hex,
anchored to the per-pentagon primary direction from ``mu3.dodec``. Digit 6
sits along the primary direction (angle 0); digit 1 is the next wedge CCW
(angle 60), and is the pentagon-deleted direction:

    0 = center
    1 =  60°   (pentagon-deleted direction; immediately CCW of primary)
    2 = 120°
    3 = 180°
    4 = 240°
    5 = 300°
    6 =   0°   (along the primary direction)

CCW 60° rotation simply increments digit 1..6: ``1 → 2 → 3 → 4 → 5 → 6 → 1``.

Aperture-7 per-resolution rotation alternates (Class II / Class III):
even resolutions are unrotated relative to res 0, odd resolutions are
rotated by ``arg(s7b) ≈ +19.106°``.

See ``reports/nearest-dodecahedron-face.md`` (Option A primary direction)
and ``figures/primary_direction_convention.png``.
"""

from __future__ import annotations

import cmath
import math

omega = cmath.exp(2j * cmath.pi / 3)

# Face-2D distance from face center to face corner, on the unit icosahedron.
# Equals tan(arccos(V · C_f)) where V is an icosa vertex and C_f is an
# incident face center; closed form = sqrt(14 - 6*sqrt(5)) ≈ 0.7639.
r_face = math.sqrt(14 - 6 * math.sqrt(5))

units = (
    1,
    1 + omega,
    omega,
    -1,
    -1 - omega,
    -omega,
)

digit_offset: dict[int, complex] = {
    0: 0,
    1: 1 + omega,   #  60°  (DELETED in pentagons; immediately CCW of primary)
    2: omega,       # 120°
    3: -1,          # 180°
    4: -1 - omega,  # 240°
    5: -omega,      # 300°
    6: 1,           #   0°  (along the primary direction)
}

pentagon_skipped_digit = 1

# Sequential +60° CCW digit cycle: 1 → 2 → 3 → 4 → 5 → 6 → 1.
_DIGIT_CCW_NEXT = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}


def rotate_digit_ccw(d: int, steps: int = 1) -> int:
    """Apply ``steps`` CCW 60° rotations to digit ``d``."""
    steps %= 6
    for _ in range(steps):
        d = _DIGIT_CCW_NEXT[d]
    return d


_OFFSET_TO_DIGIT: dict[tuple[int, int], int] = {}
for _d, _off in digit_offset.items():
    _b_imag = _off.imag / omega.imag
    _a_real = _off.real - _b_imag * omega.real
    _OFFSET_TO_DIGIT[(round(_a_real), round(_b_imag))] = _d


def digit_for_offset(z: complex) -> int:
    """Digit whose ``digit_offset`` equals ``z``; -1 if ``z`` is not one.

    ``z`` must be one of the 7 Eisenstein offsets (within 1e-9). The caller
    is expected to feed values produced by the divmod-driven neighbor walk,
    where this is guaranteed by construction.
    """
    b = z.imag / omega.imag
    a = z.real - b * omega.real
    key = (round(a), round(b))
    d = _OFFSET_TO_DIGIT.get(key, -1)
    if d < 0 or abs(z - digit_offset[d]) > 1e-9:
        return -1
    return d

s3 = 1 - omega
s7a = 2 - omega
s7b = 3 + omega


def get_rot(res: int) -> complex:
    """Accumulated scale+rotation from res 0 to the given resolution.

    ``|get_rot(res)|² = 7**res``: scale by √7 per resolution step. The
    rotational component alternates — even resolutions are Class II (no
    rotation vs res 0), odd resolutions are Class III (rotated by
    ``arg(s7b) ≈ +19.106°``).
    """
    out: complex = 7 ** (res // 2)
    if res % 2 == 1:
        out *= s7b
    return out


def to_ab(z: complex) -> tuple[float, float]:
    """Decompose ``z`` into Eisenstein components: ``z = a + b*ω``."""
    b = z.imag / omega.imag
    a = z.real - b * omega.real
    return a, b


def round_ei(z: complex) -> complex:
    """Round ``z`` to the nearest Eisenstein integer (honeycomb rounding)."""
    a, b = to_ab(z)
    p = round(a) + round(b) * omega
    f = z - p

    r = f.real
    s = (f * omega.conjugate()).real

    if abs(r) >= abs(s) and abs(r) > 0.5:
        p += round(r)
    elif abs(s) > 0.5:
        p += round(s) * omega

    return p


def divmod_ei(z: complex, d: complex) -> tuple[complex, complex]:
    """Euclidean division in Z[ω]: returns ``(quotient, remainder)``."""
    q = round_ei(z / d)
    r = z - q * d
    return q, r


def _neighbor_step_single(d: int, D: int, parity: int) -> tuple[int, int]:
    """Bottom-up neighbor-walk transition at one resolution level.

    Given current digit ``d`` and incoming direction ``D`` to add at the
    same resolution, returns ``(d_new, D_carry)`` — the new digit at this
    resolution and the direction to add at the parent resolution. ``parity``
    is the resolution's parity (1 = odd → ratio ``s7b``, 0 = even → ``s7a``),
    reflecting the Class III / Class II alternation of :func:`get_rot`.

    Derived from the algebraic identity
    ``offset[d] + offset[D] = offset[d_new] + offset[D_carry] * ratio``,
    where ``ratio = get_rot(res) / get_rot(res - 1)``. The RHS is obtained
    by Euclidean division in Z[ω] by ``ratio``.
    """
    ratio = s7b if parity == 1 else s7a
    S = digit_offset[d] + digit_offset[D]
    if abs(S) < 1e-12:
        return 0, 0
    q, r = divmod_ei(S, ratio)
    D_carry = digit_for_offset(q)
    d_new = digit_for_offset(r)
    if D_carry < 0 or d_new < 0:
        raise RuntimeError(
            f"neighbor-step transition failed: d={d} D={D} parity={parity} "
            f"S={S} q={q} r={r}"
        )
    return d_new, D_carry


# NEIGHBOR_TRANS[parity][d][D] = (d_new, D_carry).
# Precomputed once at import — 2·7·7 = 98 entries per parity.
NEIGHBOR_TRANS: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = tuple(
    tuple(
        tuple(_neighbor_step_single(d, D, parity) for D in range(7))
        for d in range(7)
    )
    for parity in range(2)
)
