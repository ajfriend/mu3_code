"""Per-face Eisenstein-integer hex lattice arithmetic (aperture 7).

Adapted from prototype code in ``../2026-04-02_eisint/src/eisint/eisint.py``.
Math stays in the 2D complex plane of a single icosahedron face; cross-face
bookkeeping lives in the indexing layer.

Digit labeling follows H3's convention (see
``/Users/aj/work/h3/src/h3lib/include/coordijk.h``):

    0 = center
    1 = k        (pentagon-deleted direction)
    2 = j
    3 = j+k  = -i
    4 = i
    5 = i+k  = -j
    6 = i+j  = -k

CCW 60° rotation cycles the digits ``1 → 5 → 4 → 6 → 2 → 3 → 1``.

The 2D embedding places H3's i-axis at angle 0° (aligned with the face's
local ``u`` direction from :func:`mu3.icosahedron.face_frames`), j at 120°,
k at 240°.

Aperture-7 per-resolution rotation alternates (Class II / Class III):
even resolutions are unrotated relative to res 0, odd resolutions are
rotated by ``arg(s7b) ≈ +19.106°``.
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

h3_digit_offset: dict[int, complex] = {
    0: 0,
    1: -1 - omega,  # k,     240°
    2: omega,       # j,     120°
    3: -1,          # j+k,   180°
    4: 1,           # i,       0°
    5: -omega,      # i+k,   300°
    6: 1 + omega,   # i+j,    60°
}

pentagon_skipped_digit = 1

# H3 CCW digit cycle from coordijk.h: a +60° CCW rotation of an Eisenstein
# offset cycles digits 1 → 5 → 4 → 6 → 2 → 3 → 1.
_DIGIT_CCW_NEXT = {0: 0, 1: 5, 5: 4, 4: 6, 6: 2, 2: 3, 3: 1}


def rotate_digit_ccw(d: int, steps: int = 1) -> int:
    """Apply ``steps`` CCW 60° rotations to digit ``d``."""
    steps %= 6
    for _ in range(steps):
        d = _DIGIT_CCW_NEXT[d]
    return d


_OFFSET_TO_DIGIT: dict[tuple[int, int], int] = {}
for _d, _off in h3_digit_offset.items():
    _b_imag = _off.imag / omega.imag
    _a_real = _off.real - _b_imag * omega.real
    _OFFSET_TO_DIGIT[(round(_a_real), round(_b_imag))] = _d


def digit_for_offset(z: complex) -> int:
    """Digit whose ``h3_digit_offset`` equals ``z``; -1 if ``z`` is not one.

    ``z`` must be one of the 7 Eisenstein offsets (within 1e-9). The caller
    is expected to feed values produced by the divmod-driven neighbor walk,
    where this is guaranteed by construction.
    """
    b = z.imag / omega.imag
    a = z.real - b * omega.real
    key = (round(a), round(b))
    d = _OFFSET_TO_DIGIT.get(key, -1)
    if d < 0 or abs(z - h3_digit_offset[d]) > 1e-9:
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
    S = h3_digit_offset[d] + h3_digit_offset[D]
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
