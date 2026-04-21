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

omega = cmath.exp(2j * cmath.pi / 3)

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
