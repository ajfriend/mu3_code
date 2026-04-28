# /// script
# dependencies = ["numpy"]
# ///
"""Sketch: rebuild pentagon_face_table from the (family, s1, s2) tuple
with no lookups or per-vertex geometric CCW sort.

Produces a face table in *standard-position* coordinates. The icosahedron
vertices, faces, and pentagon corners are all derived from the tuple
structure. The output is comparable to mu3.icosahedron.pentagon_face_table()
up to the rotation that takes pole-up layout to standard position and the
choice of digit-2 offset convention.

Runs standalone; does not import mu3.
"""

from __future__ import annotations

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
NORM = np.sqrt(1 + PHI**2)


# =====================================================================
# Core tuple -> geometry, all closed form.
# =====================================================================

def vertex(tup: tuple[int, int, int]) -> np.ndarray:
    """Unit vertex for tuple (f, s1, s2)."""
    f, s1, s2 = tup
    return np.roll([0.0, s1, s2 * PHI], f) / NORM


def primary_tangent(tup: tuple[int, int, int]) -> np.ndarray:
    """Unit tangent at vertex pointing toward same-family neighbor (flip s1)."""
    f, s1, s2 = tup
    return np.roll([0.0, -s1 * PHI, s2], f) / NORM


def tuple_to_id(tup: tuple[int, int, int]) -> int:
    f, s1, s2 = tup
    return 4 * f + 2 * (1 if s1 == 1 else 0) + (1 if s2 == 1 else 0)


def all_tuples() -> list[tuple[int, int, int]]:
    return [(f, s1, s2) for f in range(3) for s1 in (1, -1) for s2 in (1, -1)]


# =====================================================================
# Neighbor pattern. Derived once by hand for the reference pentagon
# (0, +1, +1) via direct geometry; see reports/nearest-dodecahedron-face.md
# for the argument.
#
# CCW order starting from the primary-direction target:
# =====================================================================

_NBR_CCW_REFERENCE: list[tuple[int, int, int]] = [
    (0, -1, +1),   # pos 0: same-family neighbor (primary direction target)
    (+1, +1, +1),  # pos 1
    (+2, +1, +1),  # pos 2
    (+2, -1, +1),  # pos 3
    (+1, +1, -1),  # pos 4
]


def neighbors_ccw(tup: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """5 neighbor tuples in CCW order starting from the primary direction.

    Obtained by applying the symmetry that sends (0, +1, +1) to `tup`:
    cyclic axis permutation by `f`, then the sign reflections needed to
    match (s1, s2). A reflection reverses orientation, so the list is
    reversed (minus position 0) when an odd number of sign-flips applies.
    """
    f0, s10, s20 = tup
    sign_flips = (1 if s10 == -1 else 0) + (1 if s20 == -1 else 0)

    mapped = []
    for (df, ds1, ds2) in _NBR_CCW_REFERENCE:
        f = (f0 + df) % 3
        # Apply the sign-reflection: if we flipped s1 (y-reflection in the
        # reference family), the neighbor's "coefficient-1 sign" flips
        # whenever it sits on the coefficient-1 axis of its family, etc.
        # The cleanest bookkeeping: transform the literal vertex.
        ref_v = np.roll([0.0, ds1, ds2 * PHI], df)
        # apply axis permutation by f0:
        v = np.roll(ref_v, f0)
        # apply sign reflection for s10 and s20 (y- and z- flips in the
        # reference frame's coordinate-1 and coordinate-phi axes, then
        # rotated through the axis permutation):
        # In reference frame: coord-1 axis is y, coord-phi axis is z.
        # After rolling by f0, those axes move.
        flip_axes = np.ones(3)
        if s10 == -1:
            flip_axes[(1 + f0) % 3] *= -1     # coord-1 axis after roll
        if s20 == -1:
            flip_axes[(2 + f0) % 3] *= -1     # coord-phi axis after roll
        v = v * flip_axes
        mapped.append(_vertex_to_tuple(v))

    # A single sign-reflection (odd parity) reverses orientation, so we
    # reverse the cyclic tail (positions 1..4). Position 0 is always
    # primary-direction by construction.
    if sign_flips % 2 == 1:
        mapped = [mapped[0]] + list(reversed(mapped[1:]))
    return mapped


def _vertex_to_tuple(v: np.ndarray) -> tuple[int, int, int]:
    """Inverse of `vertex` up to normalization (works on unnormalized too)."""
    # Which component is zero tells the family.
    for f in range(3):
        vv = np.roll(v, -f)
        if abs(vv[0]) < 1e-9:
            return (f, int(np.sign(vv[1])), int(np.sign(vv[2])))
    raise ValueError(f"not an icosa vertex: {v}")


# =====================================================================
# Pentagon face table, closed form.
# Face k (0..4) is the triangle (pentagon, neighbors[k], neighbors[(k+1)%5]);
# its center = (v + n_k + n_{k+1}) / 3, normalized.
# Digit mapping: first face CCW from the primary direction gets digit 2,
# then sequential CCW cycle 2,3,4,5,6 (d=1 deleted).
# =====================================================================

PENTAGON_DIGIT_CCW = (2, 3, 4, 5, 6)


def pentagon_faces(tup: tuple[int, int, int]) -> list[np.ndarray]:
    """5 incident face centers (unit vectors), CCW from primary direction.

    Face k is between neighbor k and neighbor (k+1)%5.
    """
    nbrs = neighbors_ccw(tup)
    v = vertex(tup)
    nbr_vs = [vertex(n) for n in nbrs]
    centers = []
    for k in range(5):
        c = (v + nbr_vs[k] + nbr_vs[(k + 1) % 5]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return centers


def pentagon_face_digits(tup: tuple[int, int, int]) -> dict[int, np.ndarray]:
    """Return {digit: face-center-unit-vector} for digits 2,3,5,4,6.

    Convention: digit 2 is the first face CCW from the primary direction.
    """
    centers = pentagon_faces(tup)
    return {d: centers[pos] for pos, d in enumerate(PENTAGON_DIGIT_CCW)}


# =====================================================================
# Sanity checks
# =====================================================================

def check_all_edges():
    """Verify every claimed neighbor is at icosahedron-edge distance."""
    target = PHI / (PHI + 2)
    for tup in all_tuples():
        v = vertex(tup)
        for n in neighbors_ccw(tup):
            nv = vertex(n)
            assert abs(v @ nv - target) < 1e-9, f"{tup} -> {n} not an edge"


def check_ccw_ordering():
    """Verify the returned ordering is CCW at angles 0, 72, 144, 216, 288 deg."""
    for tup in all_tuples():
        v = vertex(tup)
        t = primary_tangent(tup)
        e2 = np.cross(v, t)
        angles = []
        for n in neighbors_ccw(tup):
            nv = vertex(n)
            nt = nv - (nv @ v) * v
            angles.append(np.degrees(np.arctan2(nt @ e2, nt @ t)) % 360)
        expected = [0, 72, 144, 216, 288]
        for a, e in zip(angles, expected):
            assert abs((a - e + 180) % 360 - 180) < 1e-6, (
                f"{tup}: got angles {angles}, expected {expected}"
            )


def main():
    check_all_edges()
    check_ccw_ordering()
    print("All sanity checks passed.\n")

    # Show the face table in a readable form.
    print("Pentagon face centers (unit vectors), digit -> center:")
    print("  id  tuple       " + "   ".join(f"d={d}" for d in PENTAGON_DIGIT_CCW))
    for tup in all_tuples():
        digits = pentagon_face_digits(tup)
        centers_str = "  ".join(
            f"({digits[d][0]:+.2f},{digits[d][1]:+.2f},{digits[d][2]:+.2f})"
            for d in PENTAGON_DIGIT_CCW
        )
        print(f"  {tuple_to_id(tup):2d}  {tup}   {centers_str}")


if __name__ == "__main__":
    main()
