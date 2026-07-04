"""Ring-1 invariants — the implementation-agnostic ground truth.

These tests define what a correct ring-1 walk *is*, without reference
to the implementation's internals: ring sizes, validity, symmetry,
self-exclusion, the sphere-distance band, CCW closure, and the
primary-direction phase convention. (If a second implementation ever
exists — e.g. a compiled port — parametrize these over both.)

The distance band deserves emphasis: the historically dangerous
failure — resolving a phantom to the wrong stitch twin — places a
"neighbor" at roughly sqrt(3) times the true neighbor distance, far
outside the max/min < 1.3 band. Any ring passing all invariants here
is the right ring.
"""

import numpy as np
import pytest

from mu3 import (
    cell_center,
    cell_resolution,
    cell_ring1,
    cells_at_res,
    dodec,
    is_pentagon,
    is_valid_cell,
)
from mu3.cell import _eisenstein_center, _project
from mu3.face_lattice import digit_offset, get_rot


def _test_cells():
    for res in [0, 1, 2, 3]:
        yield from cells_at_res(res)


def test_ring1_size():
    for c in _test_cells():
        N = len(set(cell_ring1(c)))

        if is_pentagon(c):
            assert N == 5
        else:
            assert N == 6


def test_all_neighbors_are_valid():
    for c in _test_cells():
        for nb in cell_ring1(c):
            assert is_valid_cell(nb), (c, nb)
            assert cell_resolution(nb) == cell_resolution(c), (c, nb)


def test_ring1_symmetry():
    """c' ∈ ring1(c) ⇒ c ∈ ring1(c'), checked over one precomputed
    map (each ring computed exactly once)."""
    rings = {c: cell_ring1(c) for c in _test_cells()}
    for c, ring in rings.items():
        for nb in ring:
            assert c in rings[nb], (c, nb)


def test_ring1_excludes_self():
    for c in _test_cells():
        assert c not in cell_ring1(c), c


def _check_neighbor_lengths(cell):
    c = cell_center(cell)
    lens = [
        np.linalg.norm(c - cell_center(nb))
        for nb in cell_ring1(cell)
    ]

    assert min(lens) > 0
    assert max(lens)/min(lens) < 1.3


def test_ring1_sphere_distance():
    for c in _test_cells():
        _check_neighbor_lengths(c)


def _tangent_toward(cell, p3d):
    """Unit tangent at ``cell``'s center pointing toward ``p3d``."""
    c = cell_center(cell)
    t = p3d - (p3d @ c) * c
    return t / np.linalg.norm(t)


def _check_neighbors_ccw(cell):
    """Neighbors should be ordered CCW around the source cell on the
    sphere AND span exactly one full revolution.

    Project each neighbor's 3D center onto the source's tangent plane
    and unit-normalize. Each consecutive pair (a, b) -- including the
    wrap-around -- contributes a signed angle ``atan2((a × b)·c, a·b)``.
    Each angle must be positive (neighbors rotate CCW); the sum must
    equal 2π (the loop closes around the source exactly once).
    """
    c = cell_center(cell)
    proj = [_tangent_toward(cell, cell_center(nb)) for nb in cell_ring1(cell)]

    angles = []
    n = len(proj)
    for i in range(n):
        a = proj[i]
        b = proj[(i + 1) % n]

        angles.append(
            np.arctan2(
                np.cross(a, b) @ c,
                a @ b,
            )
        )

    assert all(a > 0 for a in angles)
    assert abs(sum(angles) - 2 * np.pi) < 1e-9
    assert max(angles)/min(angles) < 1.5


def test_ring1_ccw_order():
    for c in _test_cells():
        _check_neighbors_ccw(c)


def test_ring1_ends_at_primary_direction():
    """The last neighbor in the ring is the primary-direction neighbor.

    Implementation-agnostic phase check: the expected direction is a
    half-step along D=6 in the flat frame, rendered to the sphere (the
    half-step stays inside the cell, so it stitches consistently with
    the center). The ring's last cell must be the one angularly
    closest to that direction — neighbors are ~60 degrees apart, so
    "closest" is unambiguous.

    Pentagon-adjacent hexes whose ring collapsed to 5 are skipped: the
    collapsed direction may be D=6 itself, in which case "last" is
    legitimately a different neighbor.
    """
    for c in _test_cells():
        ring = cell_ring1(c)
        last = ring[-1]
        res = cell_resolution(c)
        if res == 0:
            primary = (int(dodec.neighbors[c[0]][0]),)
            assert last == primary, (c, last, primary)
            continue
        if not is_pentagon(c) and len(ring) == 5:
            continue
        z_half = _eisenstein_center(c[1:]) \
            + digit_offset[6] / (2 * get_rot(res))
        t6 = _tangent_toward(c, _project(z_half, c[0]))
        best = max(ring, key=lambda nb:
                   _tangent_toward(c, cell_center(nb)) @ t6)
        assert last == best, (c, last, best)


# --- named killer regressions ------------------------------------------
#
# Members come from the adversarial-zoo registry (tests/adversarial.py,
# where each family's full story lives). Each pins a specific wrong
# seam-disambiguation rule that a plausible "simplification" of
# neighbor._resolve would reintroduce (see the module
# docstring there).


def test_phantom_corner_family():
    """3-hex phantom corners — the family that killed the old
    NEIGHBOR_TRANS walk. Subsumed by the full sweeps above; kept named
    for its history."""
    from adversarial import phantom_corner_cells
    for cell in phantom_corner_cells():
        _check_neighbor_lengths(cell)
        _check_neighbors_ccw(cell)


def test_seam_crossing_cw():
    """CW seam-crossing ring-1 adjacencies. Defeats the endpoint-side
    rule (deciding the stitch from where the walk *landed* instead of
    where it came from)."""
    from adversarial import seam_crossing_pairs
    for src, nb in seam_crossing_pairs():
        assert nb in cell_ring1(src), (src, nb)


def test_post_hop_wedge_ccw():
    """Phantoms reached after a cross-pentagon hop. Defeats deciding
    the stitch from the source in its *original* frame — the arrow
    must be carried through the hop."""
    from adversarial import post_hop_pairs
    for src, nb in post_hop_pairs():
        assert nb in cell_ring1(src), (src, nb)


def test_cell_ring1_validation():
    """Invalid cells raise instead of resolving a phantom position to
    a plausible-looking (and wrong) ring — matching step()'s contract."""
    for bad in [(0, 1), (12,), (0, 7), ()]:
        with pytest.raises(ValueError):
            cell_ring1(bad)


def test_step_equals_position_walk_exhaustive():
    """The carry fast tier must equal the pure position walk on EVERY
    cell x direction, res 0-3.

    Honest scope: seam/phantom rows are vacuous (step's fallback IS
    _position_step), so the teeth are (a) every fast-path result
    matches the walk, and (b) no case is wrongly ROUTED to the fast
    path. Mis-routing the other way (fast-eligible sent to fallback)
    is a perf bug, not a correctness bug — caught by the routing
    floor: at res 3 the carry is expected to absorb ~90% of steps
    (measured 90.09%; scripts/verify_carry_walk.py extends this gate
    to res 0-6, 8.2M steps)."""
    from mu3.eisenstein import carry_digits, first_nonzero_digit
    from mu3.neighbor import _position_step, step

    fast = total = 0
    for res in range(4):
        for c in cells_at_res(res):
            for d in (1, 2, 3, 4, 5, 6):
                assert step(c, d) == _position_step(c, res, d), (c, d)
                if res == 3:
                    total += 1
                    cd = carry_digits(c[1:], d)
                    if cd is not None and first_nonzero_digit(cd) != 1:
                        fast += 1
    assert fast / total > 0.85, f'fast-path routing eroded: {fast/total:.1%}'
