"""cells_to_multipolygon, judged by implementation-agnostic oracles:

- AREA ADDITIVITY: the signed excess of a polygon's rings (outer plus
  CW holes) equals the sum of its cells' areas — spherical excess is
  additive, so any wrong ring, wrong orientation, missed hole, or
  duplicated corner shows up as an area mismatch.
- THE ISLAND ORACLE: for a parent cell's descendant set, the boundary
  edge set must equal ``island_edges`` (the independently derived
  Gosper island boundary).
- Ring geometry: a single cell's ring is its ``cell_boundary`` (up to
  cyclic rotation).
"""

import math
import random

import numpy as np
import pytest

from mu3 import cells_at_res
from mu3.cell import _signed_spherical_excess, cell_area, cell_boundary
from mu3.edge import outgoing_directions
from mu3.island import island_edges
from mu3.neighbor import step
from mu3.polygon import cells_to_multipolygon
from mu3.traversal import disk_k
from conftest import random_valid_cells


def _polygon_area(polygon):
    """Sum of the rings' signed excesses, normalized: a polygon whose
    outer encloses more than a hemisphere sums negative (each signed
    excess is the mod-4π representative in (-2π, 2π]) and needs the
    +4π, exactly like a single ring."""
    total = sum(_signed_spherical_excess(ring) for ring in polygon)
    return total if total >= 0.0 else total + 4.0 * math.pi


def _assert_area_matches(polygons, cells):
    total = sum(_polygon_area(p) for p in polygons)
    expect = sum(cell_area(c) for c in cells)
    assert math.isclose(total, expect, rel_tol=1e-9), (total, expect)


def _cyclic_equal(A, B, atol=1e-12):
    if len(A) != len(B):
        return False
    return any(
        np.allclose(np.roll(A, shift, axis=0), B, rtol=0, atol=atol)
        for shift in range(len(A))
    )


@pytest.mark.parametrize('cell', [(3,), (0, 0), (7, 4, 2), (11, 0, 0, 5)])
def test_single_cell_ring_is_its_boundary(cell):
    polygons = cells_to_multipolygon([cell])
    assert len(polygons) == 1 and len(polygons[0]) == 1
    assert _cyclic_equal(polygons[0][0], cell_boundary(cell, closed=False))


@pytest.mark.parametrize('center,k', [
    ((4, 3, 2), 2),      # hex-centered disk
    ((0, 0, 0), 2),      # pentagon-centered disk (cut corners on rings)
    ((9, 0), 1),         # res-1 pentagon disk
])
def test_disk_one_polygon_no_holes(center, k):
    cells = disk_k(center, k)
    polygons = cells_to_multipolygon(cells)
    assert len(polygons) == 1
    assert len(polygons[0]) == 1
    _assert_area_matches(polygons, cells)


@pytest.mark.parametrize('center', [(4, 3, 2), (0, 0, 0)])
def test_punctured_disk_has_hole(center):
    cells = [c for c in disk_k(center, 2) if c != center]
    polygons = cells_to_multipolygon(cells)
    assert len(polygons) == 1
    outer, *holes = polygons[0]
    assert len(holes) == 1
    # the hole is the removed cell's boundary, traversed CW
    assert _cyclic_equal(holes[0][::-1], cell_boundary(center, closed=False))
    _assert_area_matches(polygons, cells)


def test_two_components():
    a = disk_k((2, 5, 3), 1)
    b = disk_k((8, 2, 6), 1)
    assert not set(a) & set(b)
    polygons = cells_to_multipolygon(a + b)
    assert len(polygons) == 2
    _assert_area_matches(polygons, a + b)


@pytest.mark.parametrize('parent', [(5, 2), (0, 0)])
def test_descendant_set_matches_island_oracle(parent):
    """Boundary edges of a parent's descendant set == island_edges,
    and the multipolygon is one hole-free polygon with the right
    area."""
    res = len(parent) + 1
    cells = [c for c in cells_at_res(res) if c[:len(parent)] == parent]
    cell_set = set(cells)
    ours = {(c, d) for c in cell_set for d in outgoing_directions(c)
            if step(c, d)[0] not in cell_set}
    assert ours == set(island_edges(parent, res))

    polygons = cells_to_multipolygon(cells)
    assert len(polygons) == 1 and len(polygons[0]) == 1
    _assert_area_matches(polygons, cells)


def test_random_blobs_area_additivity():
    """Random unions of disks at res 3: whatever the shape (holes,
    multiple components, pentagon contact), ring areas must add up."""
    rng = random.Random(7)
    for _ in range(6):
        cells = set()
        for seed in random_valid_cells(rng, 3, 3):
            cells.update(disk_k(seed, rng.randrange(1, 3)))
        interior = [c for c in cells
                    if all(step(c, d)[0] in cells
                           for d in outgoing_directions(c))]
        if interior:
            cells.discard(interior[0])   # poke a hole
        polygons = cells_to_multipolygon(cells)
        _assert_area_matches(polygons, cells)


def test_validation():
    assert cells_to_multipolygon([]) == []
    with pytest.raises(ValueError):
        cells_to_multipolygon([(0, 2), (1,)])          # mixed res
    with pytest.raises(ValueError):
        cells_to_multipolygon([(0, 1)])                # invalid cell


def test_whole_sphere_no_boundary():
    assert cells_to_multipolygon(list(cells_at_res(1))) == []


def test_sphere_minus_two_disks():
    """A component with NO positively-oriented ring: the sphere minus
    two separated disks. Both rings run CW around their removed disk
    (interior-left), so both signed excesses are negative — outer
    selection must fall back to smallest enclosed area, not sign.
    (Regression: the sign-based classification asserted here.)"""
    removed = set(disk_k((0, 0, 0), 1)) | set(disk_k((3, 4, 2), 1))
    cells = [c for c in cells_at_res(2) if c not in removed]
    polygons = cells_to_multipolygon(cells)
    assert len(polygons) == 1
    assert len(polygons[0]) == 2
    _assert_area_matches(polygons, cells)
