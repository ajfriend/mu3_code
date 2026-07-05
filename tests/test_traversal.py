"""Tier-3 BFS traversal: self-consistency and count laws.

This is the reference implementation, so the tests are the
implementation-agnostic laws any traversal must satisfy — when the
faster tiers land they run against these AND against this BFS as
oracle.
"""

import pytest

from mu3 import cell_ring1, cells_at_res, is_pentagon
from mu3.eisenstein import scaled_center
from mu3.traversal import (
    _bfs_distances,
    _bfs_pair,
    _fast_path_ok,
    _flat_distances,
    _pair_fast,
    disk_k,
    grid_distance,
    grid_distances,
    grid_path,
    ring_k,
)


def test_k0():
    for c in [(0,), (3, 2), (7, 4, 1)]:
        assert disk_k(c, 0) == [c]
        assert ring_k(c, 0) == [c]


def test_k1_matches_ring1():
    for res in [0, 1, 2]:
        for c in cells_at_res(res):
            assert set(ring_k(c, 1)) == set(cell_ring1(c)), c
            assert set(disk_k(c, 1)) == {c, *cell_ring1(c)}, c


def test_distances_self_consistent():
    """d(a, b) as seen from a equals d(b, a) as seen from b, for all
    pairs within range — grid distance is a metric restriction."""
    K = 2
    for res in [0, 1, 2]:
        dists = {c: grid_distances(c, K) for c in cells_at_res(res)}
        for a, da in dists.items():
            for b, d_ab in da.items():
                assert dists[b][a] == d_ab, (a, b)


def test_disk_monotone_and_ring_partition():
    """disk_k grows with k, and its rings partition it."""
    for c in [(0,), (2, 3), (5, 6, 1), (11, 2, 6, 6)]:
        prev: set = set()
        for k in range(4):
            dk = set(disk_k(c, k))
            assert prev <= dk
            assert dk - prev == set(ring_k(c, k)), (c, k)
            prev = dk


# --- tier-1 fast path vs the BFS oracle --------------------------------


def _applicable(res, k):
    """(cell, scaled center) pairs the dispatch guard admits — via the
    same _fast_path_ok the dispatcher uses, so the tested population
    cannot drift from the shipped one."""
    for c in cells_at_res(res):
        z_c = scaled_center(c[1:])
        if _fast_path_ok(c[0], z_c, res, k):
            yield c, z_c


def test_fast_path_matches_bfs_res2_exhaustive():
    """Every (cell, k) pair the guard admits at res 2."""
    for k in range(1, 7):
        for c, z_c in _applicable(2, k):
            assert _flat_distances(c[0], z_c, 2, k) \
                == _bfs_distances(c, k), (c, k)


@pytest.mark.parametrize('k', [2, 5, 8])
def test_fast_path_matches_bfs_res3_sampled(k):
    """Sampled guard-admitted cells at res 3 (the full sweep — 10,512
    pairs at k <= 10 — ran clean during design, 2026-07-03)."""
    pairs = list(_applicable(3, k))
    assert pairs, k   # the guard admits a real population
    for c, z_c in pairs[::17]:
        assert _flat_distances(c[0], z_c, 3, k) \
            == _bfs_distances(c, k), (c, k)


def test_dispatch_transparent():
    """Public grid_distances agrees with the BFS reference regardless
    of which tier the guard picks (fast-path cell, near-cut cell,
    pentagon center)."""
    for c in [(4, 3, 3, 3), (0, 2, 1, 5), (7, 0, 0, 0)]:
        for k in (2, 4):
            assert grid_distances(c, k) == _bfs_distances(c, k), (c, k)


def test_saturation_semantics():
    """Rings and disks are metric level sets: past the source's
    eccentricity the ring is EMPTY and the disk is the whole sphere,
    constant. (The 'wavefront wrapping through the antipode' is the
    exponential map, a different object — see module docstring.)"""
    RES = 1
    n_cells = sum(1 for _ in cells_at_res(RES))
    for src in [(0, 0), (5, 3), (11, 6)]:
        # Eccentricity via the natural termination pattern.
        ecc = 0
        while ring_k(src, ecc + 1):
            ecc += 1
        full = grid_distances(src, ecc)
        assert len(full) == n_cells                      # whole sphere
        assert max(full.values()) == ecc
        assert ring_k(src, ecc + 1) == []                # empty beyond
        assert ring_k(src, ecc + 7) == []
        assert set(disk_k(src, ecc + 1)) == set(full)    # constant beyond
        # Rings partition the disk all the way out.
        assert sum(len(ring_k(src, k)) for k in range(ecc + 1)) == n_cells


@pytest.mark.parametrize('b', range(12))
def test_pentagon_ring_counts(b):
    """Rings around a pentagon have 5k cells (five wedges of k), while
    hex interiors have 6k — the deleted subtree in one number."""
    pent = (b, 0, 0, 0)
    for k in range(1, 5):
        assert len(ring_k(pent, k)) == 5 * k, (b, k)


def test_interior_hex_ring_counts():
    """Cells farther than k from every pentagon have the pure-lattice
    counts: |ring_k| = 6k, |disk_k| = 1 + 3k(k+1). "Within k of a
    pentagon" is the union of the pentagons' own k-disks (distance
    symmetry, itself tested above)."""
    RES = 3
    K = 3
    near_pent: set = set()
    for b in range(12):
        near_pent.update(disk_k((b,) + (0,) * RES, K))

    eligible = 0
    for c in cells_at_res(RES):
        if c in near_pent:
            continue
        eligible += 1
        if eligible % 37:   # sample ~1/37 of eligible cells for speed
            continue
        assert len(ring_k(c, K)) == 6 * K, c
        assert len(disk_k(c, K)) == 1 + 3 * K * (K + 1), c
    assert eligible > 2000   # the eligible population is real


# --- pair queries: grid_distance / grid_path ----------------------------


def test_pair_trivial_and_validation():
    c = (4, 3, 2)
    assert grid_distance(c, c) == 0
    assert grid_path(c, c) == [c]
    for nb in cell_ring1(c):
        assert grid_distance(c, nb) == 1
        assert grid_path(c, nb) == [c, nb]
    with pytest.raises(ValueError):
        grid_distance((0, 2), (0, 2, 2))       # mixed resolutions
    with pytest.raises(ValueError):
        grid_distance((0, 1), (0, 2))          # invalid cell


def test_pair_fast_claims_match_bfs_truth():
    """Every fast-path distance claim vs full-BFS ground truth, for a
    spread of res-2 sources against ALL targets. (The BFS fallback IS
    the truth algorithm, so only fast claims need independent checks;
    the full exhaustive sweep — res 1-2 all pairs, res 3 sampled,
    0 errors — ran during design, 2026-07-03.)"""
    cells = list(cells_at_res(2))
    checked = 0
    for src in cells[::17]:
        truth = _bfs_distances(src, 30)   # res-2 eccentricity < 30:
        assert len(truth) == len(cells)   # saturated = whole sphere
        for tgt in cells:
            if tgt == src:
                continue
            fast = _pair_fast(src, tgt, 2)
            if fast is not None:
                checked += 1
                assert fast[0] == truth[tgt], (src, tgt)
    assert checked > 1000   # the fast path covers a real population


def test_pair_path_laws_near_pentagons():
    """Path laws on the adversarial population: sources adjacent to a
    pentagon, all targets within distance 4 (pentagon-crossing pairs
    included — the capability H3's gridPathCells documents away)."""
    for b in range(12):
        src = (b, 0, 0, 2)
        truth = _bfs_distances(src, 4)
        for tgt, dt in truth.items():
            assert grid_distance(src, tgt) == dt, (src, tgt)
        for tgt in list(truth)[::7]:
            p = grid_path(src, tgt)
            assert p[0] == src and p[-1] == tgt
            assert len(p) == truth[tgt] + 1
            assert all(p[i + 1] in cell_ring1(p[i])
                       for i in range(len(p) - 1)), (src, tgt, p)


def test_pair_symmetry():
    """d(a,b) == d(b,a) across the fast/BFS dispatch boundary (disk
    neighborhoods keep the BFS legs cheap)."""
    for src in [(0, 2, 6), (5, 0, 3), (9, 4, 0), (11, 6, 6)]:
        for tgt in disk_k(src, 5)[::3]:
            assert grid_distance(src, tgt) == grid_distance(tgt, src), \
                (src, tgt)


# --- the adversarial zoo (tests/adversarial.py) --------------------------
#
# Historical failure instances pinned through the PUBLIC API, so they
# hold however the dispatch guards evolve. Each family's story lives
# in the registry module.


def test_zoo_interloper_disks():
    """The (cell, k) disk queries that refuted tier-1's cone-based
    guard, at their historical failing k values."""
    from adversarial import interloper_disk_cases
    for cell, k in interloper_disk_cases():
        assert grid_distances(cell, k) == _bfs_distances(cell, k), (cell, k)


def test_zoo_pairs():
    """The pair queries that refuted the segment-proximity involvement
    test (lens pairs) and exposed second-seam reach (two-seam pairs)."""
    from adversarial import LENS_PAIRS, TWO_SEAM_PAIRS
    for a, b in TWO_SEAM_PAIRS + LENS_PAIRS:
        truth = len(_bfs_pair(a, b)) - 1
        assert grid_distance(a, b) == truth, (a, b)
        p = grid_path(a, b)
        assert p[0] == a and p[-1] == b and len(p) == truth + 1
        assert all(p[i + 1] in cell_ring1(p[i]) for i in range(len(p) - 1))
