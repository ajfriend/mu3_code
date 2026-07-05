"""Grid traversal: rings, disks, distances, and pair queries
(``grid_distance`` / ``grid_path``).

Two tiers:

- **Tier 1 — flat fast path.** When the disk stays clear of every
  nearby CUT RAY (each of the six nearby pentagons carries one
  deleted-wedge cut; see :func:`_cut_clearance`), the chart is
  injective on the disk and grid distance equals the closed-form hex
  distance of the developed offset. The disk is then a direct
  enumeration: every lattice delta with ``hex_dist <= k``, each
  resolved to its canonical cell by the exact holonomy walk. ~3x
  faster than BFS in Python; the guard admits ~a third of res-3
  queries at k <= 10 and almost nothing at coarse res (which is all
  seams — BFS territory by design).
- **Tier 3 — BFS reference.** Breadth-first expansion on exact ring-1
  walks: always correct, O(k^2) cells explored. The fallback whenever
  the tier-1 guard declines, and the oracle the fast path is verified
  against in tests.

(An exact fast tier for seam-overlapping DISKS was attempted three
times and refuted — near the seam, position-to-cell needs path
context, and disks must pay for it per cell; those disks stay on BFS.
The machinery's economics were redirected to PAIR queries, shipped in
the section at the bottom of this module. Full attempt logs in the
punch list.)

FLOAT QUARANTINE INVARIANT: every cell, distance, and path this
module returns is produced by exact integer arithmetic. Float
geometry appears ONLY in dispatch guards (``_cut_clearance``,
``_min_detour``, the pair corridor cap), which choose WHICH exact
method runs and are conservative: a wrong guard decision falls back
to BFS or fails exact walk-verification — it can never change an
answer. Keep it that way; the branch's refuted designs all failed by
letting geometric proxies leak into answer production.

Output ordering is deliberately UNSPECIFIED: ``grid_distances``
returns a dict, and ``ring_k``/``disk_k`` order follows the active
tier's traversal. Callers needing the oriented ring-1 convention (CCW,
primary-direction last) should use ``cell_ring1``; no consumer has yet
needed an oriented k-ring, so no spiral contract is promised.

Whole-globe semantics: rings and disks are METRIC LEVEL SETS of the
grid distance, which saturates at the source's eccentricity (~51 at
res 3). ``disk_k`` therefore grows to the whole sphere and is constant
beyond; ``ring_k`` is empty beyond — the same reason no point of a
round sphere is at geodesic distance > pi*R. The other intuition — a
wavefront that keeps propagating "through" the antipode and comes back
around — is a real object but a different one (the exponential map /
developed-geodesic front, whose members are NOT at distance k); it is
a path-family object, cousin to the pair queries' walked lines, not a
ring. Keeping rings as level sets preserves the partition (``disk_k``
is the disjoint union of rings 0..k) and source/target symmetry.
"""

from typing import Sequence

from . import dodec
from .cell import cell_resolution, is_valid_cell
from .cross_pentagon import cut_rays, tau_between
from .eisenstein import (
    Eis,
    get_rot_eis,
    hex_dist,
    line_digits,
    scaled_center,
)
from .face_lattice import rotate_digit_ccw
from .neighbor import _ExactWitness, _resolve, cell_ring1, step
from .p6 import rotation_about

# The fast path requires the disk to clear each cut RAY by this many
# lattice units beyond k: 1 covers cell extent at the ball boundary,
# +1 covers the Gosper wiggle of the glued boundary around the
# straight ray (interloper strings live within ~a cell of it).
# Validated by the exhaustive tier-1-vs-BFS sweep in tests.
_CUT_MARGIN = 2.0


def _fast_path_ok(base: int, z_c: Eis, res: int, k: int) -> bool:
    """The tier-dispatch predicate: flat fast path iff the whole disk
    clears every nearby cut ray by the margin. One name shared by
    ``grid_distances`` and the tests' population sweep, so the guard
    and what the oracle tests verify cannot drift apart."""
    return k > 0 and _cut_clearance(base, z_c, res) > k + _CUT_MARGIN


def _cut_clearance(base: int, z_c: Eis, res: int) -> float:
    """Euclidean distance (lattice units) from a scaled cell center to
    the nearest cut ray (exact ray data from
    ``cross_pentagon.cut_rays``, next to the branch-cut spec it
    encodes).

    Why rays and not just cone points: the deleted-wedge identification
    glues along the whole ray, so cells straight across it sit 60
    degrees apart in chart coordinates at ANY radius. A disk that
    straddles a cut needs the multi-sheet (tier-2) treatment — its
    straight developments land on wedge positions whose extraction can
    be a canonical Gosper-interloper string, which the witness rule
    never sees (unlike ring-1 steps, which always produce phantom-FORM
    strings at the seam). Float geometry is fine here: the guard only
    picks the tier, conservatively; correctness never depends on it.
    """
    zc = z_c.to_complex()
    dists = []
    for _owner, apex, direction in cut_rays(base, res):
        w = zc - apex.to_complex()
        p = w * direction.to_complex().conjugate()   # |direction| == 1
        # Real part: position along the ray; imag: perpendicular
        # offset. Behind the apex, the apex itself is nearest.
        dists.append(abs(w) if p.real <= 0.0 else abs(p.imag))
    return min(dists)


def _flat_distances(
    base: int, z_c: Eis, res: int, k: int
) -> dict[tuple, int]:
    """Tier-1 disk: enumerate every lattice delta with hex distance
    <= k, resolve each developed position exactly, distance = the
    closed-form hex distance. Under the cut-clearance guard the chart
    is injective on the disk — each cell appears exactly once
    (asserted)."""
    out: dict[tuple, int] = {}
    for a in range(-k, k + 1):
        for b in range(-k, k + 1):
            delta = Eis(a, b)
            d = hex_dist(delta)
            if d > k:
                continue
            nb, _ = _resolve(base, z_c + delta, _ExactWitness(z_c), res)
            assert nb not in out, nb
            out[nb] = d
    return out


def _bfs_distances(cell_t: tuple, k: int) -> dict[tuple, int]:
    """Tier-3 disk: BFS on ring-1. Always correct; the oracle."""
    out: dict[tuple, int] = {cell_t: 0}
    frontier = [cell_t]
    for dist in range(1, k + 1):
        nxt = []
        for c in frontier:
            for nb in cell_ring1(c):
                if nb not in out:
                    out[nb] = dist
                    nxt.append(nb)
        frontier = nxt
    return out


def grid_distances(cell: Sequence[int], k: int) -> dict[tuple, int]:
    """Every cell within grid distance ``k`` of ``cell``, mapped to its
    exact distance (0 for ``cell`` itself). Dispatches to the flat
    fast path when the conservative cut-clearance guard allows, else
    BFS."""
    cell_t = tuple(int(x) for x in cell)
    if not is_valid_cell(cell_t):
        raise ValueError(f'grid_distances: invalid cell {cell_t}')
    if k < 0:
        raise ValueError(f'grid_distances: k must be >= 0, got {k}')
    res = cell_resolution(cell_t)
    z_c = scaled_center(cell_t[1:])
    if _fast_path_ok(cell_t[0], z_c, res, k):
        return _flat_distances(cell_t[0], z_c, res, k)
    return _bfs_distances(cell_t, k)


def disk_k(cell: Sequence[int], k: int) -> list[tuple]:
    """All cells within grid distance ``k`` (including ``cell``).
    Saturates: for ``k`` >= the source's eccentricity this is the
    whole sphere, constant thereafter."""
    return list(grid_distances(cell, k))


def ring_k(cell: Sequence[int], k: int) -> list[tuple]:
    """All cells at grid distance exactly ``k`` (``[cell]`` for
    ``k == 0``; EMPTY for ``k`` beyond the source's eccentricity —
    rings are level sets of a distance that saturates, see module
    docstring). ``while ring_k(c, k): ...`` therefore terminates
    naturally at whole-globe coverage."""
    return [c for c, d in grid_distances(cell, k).items() if d == k]


# --- pair queries: grid_distance / grid_path ---------------------------
#
# Economics inverted from disks (see the tier-2 attempt log in the
# punch list): for a PAIR we can afford to VERIFY a candidate route by
# walking it with the carried arrow — O(route length) exact ring-1
# steps, which grid_path must spend anyway to emit the path. A walked
# candidate is a real connected cell path by construction, so its
# length is a true upper bound; taking the shortest VERIFIED candidate
# from the homotopy classes of a single-seam corridor (direct, the two
# windings about the cone, and through the pentagon cell) is exact —
# validated by exhaustive pair sweeps against BFS.

# Corridor guard margin: how close (Euclidean, lattice units) a cut
# ray may come to the straight corridor before its seam's homotopy
# classes are added as candidates. Covers the walked line's deviation
# from the straight segment (~1), cell extent (~1), and the Gosper
# wiggle (~2). Conservative: too-large only costs fast-path coverage.
_PAIR_MARGIN = 4.0


def _rep_in_frame(base: int, cell: tuple, res: int) -> Eis | None:
    """``cell``'s canonical center in ``base``'s scaled frame: identity
    for same-base, one inverse-TAU transform for neighbor-base, None
    otherwise.

    Deliberately NOT extended to two hops: representations through two
    chart transitions are chain-dependent — the k=0,2,4 corner
    products are stitch conjugates, not the identity, so chains
    through different intermediates differ by cut crossings, i.e. by
    homotopy class. Handling that correctly means one candidate per
    chain class, which is the multi-seam problem; farther pairs take
    the BFS fallback instead (tried and refuted by sweep, 2026-07-03).
    """
    z = scaled_center(cell[1:])
    if cell[0] == base:
        return z
    if cell[0] not in dodec.neighbors[base]:
        return None
    gi = tau_between(base, cell[0]).inverse()
    return gi.apply_scaled(z, get_rot_eis(res))


def _min_detour(p: complex, q: complex, o: complex, d: complex) -> float:
    """min over points ``x`` on the ray (``o``, unit ``d``) of
    ``|p - x| + |x - q|`` — the shortest Euclidean route from ``p`` to
    ``q`` that touches the ray. Convex in the ray parameter; ternary
    search is plenty for guard geometry."""
    def f(t: float) -> float:
        x = o + t * d
        return abs(p - x) + abs(q - x)

    lo = 0.0
    hi = max(0.0,
             ((p - o) * d.conjugate()).real,
             ((q - o) * d.conjugate()).real) + abs(p - q) + 1.0
    for _ in range(36):   # (2/3)^36 of the range: far below the margin
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
    return f((lo + hi) / 2)


def _walk_line(cell: tuple, delta: Eis, rot0: int = 0) -> list[tuple]:
    """Walk the hex line for source-frame offset ``delta`` step by
    step with the carried arrow (``neighbor.step``), starting from
    ``cell`` whose frame is rotated ``rot0`` from the frame ``delta``
    is expressed in. Returns the full cell path — real and connected
    by construction; where it ENDS is the ground truth the candidate
    machinery verifies against."""
    path = [cell]
    rot = rot0
    for d in line_digits(delta):
        cell, r = step(cell, rotate_digit_ccw(d, rot))
        rot = (rot + r) % 6
        path.append(cell)
    return path


def _pair_fast(a: tuple, b: tuple, res: int):
    """Fast-path pair resolution: ``(distance, path_builder)`` or None.

    Scope (v1): ``b`` representable in ``a``'s frame (same or neighbor
    base) and at most ONE cut ray near the straight corridor. The
    candidate set for a single-seam corridor is the cone's homotopy
    classes: direct, the two windings (rotations by the deficit about
    the apex), and through the pentagon cell itself. Candidates are
    tried in ascending length; the first whose WALKED line actually
    ends at ``b`` wins. A clean corridor walks its single direct
    candidate — EVERY distance this function returns is backed by a
    verified walked path, so a wrong guard decision falls back to
    BFS rather than changing an answer (the module's quarantine
    invariant).
    """
    base = a[0]
    z_a = scaled_center(a[1:])
    rep = _rep_in_frame(base, b, res)
    if rep is None:
        return None

    # Corridor-radius guard: only the six listed seams (own base + 5
    # neighbors, via cut_rays) are known in this frame; the nearest
    # UNLISTED seam (a 2-hop pentagon's, its cone at ~1.7 GN with the
    # seam segment extending ~0.58 GN back toward us) approaches no
    # closer than ~1.1 GN to the origin. The corridor's max radius is
    # attained at an endpoint (convexity), and z_a is always within
    # its own territory, so capping |rep| keeps every relevant seam
    # in the checked list. Farther pairs go to BFS.
    gn_abs = abs(get_rot_eis(res).to_complex())
    if abs(rep.to_complex()) > 1.05 * gn_abs - _PAIR_MARGIN / 2:
        return None

    # Seam involvement is a DETOUR test, not corridor proximity: a
    # shortcut seam only needs to sit inside the lens of near-optimal
    # routes, which fattens with distance. A route via seam point x
    # has hex length >= (sqrt(3)/2) * (|a - x| + |x - b|), so a seam
    # can only matter if its minimal Euclidean detour clears that bar.
    direct = rep - z_a
    bar = (hex_dist(direct) + _PAIR_MARGIN) * 2.0 / (3.0 ** 0.5)
    rays = cut_rays(base, res)
    involved = [
        i for i, (_owner, apex, direction) in enumerate(rays)
        if _min_detour(z_a.to_complex(), rep.to_complex(),
                       apex.to_complex(), direction.to_complex()) <= bar
    ]
    if not involved:
        path = _walk_line(a, direct)
        if path[-1] != b:
            return None   # guard miss: the corridor was not clean
        return len(path) - 1, lambda: path
    if len(involved) > 1:
        return None

    # The one involved seam: its owning pentagon travels with the ray.
    owner, apex, _dir = rays[involved[0]]
    pent = (owner,) + (0,) * res
    leg2rot = 0 if owner == base else tau_between(base, owner).u

    def via_apex():
        leg1 = _walk_line(a, apex - z_a)
        if leg1[-1] != pent:
            return None
        leg2 = _walk_line(pent, rep - apex, rot0=leg2rot)
        return leg1 + leg2[1:]

    cands = [(hex_dist(direct), lambda: _walk_line(a, direct))]
    for w in (1, 5):
        # Winding candidate: the rep rotated about the cone apex.
        d2 = rotation_about(w, apex).apply(rep) - z_a
        cands.append((hex_dist(d2), lambda d2=d2: _walk_line(a, d2)))
    cands.append((hex_dist(apex - z_a) + hex_dist(rep - apex), via_apex))

    cands.sort(key=lambda c: c[0])
    for length, build in cands:
        path = build()
        if path is not None and path[-1] == b:
            return length, lambda: path
    return None


def _bfs_pair(a: tuple, b: tuple) -> list[tuple]:
    """Exact fallback: BFS from ``a`` until ``b`` is reached, with
    parent recovery. Always terminates (the grid is connected)."""
    parent: dict[tuple, tuple | None] = {a: None}
    frontier = [a]
    while b not in parent:
        nxt = []
        for c in frontier:
            for nb in cell_ring1(c):
                if nb not in parent:
                    parent[nb] = c
                    nxt.append(nb)
        frontier = nxt
    path = [b]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _pair_check(a: Sequence[int], b: Sequence[int]) -> tuple[tuple, tuple]:
    a_t = tuple(int(x) for x in a)
    b_t = tuple(int(x) for x in b)
    for c in (a_t, b_t):
        if not is_valid_cell(c):
            raise ValueError(f'pair query: invalid cell {c}')
    if cell_resolution(a_t) != cell_resolution(b_t):
        raise ValueError(
            f'pair query: cells must share a resolution: {a_t} vs {b_t}')
    return a_t, b_t


def grid_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Exact grid distance between two same-resolution cells.

    Single-frame pairs with at most one seam near the corridor resolve
    by walk-verified candidate routes (O(distance)); everything else
    falls back to pair-BFS.
    Need the path as well? Call :func:`grid_path` once instead — its
    length minus 1 is the distance (avoids paying the fallback twice).
    """
    a_t, b_t = _pair_check(a, b)
    if a_t == b_t:
        return 0
    fast = _pair_fast(a_t, b_t, cell_resolution(a_t))
    if fast is not None:
        return fast[0]
    return len(_bfs_pair(a_t, b_t)) - 1


def grid_path(a: Sequence[int], b: Sequence[int]) -> list[tuple]:
    """A shortest connected cell path from ``a`` to ``b``, inclusive
    (length == grid_distance + 1; shortest paths are not unique — this
    returns one). Pentagon-crossing pairs work: candidate routes are
    walked with the carried arrow, so seams and the pentagons' 5-way
    junctions are handled by the same machinery as ring-1."""
    a_t, b_t = _pair_check(a, b)
    if a_t == b_t:
        return [a_t]
    fast = _pair_fast(a_t, b_t, cell_resolution(a_t))
    if fast is not None:
        return fast[1]()
    return _bfs_pair(a_t, b_t)
