# mu3 working notes

## Workflow

- **Tests**: `just test` (full suite, ~35s, 663 tests) or
  `just test-one path` for one file or node id. Both reinstall the
  editable package first via `uv sync --reinstall-package mu3`.
  Don't use `uv run --with-editable .` — it bypasses the reinstall.
- **Scripts**: `uv run scripts/foo.py`. Scripts use PEP 723 inline
  metadata for any extra deps; never `--with` flags.
- **Plan files**: persisted in `~/.claude/plans/`; only one is active.

## Project conventions

### Digits

Sequential CCW around each parent hex, anchored to the per-pentagon
primary direction. ``digit_offset`` from `face_lattice.py`:

```
0 = center            4 = 240°
1 =  60° (DELETED)    5 = 300°
2 = 120°              6 =   0° (primary)
3 = 180°
```

CCW 60° rotation: `1 → 2 → 3 → 4 → 5 → 6 → 1`. The +60° intra-pentagon
stitch maps the deleted wedge ``[0°, 60°)`` to the d=2 wedge
``[60°, 120°)``.

### Cell tuples

`(base, d_1, ..., d_N)` where `base ∈ {0..11}` (icosa vertex index) and
each `d_k ∈ {0..6}`. Paths whose first non-zero digit is 1 are
*excluded* — they'd live in some pentagon's deleted wedge. Pentagon-
center cells are `is_pentagon(cell)` = "all child digits zero."

Public helpers (in `mu3.cell`): `is_valid_cell`, `is_pentagon`,
`cell_resolution`. Prefer these over inline `len(cell) - 1` /
`all(d == 0 for d in cell[1:])` in package code.

### Resolution-rotation alternation

`get_rot(res)` accumulates `s7b = 3 + ω` per odd level (Class III,
~+19.106° rotation) and `7` per even level (Class II, no rotation).
Walks at level k use ratio `s7b` if k is odd, `s7a = 2 - ω` if even.

## Ring-1 neighbor walk & point location — holonomy walk (adopted)

`mu3.neighbor` is the exact holonomy walk (adopted 2026-07-02;
formerly `neighbor_holonomy.py`): exact `Eis` positions carrying the
groupoid arrow (a transported source center / query witness); one
seam-side sign test replaces all geometric disambiguation. `index.py`
point location goes `_sphere_to_flat` → exact snap
(`eisenstein.from_complex`) → `resolve_position` → banded spherical
polish (`_polish_banded`: flat edge distances gate which edges get a
great-circle side test; stitch-region cells and actual crossings fall
back to the full `_polish`). The polish contract is the SINGLE-EDGE
INVARIANT: the raw cell is correct or off by exactly one violated
edge — an invariant of the architecture (exact shared corners confine
flat/spherical mismatch to per-edge bow lenses), never a search
radius. Do not reason "if X changes we may need to check 2-3 hops":
if a margin erodes, the projection or its fitted constants are what
get fixed (`test_one_hop_contract.py` fails by name).
Correctness rests on implementation-agnostic tests only:
`test_neighbor.py` invariants (size/symmetry/distance-band/CCW/
primary-last) and `test_point_location.py` containment.

The cocycle `TAU` in `cross_pentagon.py` retypes `CROSS_PENTAGON` as
`P6` group elements; `_check_cocycle()` derives the +60° stitch and
asserts the (branch-cut-aware) consistency laws at import — don't
"simplify" them to the idealized single-stitch identity.

The retired geometric walk (sphere-distance twin picks) matched the
holonomy walk exactly on all cells res 0–6
(~1.37M) before deletion. The bottom-up digit carry (the old
`NEIGHBOR_TRANS` idea) was revived 2026-07-04 as `step`'s interior
fast tier (`eisenstein.CARRY`/`carry_digits`, dispatch in
`neighbor._step_fast`): tables derived from `divmod_eis` at import;
any seam outcome (root escape, phantom form) falls back to the full
arrow walk, so the seam-side bit that killed the original is never
needed on the fast path. Pinned by exhaustive carry-vs-walk equality
(res 0–3 in-suite; res 0–5 via `scripts/verify_carry_walk.py`).

## Gotchas

- **`_BOW_COEFFS` in `index.py` is coupled to the active projection.**
  The banded polish's parabolic bow envelope is projection-independent
  in FORM (exact shared corners + smooth edges), but its per-res
  coefficients are fitted to `_PROJECTION_CLS`. Swapping projections:
  rerun `scripts/measure_polish_band.py`, add the new table entry
  (keyed by class name — missing entry raises, stale one fails
  `test_bow_envelope_headroom`). The coefficients PLATEAU with res
  (~0.06 measured): worst edges cross chart seams, where the C⁰ kink
  makes relative bow scale-free — don't "fix" the non-decaying tail.

- **Same-base flat distance is NOT preserved by the +60° stitch.**
  Cells like `(0, 0, 2)` and `(0, 0, 6)` are 3D-adjacent but at flat
  distance `√3 / 7` — not `1 / 7`. Sphere distance is the right
  invariant; the test suite uses it.
- **Cells_at_res excludes leading-zero d=1 paths**, but some
  walked-position digit strings *only* admit a leading-zero d=1
  representation. These are 3-hex corners shared by 3 cells (one
  phantom + two real); the phantom's two rotation twins represent
  different 3D points and `cell_ring1` picks the geometrically nearer.
- **Tests run in `tests/test_neighbor.py` and `tests/test_cell.py`** —
  most invariants are checked at res 0-3. Higher-res spot checks would
  require new tests.

## Key files

- `src/mu3/face_lattice.py` — float Eisenstein arithmetic,
  digit_offset, rot.
- `src/mu3/eisenstein.py` — exact integer `Eis` type, exact divmod.
- `src/mu3/p6.py` — the `P6` group (`z ↦ ζᵘz + t`), stitch constant.
- `src/mu3/cell.py` — `cells_at_res`, `cell_center`, `is_valid_cell`,
  `is_pentagon`.
- `src/mu3/cross_pentagon.py` — cross-pentagon transform table,
  `TAU` cocycle + `_check_cocycle`, float `z_to_cell` (scripts only).
- `src/mu3/neighbor.py` — exact `cell_ring1` + `resolve_position`
  (the holonomy walk), `step` (the arrow-returning walk).
- `src/mu3/edge.py` — `DirectedEdge`/`UndirectedEdge`:
  arrow-transported `reverse`, lex-min canonical form; wire-pair
  geometry fast path `edge_to_boundary` (the identity-vs-geometry
  tier rule lives in this module's docstring).
- `src/mu3/vertex.py` — `Vertex`: Z/3 orbit of `(cell, d)` corner
  names, exact S3-scaled corner positions, and the edge↔vertex
  incidence block (complete combinatorial map; laws in
  `tests/test_incidence.py`).
- `src/mu3/island.py` — Gosper island boundary iterator: the directed
  edges around a cell's descendant set, O(1) sequence work per edge
  (H3 PR #1138 design, constants fitted + oracle-pinned).
