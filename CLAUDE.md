# mu3 working notes

## Workflow

- **Tests**: `just test` (full suite, ~6s, 148+3 skipped) or
  `just test-one path` for one file or node id. Both reinstall the
  editable package first via `uv sync --reinstall-package mu3`.
  Don't use `uv run --with-editable .` — it bypasses the reinstall.
- **Scripts**: `uv run scripts/foo.py`. Scripts use PEP 723 inline
  metadata for any extra deps; never `--with` flags.
- **Plan files**: persisted in `~/.claude/plans/`; only one is active.
  `todo/*.md` are project-level punch lists, not Claude plans.

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

## Ring-1 neighbor walk

Lives in `mu3.cross_pentagon` + `mu3.neighbor`. Geometric, not
algebraic — walks `z_n = z_C + digit_offset[D]/get_rot(res)`, then
uses `canonicalize` (territory check + intra-pentagon stitch) and
`z_to_cell` (iterated `divmod_ei`). Cross-pentagon hops via
`CROSS_PENTAGON` table; deleted-wedge phantoms via twin disambiguation
(sphere distance + cross-product tiebreak).

The earlier algebraic walk (`NEIGHBOR_TRANS` bottom-up carry, removed
in `8b17e36`) is documented in
`todo/2026-04-28-algebraic-neighbor-walk.md` should it ever need to
come back (e.g. as a compiled fast path).

## Gotchas

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

- `src/mu3/face_lattice.py` — Eisenstein arithmetic, digit_offset, rot.
- `src/mu3/cell.py` — `cells_at_res`, `cell_center`, `is_valid_cell`,
  `is_pentagon`.
- `src/mu3/cross_pentagon.py` — `canonicalize`, `z_to_cell`, the
  cross-pentagon transform table.
- `src/mu3/neighbor.py` — `cell_ring1` (the geometric ring-1 walk).
- `reports/pentagon-centric-indexing.md` — design background.
- `figures/intrapentagon_ring1.png`,
  `figures/phantom_both_twins.png` — the deleted-wedge geometry.
