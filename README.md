# mu3

A prototype Discrete Global Grid System (DGGS).

## Design

- **Base polyhedron:** icosahedron.
- **Resolution 0:** 12 pentagon cells, centered at the icosahedron vertices.
- **Aperture:** 7 (hexagonal refinement); finer resolutions are hexagons plus
  the 12 pentagons that persist at every resolution.
- **Projection:** swappable. We start with a simple gnomonic map (face-centered)
  to get something end-to-end, then swap in a better projection that targets
  approximately equal-area cells.
- **Indexing:** designed so that cell neighborhoods, traversal, and hierarchy
  have no kinks across icosahedron edges or at pentagon vertices.

## Relationship to prior exploration

This package consolidates ideas prototyped in sibling folders:

- `2026-04-18_distort/` — projection distortion analysis, α-slerp-rich maps,
  D₃ corrections, pentagon corner-aspect floor, subdivided-icosa null result.
- `2025-09-29_conic_aspect_ratio/` — conic aspect-ratio experiments.
- `2026-04-02_eisint/` — Eisenstein-integer hex arithmetic on a single face.
- `mu3/` — written notes and proposals.

`mu3` is the working-code home for the DGGS those notes point toward. It
is self-contained — no dependency on dggrid or other DGGS libraries.

## Layout

```
src/mu3/
    icosahedron.py   # base geometry: vertex, edge, face tables
    projection.py    # swappable face <-> sphere maps (gnomonic, ...)
    cell.py          # hex/pentagon cell model
    index.py         # cell indexing (kink-free across faces)
```

## Development

```
just test     # reinstall + run pytest
just lab      # jupyter lab
just clean    # scrub build/cache junk
just purge    # also remove .venv and uv.lock
```
