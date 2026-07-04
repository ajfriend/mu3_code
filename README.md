# mu3

A prototype Discrete Global Grid System (DGGS).

## Design

- **Base polyhedron:** icosahedron.
- **Resolution 0:** 12 pentagon cells, centered at the icosahedron vertices.
- **Aperture:** 7 (hexagonal refinement); finer resolutions are hexagons plus
  the 12 pentagons that persist at every resolution.
- **Projection:** swappable; the default (α-slerp) targets approximately
  equal-area cells.
- **Indexing:** designed so that cell neighborhoods, traversal, and hierarchy
  have no kinks across icosahedron edges or at pentagon vertices. The runtime
  core is exact — integer Eisenstein arithmetic plus a holonomy (carried-arrow)
  treatment of the pentagon seams — with floats confined to rendering and
  final point-location polish.

`mu3` is self-contained — no dependency on dggrid or other DGGS libraries.

## Layout

```
src/mu3/
    dodec.py           # base-cell adjacency tables
    icosahedron.py     # base geometry: vertex, edge, face tables
    face_lattice.py    # float Eisenstein arithmetic, digit offsets, rotations
    eisenstein.py      # exact integer Eis type, exact divmod, carry tables
    p6.py              # the P6 group (z -> zeta^u z + t), stitch constant
    cross_pentagon.py  # pentagon-to-pentagon atlas: transitions, TAU cocycle
    cell.py            # cell model: enumeration, centers, boundaries
    neighbor.py        # exact ring-1 / step / position walk (holonomy walk)
    index.py           # point location: latlng/vec3 -> cell
    edge.py            # directed + undirected edges (combinatorial map)
    vertex.py          # vertices as Z/3 orbits; edge<->vertex incidence
    traversal.py       # rings, disks, grid_distance, grid_path
    island.py          # Gosper island boundary iterator
    projection.py      # swappable face <-> sphere maps (alpha-slerp, ...)
    rotation.py        # sphere rotations
```

## Development

```
just test           # reinstall + run pytest
just test-one path  # one test file or node id
just lab            # jupyter lab
just clean          # scrub build/cache junk
just purge          # also remove .venv and uv.lock
```
