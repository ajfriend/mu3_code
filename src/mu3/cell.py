"""Cell model (placeholder).

At resolution 0 there are 12 pentagon cells, one centered at each
icosahedron vertex. At resolution :math:`r > 0`, refinement is aperture 7
(hexagonal), so the cell count is :math:`12 + 10 \\cdot (7^r - 1)`: the
12 pentagons persist at every resolution, and each icosa face contributes
a growing lattice of hexagons.

This module will grow to expose cell geometry (boundaries on the sphere,
centers) and the pentagon/hex predicate. It is intentionally empty for now.
"""
