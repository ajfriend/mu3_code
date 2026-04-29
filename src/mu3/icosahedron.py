"""Icosahedron base geometry: 12 vertices, 30 edges, 20 faces.

Indexing is delegated to :mod:`mu3.dodec`: vertex ``i`` is
``dodec.normals[i]`` (the H3-aligned, water-vertex orientation), and the
per-vertex 5-fold neighbor order is :data:`mu3.dodec.neighbors` -- the same
CCW order from each pentagon's primary direction the digit convention is
anchored to.
"""

from functools import lru_cache

import numpy as np

from . import dodec


# Results of these builders are cached: the icosahedron is fixed, and
# callers in the projection pipeline hit these on every corner. The returned
# numpy arrays are treated as read-only (do not mutate in place).
@lru_cache(maxsize=None)
def vertices() -> np.ndarray:
    """Return the 12 unit-sphere icosahedron vertices as a (12, 3) array.

    Indexing matches :data:`mu3.dodec.normals`: the icosahedron is rotated
    into H3's orientation (vertices over water), with ``V[0]`` at H3's
    pole-closest vertex (lat 64.7 N, lng 10.5 E -- Greenland Sea).
    """
    return np.stack(dodec.normals)


@lru_cache(maxsize=None)
def faces() -> np.ndarray:
    """Return the 20 triangular faces as a (20, 3) array of vertex indices.

    Built from :data:`mu3.dodec.neighbors`: for each pentagon ``p``, every
    consecutive pair ``(neighbors[p][k], neighbors[p][k+1])`` together with
    ``p`` forms an icosahedron face. Each face is enumerated 3 times (once
    per corner pentagon); we dedupe by sorted vertex triple, then orient
    each row CCW when viewed from outside the sphere (so
    ``cross(V[F[f,1]] - V[F[f,0]], V[F[f,2]] - V[F[f,0]])`` points outward).
    Row order is the lexicographic sort of the sorted triples.
    """
    V = vertices()
    triples: set[tuple[int, int, int]] = set()
    for p in range(12):
        n = dodec.neighbors[p]
        for k in range(5):
            triples.add(tuple(sorted((p, n[k], n[(k + 1) % 5]))))
    assert len(triples) == 20, f"expected 20 unique faces, got {len(triples)}"
    out = []
    for tri in sorted(triples):
        a, b, c = tri
        normal = np.cross(V[b] - V[a], V[c] - V[a])
        # Center of the (rough) triangle is V[a]+V[b]+V[c]; if normal
        # points inward, swap two vertices to flip orientation.
        center = V[a] + V[b] + V[c]
        if normal @ center < 0:
            tri = (a, c, b)
        out.append(tri)
    return np.array(out, dtype=np.int64)


@lru_cache(maxsize=None)
def face_centers() -> np.ndarray:
    """Return (20, 3) unit-sphere face centers (centroids, renormalized)."""
    V = vertices()
    F = faces()
    c = V[F].mean(axis=1)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def vertex_neighbors() -> np.ndarray:
    """Return (12, 5) array: the 5 icosa-vertex neighbors of each vertex.

    Order is CCW around each vertex from the primary direction, taken
    directly from :data:`mu3.dodec.neighbors` -- column 0 is the same-axis
    (merged-corner) neighbor (target of the primary tangent).
    """
    return np.array(dodec.neighbors, dtype=np.int64)


@lru_cache(maxsize=None)
def pentagon_face_table() -> np.ndarray:
    """Return (12, 5) int64: ``pentagon_face_table[p, d - 2]`` is the icosa
    face that digit ``d`` in ``{2, 3, 4, 5, 6}`` points into from base
    pentagon ``p``.

    Closed-form from :data:`mu3.dodec.neighbors`. Digit ``d``'s face is the
    triangle ``(p, n[d - 2], n[(d - 1) % 5])`` where ``n = neighbors[p]``
    is the 5-tuple of neighbor vertices CCW from the primary direction.
    """
    F = faces()
    triple_to_idx = {
        tuple(sorted(int(x) for x in F[i])): i for i in range(20)
    }
    out = np.empty((12, 5), dtype=np.int64)
    for p in range(12):
        n = dodec.neighbors[p]
        for k in range(5):
            triple = tuple(sorted((p, n[k], n[(k + 1) % 5])))
            out[p, k] = triple_to_idx[triple]
    return out


