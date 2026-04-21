import numpy as np

from mu3 import icosahedron


def test_shape_and_dtype():
    t = icosahedron.pentagon_face_table()
    assert t.shape == (12, 5)
    assert t.dtype == np.int64


def test_each_row_has_5_distinct_incident_faces():
    t = icosahedron.pentagon_face_table()
    F = icosahedron.faces()
    for p in range(12):
        row = t[p]
        assert len(set(row.tolist())) == 5
        for f in row:
            assert p in F[f], f"pentagon {p} not in face {f}"


def test_each_face_appears_in_3_rows():
    # Every icosa face has 3 vertices, each hosting a base pentagon.
    t = icosahedron.pentagon_face_table()
    counts = np.bincount(t.ravel(), minlength=20)
    assert np.all(counts == 3)


def test_digit_2_is_smallest_index_incident_face():
    t = icosahedron.pentagon_face_table()
    F = icosahedron.faces()
    for p in range(12):
        incident = sorted(f for f in range(20) if p in F[f])
        # Column for digit 2 is index 0 (since d - 2 == 0 for d == 2).
        assert t[p, 0] == incident[0]


def test_digits_progress_ccw_around_each_pentagon():
    # In V[p]'s tangent plane, visiting faces in the CCW digit sequence
    # (2, 3, 5, 4, 6) must yield angles increasing by ~72° mod 2π.
    t = icosahedron.pentagon_face_table()
    V = icosahedron.vertices()
    C = icosahedron.face_centers()
    ccw_sequence = (2, 3, 5, 4, 6)

    for p in range(12):
        v = V[p]
        ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, v) * v
        u = u / np.linalg.norm(u)
        w = np.cross(v, u)

        angles = []
        for d in ccw_sequence:
            f = t[p, d - 2]
            c_tan = C[f] - np.dot(C[f], v) * v
            angles.append(np.arctan2(np.dot(c_tan, w), np.dot(c_tan, u)))

        # normalize step-to-step differences to [0, 2π)
        diffs = [(angles[(i + 1) % 5] - angles[i]) % (2 * np.pi) for i in range(5)]
        # Each step should be 72° (2π/5) CCW.
        for step in diffs:
            assert abs(step - 2 * np.pi / 5) < 1e-9


def test_reverse_lookup_consistency():
    # If pentagon_face_table[p, d-2] == f, then face f must contain vertex p
    # (trivially, the whole point of the table).
    t = icosahedron.pentagon_face_table()
    F = icosahedron.faces()
    for p in range(12):
        for d in (2, 3, 4, 5, 6):
            f = t[p, d - 2]
            assert p in F[f]
