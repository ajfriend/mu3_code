import cmath
import math

from mu3.face_lattice import (
    digit_offset,
    divmod_ei,
    get_rot,
    omega,
    pentagon_skipped_digit,
    round_ei,
    s3,
    s7a,
    s7b,
    to_ab,
    units,
)


def test_omega_is_primitive_cube_root_of_unity():
    assert abs(omega**3 - 1) < 1e-12
    assert abs(1 + omega + omega**2) < 1e-12


def test_units_have_unit_magnitude_at_60deg_steps():
    angles = sorted((cmath.phase(u) % (2 * math.pi)) for u in units)
    for k, u in enumerate(units):
        assert abs(abs(u) - 1.0) < 1e-12
    for k in range(6):
        assert abs(angles[k] - k * math.pi / 3) < 1e-12


def test_digit_offsets_are_unit_magnitude():
    for d in range(1, 7):
        assert abs(abs(digit_offset[d]) - 1.0) < 1e-12
    assert digit_offset[0] == 0


def test_ccw_cycle_is_sequential():
    # +60° CCW rotation cycles digits 1 → 2 → 3 → 4 → 5 → 6 → 1.
    rot60 = cmath.exp(1j * math.pi / 3)
    for d in range(1, 7):
        d_next = (d % 6) + 1
        assert abs(digit_offset[d] * rot60 - digit_offset[d_next]) < 1e-12


def test_d6_is_along_primary_direction():
    # Digit 6 sits along the primary direction (angle 0°).
    assert abs(digit_offset[6] - 1.0) < 1e-12


def test_pentagon_skipped_digit_is_1():
    assert pentagon_skipped_digit == 1


def test_s3_s7_magnitudes():
    assert abs(abs(s3) ** 2 - 3.0) < 1e-12
    assert abs(abs(s7a) ** 2 - 7.0) < 1e-12
    assert abs(abs(s7b) ** 2 - 7.0) < 1e-12


def test_s7a_and_s7b_are_conjugate_rotations():
    # s7a and s7b have the same magnitude but opposite rotation sign.
    assert abs(cmath.phase(s7a) + cmath.phase(s7b)) < 1e-12


def test_get_rot_scale_by_sqrt7_per_res():
    for res in range(0, 7):
        assert abs(abs(get_rot(res)) ** 2 - 7**res) < 1e-10


def test_get_rot_class_ii_iii_alternation():
    for res in (0, 2, 4, 6):
        # Even: pure real positive — Class II, unrotated vs res 0.
        rot = get_rot(res)
        assert abs(rot.imag) < 1e-10
        assert rot.real > 0
    for res in (1, 3, 5):
        # Odd: proportional to s7b — Class III, rotated by +arg(s7b).
        ratio = get_rot(res) / s7b
        assert abs(ratio.imag) < 1e-10
        assert ratio.real > 0


def test_to_ab_roundtrip():
    for z in [1.0, 1j, 2.3 + 4.5j, -1 - omega, omega**2]:
        a, b = to_ab(z)
        assert abs((a + b * omega) - z) < 1e-12


def test_round_ei_preserves_eisenstein_integers():
    # Eisenstein integers should round to themselves.
    for a in range(-3, 4):
        for b in range(-3, 4):
            z = a + b * omega
            assert abs(round_ei(z) - z) < 1e-12


def test_round_ei_distance_bound():
    # Every point is within distance 1/√3 of the nearest Eisenstein integer
    # (honeycomb covering radius).
    import random

    rng = random.Random(20260421)
    for _ in range(200):
        z = complex(rng.uniform(-5, 5), rng.uniform(-5, 5))
        p = round_ei(z)
        assert abs(z - p) < 1 / math.sqrt(3) + 1e-9


def test_divmod_ei_reconstruction():
    # z = q*d + r, with r the nearest-Eisenstein remainder.
    import random

    rng = random.Random(20260421)
    for d in (s3, s7a, s7b, 2 + 3 * omega):
        for _ in range(50):
            z = complex(rng.uniform(-10, 10), rng.uniform(-10, 10))
            q, r = divmod_ei(z, d)
            assert abs(q * d + r - z) < 1e-10
            # |r/d| should be within the covering radius.
            assert abs(r / d) < 1 / math.sqrt(3) + 1e-9
