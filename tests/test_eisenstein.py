"""Exact Eisenstein arithmetic vs the float ``complex`` implementation."""

import random

import pytest

from mu3.cell import _eisenstein_center, cells_at_res
from mu3.cross_pentagon import EISENSTEIN_UNITS, z_to_cell
from mu3.eisenstein import (
    DIGIT_OFFSET,
    ONE,
    S7A,
    S7B,
    UNITS,
    ZERO,
    ZETA,
    ZETA_INV,
    Eis,
    divmod_eis,
    extract_digits,
    get_rot_eis,
    hex_dist,
    scaled_center,
)
from mu3.face_lattice import digit_offset, divmod_ei, get_rot, s7a, s7b


def test_units_match_complex_units():
    for i in range(6):
        assert UNITS[i].to_complex() == pytest.approx(EISENSTEIN_UNITS[i])


def test_constants_match_face_lattice():
    assert S7A.to_complex() == pytest.approx(s7a)
    assert S7B.to_complex() == pytest.approx(s7b)
    for d in range(7):
        assert DIGIT_OFFSET[d].to_complex() == pytest.approx(digit_offset[d])


def test_zeta_powers():
    z = ONE
    for i in range(6):
        assert z == UNITS[i]
        z = z * ZETA
    assert z == ONE
    assert ZETA * ZETA_INV == ONE


@pytest.mark.parametrize('res', range(7))
def test_get_rot_eis_matches_float(res):
    assert get_rot_eis(res).to_complex() == pytest.approx(get_rot(res))


def test_mul_matches_complex():
    rng = random.Random(0)
    for _ in range(200):
        x = Eis(rng.randint(-50, 50), rng.randint(-50, 50))
        y = Eis(rng.randint(-50, 50), rng.randint(-50, 50))
        assert (x * y).to_complex() == pytest.approx(
            x.to_complex() * y.to_complex()
        )


def test_conj_and_norm():
    rng = random.Random(1)
    for _ in range(200):
        z = Eis(rng.randint(-50, 50), rng.randint(-50, 50))
        assert z * z.conj() == Eis(z.norm(), 0)
        assert z.norm() == pytest.approx(abs(z.to_complex()) ** 2)


@pytest.mark.parametrize('d', [S7A, S7B])
def test_divmod_eis_law_and_min_norm(d):
    rng = random.Random(2)
    for _ in range(500):
        z = Eis(rng.randint(-1000, 1000), rng.randint(-1000, 1000))
        q, r = divmod_eis(z, d)
        assert q * d + r == z
        assert r.norm() < d.norm()
        # Local minimality: no unit shift of the quotient does better.
        for u in UNITS:
            assert r.norm() <= (z - (q + u) * d).norm()


def test_divmod_eis_early_exit_equivalence():
    """The early exit (``4*norm(r0) < norm(d)`` => uniquely minimal)
    must agree with an unconditional candidate search — across walk
    divisors (norm 7, where it always fires) and small divisors
    (norms 1..4, where near-ties keep the search reachable)."""
    from mu3.eisenstein import ZETA, _round_frac

    def search_oracle(z, d):
        n = d.norm()
        w = z * d.conj()
        q = Eis(_round_frac(w.a, n), _round_frac(w.b, n))
        best_q, best_r = q, z - q * d
        for u in UNITS:
            q2 = q + u
            r2 = z - q2 * d
            if r2.norm() < best_r.norm():
                best_q, best_r = q2, r2
        return best_q, best_r

    rng = random.Random(4)
    divisors = [S7A, S7B, ZETA, Eis(1, -1), Eis(2, 0)]
    for _ in range(400):
        z = Eis(rng.randint(-500, 500), rng.randint(-500, 500))
        for d in divisors:
            q, r = divmod_eis(z, d)
            oq, orr = search_oracle(z, d)
            assert q * d + r == z
            assert r.norm() == orr.norm(), (z, d)
            if 4 * r.norm() < d.norm():
                assert (q, r) == (oq, orr), (z, d)


def test_divmod_eis_matches_float_on_digit_domain():
    """Where the remainder is a digit offset (the walk's whole domain),
    exact and float divmod must agree."""
    rng = random.Random(3)
    for _ in range(500):
        d = S7B if rng.random() < 0.5 else S7A
        q_in = Eis(rng.randint(-100, 100), rng.randint(-100, 100))
        r_in = DIGIT_OFFSET[rng.randint(0, 6)]
        z = q_in * d + r_in
        q, r = divmod_eis(z, d)
        assert (q, r) == (q_in, r_in)
        qf, rf = divmod_ei(z.to_complex(), d.to_complex())
        assert qf == pytest.approx(q.to_complex())
        assert rf == pytest.approx(r.to_complex(), abs=1e-9)


def test_carry_digits_basics():
    from mu3.eisenstein import carry_digits, scaled_center

    # interior absorb: matches center arithmetic exactly
    for d in (1, 2, 3, 4, 5, 6):
        got = carry_digits((3, 0), d)
        assert got is not None
        assert scaled_center(got) == \
            scaled_center((3, 0)) + DIGIT_OFFSET[d], d
    # res-0 cell: no digits to absorb into -> escape
    assert carry_digits((), 6) is None
    # pentagon-center step lands on the phantom FORM (leading 1) —
    # returned as-is; the walk layer owns the twin fix
    assert carry_digits((0, 0), 1) == (0, 1)


def test_hex_dist():
    assert hex_dist(ZERO) == 0
    for u in UNITS:
        assert hex_dist(u) == 1
    # Ball counts match the pure-lattice law.
    for k in range(5):
        n = sum(1 for a in range(-k, k + 1) for b in range(-k, k + 1)
                if hex_dist(Eis(a, b)) <= k)
        assert n == 1 + 3 * k * (k + 1)
    # Negation symmetry and the triangle inequality on random pairs.
    rng = random.Random(9)
    for _ in range(200):
        x = Eis(rng.randint(-20, 20), rng.randint(-20, 20))
        y = Eis(rng.randint(-20, 20), rng.randint(-20, 20))
        assert hex_dist(x) == hex_dist(-x)
        assert hex_dist(x + y) <= hex_dist(x) + hex_dist(y)


def test_line_digits_laws():
    """line_digits is the constructive counterpart of hex_dist: its
    length is the hex distance and its offsets sum to the input."""
    from mu3.eisenstein import line_digits
    rng = random.Random(11)
    for _ in range(300):
        delta = Eis(rng.randint(-15, 15), rng.randint(-15, 15))
        digits = line_digits(delta)
        assert len(digits) == hex_dist(delta)
        total = ZERO
        for d in digits:
            total = total + DIGIT_OFFSET[d]
        assert total == delta


# --- exact digit-string round-trip on all res 0-3 cells ----------------

@pytest.mark.parametrize('res', range(4))
def test_roundtrip_all_cells(res):
    for cell in cells_at_res(res):
        digits = cell[1:]
        z = scaled_center(digits)
        out, root = extract_digits(z, res)
        assert tuple(out) == digits
        assert root == ZERO
        # Exact scaled center matches the float center.
        assert z.to_complex() == pytest.approx(
            _eisenstein_center(digits) * get_rot(res), abs=1e-9
        )
        # And float z_to_cell agrees end-to-end.
        assert z_to_cell(cell[0], _eisenstein_center(digits), res) == cell
