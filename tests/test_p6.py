"""Group laws for P6, exhaustive over rotation parts."""

import itertools
import random

import pytest

from mu3.eisenstein import UNITS, ZERO, Eis
from mu3.p6 import IDENTITY, P6, STITCH, rotation_about


def _some_translations(seed=0, n=4):
    rng = random.Random(seed)
    return [Eis(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in range(n)]


def _some_elements():
    ts = _some_translations()
    return [P6(u, t) for u, t in itertools.product(range(6), ts)]


def test_identity():
    for g in _some_elements():
        assert g.compose(IDENTITY) == g
        assert IDENTITY.compose(g) == g


def test_inverse():
    for g in _some_elements():
        assert g.compose(g.inverse()) == IDENTITY
        assert g.inverse().compose(g) == IDENTITY


def test_associativity():
    els = _some_elements()
    rng = random.Random(4)
    for _ in range(300):
        f, g, h = rng.choice(els), rng.choice(els), rng.choice(els)
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


def test_action_is_homomorphism():
    """``(g o h)(z) == g(h(z))`` for all rotation pairs."""
    ts = _some_translations(seed=5)
    zs = _some_translations(seed=6)
    for u1, u2 in itertools.product(range(6), range(6)):
        for t1, t2 in itertools.product(ts, ts):
            g, h = P6(u1, t1), P6(u2, t2)
            for z in zs:
                assert g.compose(h).apply(z) == g.apply(h.apply(z))


def test_apply_matches_complex():
    for g in _some_elements():
        for z in _some_translations(seed=7):
            expected = UNITS[g.u].to_complex() * z.to_complex() \
                + g.t.to_complex()
            assert g.apply(z).to_complex() == pytest.approx(expected)


def test_stitch_is_plus_60():
    assert STITCH == P6(1, ZERO)
    assert STITCH.apply(UNITS[0]) == UNITS[1]


def test_rotation_about_fixes_center():
    for u in range(6):
        for c in _some_translations(seed=8):
            g = rotation_about(u, c)
            assert g.apply(c) == c
            assert g.u == u
