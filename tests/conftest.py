"""Shared test helpers."""

from mu3.eisenstein import first_nonzero_digit


def random_valid_cells(rng, res: int, n: int) -> list[tuple]:
    """``n`` random valid cells at ``res`` (rejection on the
    leading-digit-1 rule), from a seeded ``random.Random``."""
    out = []
    while len(out) < n:
        digits = [rng.randrange(7) for _ in range(res)]
        if first_nonzero_digit(digits) != 1:
            out.append((rng.randrange(12), *digits))
    return out
