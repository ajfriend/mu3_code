'''Carry-walk trial: exhaustive equality of the digit-carry fast tier
(+ arrow fallback) against the pure position walk, with path
distribution, amortized carry length, and step timing.

The adoption gate for the tier-1 fast path (2026-07-04): zero
mismatches over every cell x 6 directions at res 0..RES_MAX — this
EXTENDS the in-suite gate (test_step_equals_position_walk_exhaustive,
res 0-3) to the resolutions the suite doesn't sweep, mirroring the
res 0-6 old-vs-new run that gated the holonomy adoption.
'''

import time

from mu3 import cells_at_res
from mu3.eisenstein import carry_digits, first_nonzero_digit
from mu3.neighbor import _position_step, step

RES_MAX = 5     # 6 for the full adoption-grade run (~minutes)

grand_total = 0
for res in range(RES_MAX + 1):
    t0 = time.perf_counter()
    n = mismatches = 0
    direct = phantom_form = escape = 0
    carry_levels = 0
    for c in cells_at_res(res):
        digits = c[1:]
        for d in (1, 2, 3, 4, 5, 6):
            want = _position_step(c, res, d)
            got = step(c, d)
            n += 1
            if got != want:
                mismatches += 1
                print('MISMATCH', c, d, got, want)
            # classify the carry outcome for the distribution stats
            cd = carry_digits(digits, d)
            if cd is None:
                escape += 1
                carry_levels += res
            else:
                # levels touched = index of deepest unchanged prefix
                changed = next(
                    (k for k in range(res)
                     if cd[k] != digits[k]), res)
                carry_levels += res - changed
                if first_nonzero_digit(cd) == 1:
                    phantom_form += 1
                else:
                    direct += 1
    dt = time.perf_counter() - t0
    grand_total += n
    print(f'res {res}: {n:>9,} steps  mismatches={mismatches}  '
          f'direct={direct / n:7.2%}  phantom={phantom_form / n:6.2%}  '
          f'escape={escape / n:6.2%}  '
          f'avg carry levels={carry_levels / n:4.2f}  ({dt:5.1f}s)')

print(f'total steps checked: {grand_total:,}')

# step timing: interior hex, coarse + fine
for cell in [(0, 3, 5, 2), (0, 3, 5, 2, 4, 6, 3)]:
    N = 5000
    t0 = time.perf_counter()
    for _ in range(N):
        step(cell, 4)
    us = (time.perf_counter() - t0) / N * 1e6
    print(f'interior step at res {len(cell) - 1}: {us:5.1f} us')
