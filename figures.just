# Figure / plot / HTML recipes. Imported by the root justfile; calling
# convention stays flat (e.g. `just globe`).

[group('extra')]
globe: reinstall
    uv run scripts/make_globe_plot.py
    open figures/mu3_globe.html

[group('extra')]
primary-direction:
    uv run scripts/make_primary_direction_plot.py
    open figures/primary_direction.html
