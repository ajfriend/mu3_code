export UV_NO_EDITABLE := "1"
export UV_OFFLINE := "0"  # toggle on when offline to avoid failures

_:
    just --list

test: reinstall
    uv run pytest

reinstall:
    uv sync --reinstall-package mu3

[group('extra')]
lab: reinstall
    uv run jupyter lab

[group('extra')]
globe: reinstall
    uv run scripts/make_globe_plot.py
    open figures/mu3_globe.html

[group('extra')]
primary-direction:
    uv run scripts/make_primary_direction_plot.py
    open figures/primary_direction.html

clean:
    just _rm __pycache__
    just _rm '*.pytest_cache'
    just _rm '*.egg-info'
    just _rm .DS_Store
    just _rm .ipynb_checkpoints

# remove env and lockfile
[group('clean')]
purge: clean
    just _rm .venv
    just _rm uv.lock

_rm pattern:
    -@find . -name "{{pattern}}" -prune -exec rm -rf {} +
