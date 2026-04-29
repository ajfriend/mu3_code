export UV_NO_EDITABLE := "1"
export UV_OFFLINE := "0"  # toggle on when offline to avoid failures

import 'figures.just'

_:
    just --list

test: reinstall
    uv run pytest

test-one path: reinstall
    uv run pytest {{path}}

reinstall:
    uv sync --reinstall-package mu3

lab: reinstall
    uv run jupyter lab

clean:
    just _rm __pycache__
    just _rm '*.pytest_cache'
    just _rm '*.egg-info'
    just _rm .DS_Store
    just _rm .ipynb_checkpoints


purge: clean
    just _rm .venv
    just _rm uv.lock

_rm pattern:
    -@find . -name "{{pattern}}" -prune -exec rm -rf {} +
