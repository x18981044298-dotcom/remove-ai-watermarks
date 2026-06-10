#!/usr/bin/env bash

set -euo pipefail

uv sync --all-extras
uv run uv-outdated
uv run uv-secure --ignore-unfixed
uv run ruff check --fix
uv run ruff format
# Scoped to src/: a full-project pyright run OOM-crashes node on this ML-heavy
# repo (see CLAUDE.md "Test and lint"); src/ is the authoritative strict gate.
uv run pyright src/
uv run pytest -n auto
