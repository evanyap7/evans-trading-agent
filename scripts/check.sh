#!/usr/bin/env bash
# Single gate: pre-commit and CI both run this, so they cannot drift.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m compileall -q src scripts
uv run ruff check src tests scripts
uv run mypy
# Coverage floor: ratchet it up as tests are added, never down.
TRADING_MODE=shadow uv run pytest -q --cov=trading_agent --cov-fail-under=75
