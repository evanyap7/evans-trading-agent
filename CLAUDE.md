# Agent instructions

## Deterministic check (backpressure)
Before saying a task is done, or committing, run `scripts/check.sh` (compile, ruff, mypy, pytest) and
read its output. If it fails, fix the cause; never skip hooks (`--no-verify`) or loosen the config
to get green. Report the actual result. New code must pass mypy strictly; the legacy-module
`ignore_errors` list in `pyproject.toml` may only shrink.
