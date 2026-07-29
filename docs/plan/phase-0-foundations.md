# Phase 0 — Foundations

**Objective:** repository, tooling, and contracts in place before any feature work.
**Complexity:** Low · **Effort:** ~1 day · **Depends on:** nothing
**Exit criterion:** `make up` starts Postgres + RabbitMQ healthy; `make test` runs (even with zero tests); CI passes on a PR.

## Tasks

### Repo skeleton
- [x] Create folder tree per `docs/ARCHITECTURE.md` §3 (empty `__init__.py` where needed)
- [x] `pyproject.toml` — project metadata, deps, and tool config for ruff, black, mypy, pytest
- [x] Package installable in editable mode (`pip install -e .`) so imports are absolute everywhere
- [x] `.gitignore`, `.dockerignore`, `LICENSE`, `CHANGELOG.md`
- [x] `.env.example` with every variable documented and safe dummy values

### Configuration
- [x] `marketpulse/config.py` — Pydantic `BaseSettings`, `MP_` prefix, nested settings per component
- [x] Validation fails at startup on missing/invalid config (fail fast, do not defer)
- [x] Settings cached via `lru_cache` so config is parsed once

### Logging + errors
- [x] `marketpulse/logging.py` — structured JSON logger factory
- [x] Correlation ID support (contextvar, propagated into every log record)
- [x] `marketpulse/exceptions.py` — hierarchy with `TransientError` and `PermanentError` base classes (Phase 1 depends on this split)

### Infrastructure
- [x] `docker/docker-compose.yml` — postgres + rabbitmq only, with **healthchecks**
- [x] `docker-compose.dev.yml` override — exposed ports, verbose logs
- [x] Named volumes for both services
- [x] Alembic initialised; `migrations/env.py` reads the DB URL from settings
- [x] `Makefile` with all targets listed in CLAUDE.md

### Quality gates
- [x] `.pre-commit-config.yaml` — ruff, black, trailing whitespace, no-commit-to-main
- [x] `.github/workflows/ci.yml` — lint → typecheck → unit tests
- [x] `conftest.py` with shared fixtures scaffold

### Docs
- [x] `README.md` skeleton with the architecture diagram placeholder
- [x] `docs/architecture/decisions/0001-why-rabbitmq-over-kafka.md` (context / options / decision / consequences)
- [x] ADR template file for future decisions

## Tests
- [x] `test_config_loads_from_env` — settings populate from env vars
- [x] `test_config_rejects_invalid` — bad value raises at construction, not at use
- [x] `test_logger_emits_json` — output parses as JSON and contains correlation_id
- [x] `test_exception_hierarchy` — transient and permanent are distinguishable by isinstance

## Watch out for
- `depends_on` alone does **not** wait for readiness. Use `condition: service_healthy`.
- Don't add application services to compose yet — they don't exist.
