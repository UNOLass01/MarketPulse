.PHONY: up down reset logs test test-ci test-int test-e2e lint format typecheck migrate revision seed install

# Everything that is Python and ours. services/ and scripts/ were excluded
# until Phase 5 -- which is exactly how a whole FastAPI app can land in
# services/api/ without ruff, black, or mypy ever looking at it. CI runs
# these same targets, so the two can't drift apart.
LINT_PATHS := src services scripts tests airflow
TYPE_PATHS := src services scripts

# --env-file .env, not --project-directory: without either, Compose looks
# for .env next to the *first* -f file (docker/.env), not the repo root --
# silently mismatching .env.example's documented "copy to repo root"
# convention. Every var had a `:-default` fallback until Phase 4 added
# required (`:?`) Airflow secrets, which is what surfaced this.
# --project-directory would fix .env lookup too, but it *also* rebases
# every relative build.context/volume path onto the project directory
# instead of each compose file's own directory, breaking `context: ..`
# (verified: it resolves one level above the repo root). --env-file only
# changes where .env is read from.
COMPOSE := docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml --env-file .env

install:
	pip install -e ".[dev]"

up:
	$(COMPOSE) up -d
	$(COMPOSE) up --wait

down:
	$(COMPOSE) down

reset:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f $(s)

test:
	pytest tests/unit tests/contract

# Every tier that can run without compose, plus the coverage floor. The
# floor is scoped to features/ and ml/ (CLAUDE.md: "blanket repo-wide
# coverage is not a goal") -- a repo-wide number would be satisfiable by
# testing wiring instead of the leakage- and promotion-sensitive code that
# actually matters.
#
# Unit and integration run together, not separately: ml/pipeline.py is
# orchestration whose correctness is only demonstrated by the integration
# tier, so a unit-only measurement reports it as 0% and the gate becomes a
# number nobody can hit honestly.
#
# The include patterns need the leading `*/` -- coverage's `*` does not
# match a path separator, so `*marketpulse/ml/*` silently matches nothing
# and the gate passes on an empty report.
test-ci:
	pytest tests/unit tests/contract tests/integration \
	  --cov=marketpulse --cov-report=term-missing
	coverage report \
	  --include='*/marketpulse/features/*,*/marketpulse/ml/*' \
	  --fail-under=85

test-int:
	pytest tests/integration

test-e2e:
	pytest tests/e2e

lint:
	ruff check $(LINT_PATHS)
	black --check $(LINT_PATHS)

format:
	ruff check --fix $(LINT_PATHS)
	black $(LINT_PATHS)

# airflow/dags is intentionally excluded: Airflow's TaskFlow decorators
# aren't a great fit for strict mode. DAG correctness is covered instead by
# tests/unit/test_dags.py (parses every DAG file, no import errors,
# cross-cutting rules) per the phase-4 plan. services/ and scripts/ are
# *not* exempt -- they were only ever excluded by omission.
typecheck:
	mypy $(TYPE_PATHS)

migrate:
	alembic upgrade head

revision:
	alembic revision --autogenerate -m "$(m)"

seed:
	python scripts/seed_historical.py
