.PHONY: up down reset logs test test-int test-e2e lint format typecheck migrate revision seed install

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

test-int:
	pytest tests/integration

test-e2e:
	pytest tests/e2e

lint:
	ruff check src tests airflow
	black --check src tests airflow

format:
	ruff check --fix src tests airflow
	black src tests airflow

# airflow/dags is intentionally excluded: pyproject.toml's [tool.mypy]
# scopes strict checking to the marketpulse package, and Airflow's TaskFlow
# decorators aren't a great fit for that same strictness. DAG correctness is
# covered instead by tests/unit/test_dags.py (parses every DAG file, no
# import errors, cross-cutting rules) per the phase-4 plan.
typecheck:
	mypy src

migrate:
	alembic upgrade head

revision:
	alembic revision --autogenerate -m "$(m)"

seed:
	python scripts/seed_historical.py
