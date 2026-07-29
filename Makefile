.PHONY: up down reset logs test test-int test-e2e lint format typecheck migrate revision seed install

COMPOSE := docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml

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
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy src

migrate:
	alembic upgrade head

revision:
	alembic revision --autogenerate -m "$(m)"

seed:
	python -m marketpulse.ingestion.seed
