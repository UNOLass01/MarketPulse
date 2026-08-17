# MarketPulse

Real-time crypto streaming analytics + ML pipeline.

```
Producer → RabbitMQ → Consumer → PostgreSQL → Airflow → LightGBM → MLflow → FastAPI → drift monitoring
```

> Architecture diagram: TODO — add once the component boundaries in
> `docs/ARCHITECTURE.md` are stable (tracked for Phase 7).

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Project rules and conventions for contributors (human or AI): [`CLAUDE.md`](CLAUDE.md).
Phase-by-phase build plan: [`docs/plan/`](docs/plan/).

## Status

Phases 0–6 complete; Phase 7 (deploy + docs) is next. See the phase table in
`CLAUDE.md` and the per-phase plans in [`docs/plan/`](docs/plan/).

## Quickstart

```bash
cp .env.example .env        # adjust values if needed
make install                # pip install -e ".[dev]"
make up                     # start the stack (docker compose)
make migrate                # apply DB schema
make test                   # unit + contract tests (fast, no I/O)
```

Once the stack is up:

| What | Where |
|---|---|
| Prediction API + generated OpenAPI docs | <http://localhost:8000/docs> |
| Monitoring dashboard (read-only) | <http://localhost:8501> |
| Airflow | <http://localhost:8080> |
| MLflow | <http://localhost:5000> |
| RabbitMQ management | <http://localhost:15672> |

The producer and consumer run as plain processes rather than compose services:

```bash
python -m services.producer.main
python -m services.consumer.main
```

Other useful targets: `make down`, `make reset` (drops volumes), `make logs s=<service>`,
`make lint`, `make typecheck`, `make test-int`, `make test-e2e`. Full list in the `Makefile`
and in `CLAUDE.md`.

## Operations

Alerts name their runbook, and the runbooks live in [`docs/runbooks/`](docs/runbooks/):
[consumer lag / stale features](docs/runbooks/consumer_lag.md),
[model rollback](docs/runbooks/model_rollback.md),
[DLQ triage](docs/runbooks/dlq_triage.md).

## Repository layout

See `docs/ARCHITECTURE.md` §3 for the annotated tree. In short: all logic lives under
`src/marketpulse/`; `services/` and `airflow/dags/` are thin entrypoints that import from it.

## Contributing

Read `CLAUDE.md` before making changes — it lists the non-negotiable rules (feature purity,
no look-ahead, chronological splits, etc.) that keep this pipeline correct. Work one phase
at a time per `docs/plan/`; each phase has its own exit criterion.
