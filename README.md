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

Phase 0 (Foundations) — in progress. See [`docs/plan/phase-0-foundations.md`](docs/plan/phase-0-foundations.md)
and the phase table in `CLAUDE.md` for what's next.

## Quickstart

```bash
cp .env.example .env        # adjust values if needed
make install                # pip install -e ".[dev]"
make up                     # start Postgres + RabbitMQ (docker compose)
make migrate                # apply DB schema
make test                   # unit + contract tests (fast, no I/O)
```

Other useful targets: `make down`, `make reset` (drops volumes), `make logs s=<service>`,
`make lint`, `make typecheck`, `make test-int`, `make test-e2e`. Full list in the `Makefile`
and in `CLAUDE.md`.

## Repository layout

See `docs/ARCHITECTURE.md` §3 for the annotated tree. In short: all logic lives under
`src/marketpulse/`; `services/` and `airflow/dags/` are thin entrypoints that import from it.

## Contributing

Read `CLAUDE.md` before making changes — it lists the non-negotiable rules (feature purity,
no look-ahead, chronological splits, etc.) that keep this pipeline correct. Work one phase
at a time per `docs/plan/`; each phase has its own exit criterion.
