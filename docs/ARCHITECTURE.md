# MarketPulse — Architecture

## 1. Overview

Real-time crypto streaming analytics + ML pipeline:

```
Producer → RabbitMQ → Consumer → PostgreSQL → Airflow → LightGBM → MLflow → FastAPI → drift monitoring
```

A poller pulls ticks from a market-data provider and publishes them to
RabbitMQ. A consumer validates and persists them to Postgres. Airflow
orchestrates periodic feature backfill, training, evaluation, and promotion.
FastAPI serves predictions from the current Production model, reading
precomputed feature rows — it never computes features itself. A monitoring
component watches for feature/prediction drift against the training
baseline.

## 2. Components

| Component | Responsibility | Lives in |
|---|---|---|
| Ingestion | Poll provider, publish ticks | `src/marketpulse/ingestion/` |
| Messaging | RabbitMQ topology, consumer base class, (de)serialization | `src/marketpulse/messaging/` |
| Storage | Engine, ORM models, repositories — the only place queries are written | `src/marketpulse/storage/` |
| Features | Pure, versioned feature functions computed from stored ticks | `src/marketpulse/features/` |
| ML | Dataset assembly, labeling, training, evaluation, registry, prediction | `src/marketpulse/ml/` |
| Monitoring | Drift, performance, quality checks, alerting | `src/marketpulse/monitoring/` |
| Orchestration | Airflow DAGs — thin callables into the above, no logic | `airflow/dags/` |
| Serving | FastAPI app reading stored feature rows + Production model | `services/api/` (Phase 5) |

See `CLAUDE.md` for the non-negotiable rules that keep these boundaries
intact (feature purity, no look-ahead, chronological splits, API never
computes features, etc.) — this document describes *what* the system is;
`CLAUDE.md` enforces *how* it stays correct.

## 3. Repository layout

```
src/marketpulse/     shared package — ALL logic lives here
  contracts/          Pydantic schemas: messages, features, api
  ingestion/          providers (behind an interface), poller, publisher
  messaging/          connection, topology, base consumer, serialization
  features/           pure functions + versioned feature registry
  storage/            engine, ORM models, repositories (all queries)
  ml/                 dataset, labeling, train, evaluate, registry, predict
  monitoring/          drift, performance, quality, alerts
services/             thin entrypoints only — wiring, no logic
airflow/dags/         orchestration only
migrations/           alembic; never hand-edit schema
tests/                unit | contract | integration | e2e
docs/plan/            per-phase task + test checklists
docs/architecture/    ADRs and this document
```

Rule of thumb: if deleting `src/` would leave a service still doing
something interesting, logic has leaked out of the package.

## 4. Data flow and timing model

- Every record has both `observed_at` (when the underlying event occurred /
  became knowable) and `ingested_at` (when our system wrote it).
- Model-relevant queries filter on `observed_at`. `ingested_at` is
  operational metadata only — using it for training/serving data leaks
  information that wasn't actually available at that point in time.
- Labels depend on future price (`t + H`) and are therefore computed at
  training time from stored history, never at ingestion.

## 5. Environments

Local development runs the full stack via `docker compose` (Postgres +
RabbitMQ in Phase 0; application services are added as later phases
introduce them — see the phase table in `CLAUDE.md`). CI runs lint, type
checks, and unit/contract tests only; integration and e2e tiers require
Docker and are run separately (`make test-int`, `make test-e2e`).

## 6. Decisions

Non-obvious or debatable choices are recorded as ADRs in
`docs/architecture/decisions/`, using the template in that directory. Start
here: [0001 — Why RabbitMQ over Kafka](architecture/decisions/0001-why-rabbitmq-over-kafka.md).
