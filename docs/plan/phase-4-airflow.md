# Phase 4 — Orchestration

**Objective:** the batch plane runs itself.
**Complexity:** Medium · **Effort:** ~2 days · **Depends on:** Phase 3
**Exit criterion:** a full retrain runs unattended on schedule, **and** a deliberately corrupted data window causes the quality gate to block it.

## Tasks

### Setup
- [x] Airflow scheduler + webserver in compose, LocalExecutor (not Celery — no benefit at this concurrency)
- [x] `airflow.Dockerfile` — base image + the `marketpulse` package installed
- [x] Connections/Variables for DB, MLflow, S3
- [x] Resource limits set (scheduler peaks during training; the host is shared)

### `dag_data_quality` — hourly
- [x] Freshness: newest tick within expected interval (**highest-value task in the whole portfolio** — catches silent producer death)
- [x] Completeness: expected vs actual row count, gap detection
- [x] Validity: nulls in non-nullable features, out-of-range prices, non-monotonic timestamps
- [x] Distribution sanity: implausible shift in any feature mean
- [x] Persist to `quality_checks`; branch to alert on failure

### `dag_model_retraining` — daily 02:00 UTC
- [x] Sensor/gate on the latest `dag_data_quality` run passing
- [x] extract → validate sufficiency → train → evaluate → promotion gate → (promote + snapshot reference | hold in Staging with reason) → notify
- [x] Archive the previous Production version on promotion
- [x] Idempotent: re-running creates a *new* run and version, never mutates a prior one

### `dag_feature_backfill` — manual trigger only
- [x] Params: `start`, `end`, `symbols`, `feature_set_version`
- [x] Chunk by symbol + date to bound memory
- [x] Upsert so partial reruns are safe
- [x] Uses the same `marketpulse.features` module as the consumer

### `dag_data_archival` — daily 04:00 UTC
- [x] identify partitions past hot retention → export Parquet to object storage → **verify (row count + checksum)** → drop partition → record
- [x] The verify step is non-negotiable; it's what separates maintenance from data loss

### `dag_partition_maintenance` — monthly
- [x] Create the next two months of partitions ahead (a missing partition = insert error at midnight on the 1st)

### Cross-cutting
- [x] Every DAG: `catchup=False`, owner, tags, docstring
- [x] Every task: idempotent, `retries=2` with exponential backoff, explicit `execution_timeout`, `on_failure_callback`
- [x] XCom for small metadata only — data passes via Postgres/object storage

## Tests
- [x] **DAG import test in CI** — parse every DAG file. A broken DAG silently disables scheduling; this is a genuinely nasty failure mode.
- [x] No cycles; expected task counts and dependency edges
- [x] All DAGs assert `catchup=False`
- [x] All tasks assert a non-null `execution_timeout`
- [x] Quality gate blocks retraining on an injected corrupt window (integration)
- [x] Backfill is idempotent: run twice → same row count
- [x] Archival verify step fails closed on a checksum mismatch (drop must not execute)

## Watch out for
- A newly deployed DAG with a past `start_date` and `catchup=True` triggers a stampede of historical runs. Classic first-day incident.
