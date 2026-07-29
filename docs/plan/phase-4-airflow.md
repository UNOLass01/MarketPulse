# Phase 4 — Orchestration

**Objective:** the batch plane runs itself.
**Complexity:** Medium · **Effort:** ~2 days · **Depends on:** Phase 3
**Exit criterion:** a full retrain runs unattended on schedule, **and** a deliberately corrupted data window causes the quality gate to block it.

## Tasks

### Setup
- [ ] Airflow scheduler + webserver in compose, LocalExecutor (not Celery — no benefit at this concurrency)
- [ ] `airflow.Dockerfile` — base image + the `marketpulse` package installed
- [ ] Connections/Variables for DB, MLflow, S3
- [ ] Resource limits set (scheduler peaks during training; the host is shared)

### `dag_data_quality` — hourly
- [ ] Freshness: newest tick within expected interval (**highest-value task in the whole portfolio** — catches silent producer death)
- [ ] Completeness: expected vs actual row count, gap detection
- [ ] Validity: nulls in non-nullable features, out-of-range prices, non-monotonic timestamps
- [ ] Distribution sanity: implausible shift in any feature mean
- [ ] Persist to `quality_checks`; branch to alert on failure

### `dag_model_retraining` — daily 02:00 UTC
- [ ] Sensor/gate on the latest `dag_data_quality` run passing
- [ ] extract → validate sufficiency → train → evaluate → promotion gate → (promote + snapshot reference | hold in Staging with reason) → notify
- [ ] Archive the previous Production version on promotion
- [ ] Idempotent: re-running creates a *new* run and version, never mutates a prior one

### `dag_feature_backfill` — manual trigger only
- [ ] Params: `start`, `end`, `symbols`, `feature_set_version`
- [ ] Chunk by symbol + date to bound memory
- [ ] Upsert so partial reruns are safe
- [ ] Uses the same `marketpulse.features` module as the consumer

### `dag_data_archival` — daily 04:00 UTC
- [ ] identify partitions past hot retention → export Parquet to object storage → **verify (row count + checksum)** → drop partition → record
- [ ] The verify step is non-negotiable; it's what separates maintenance from data loss

### `dag_partition_maintenance` — monthly
- [ ] Create the next two months of partitions ahead (a missing partition = insert error at midnight on the 1st)

### Cross-cutting
- [ ] Every DAG: `catchup=False`, owner, tags, docstring
- [ ] Every task: idempotent, `retries=2` with exponential backoff, explicit `execution_timeout`, `on_failure_callback`
- [ ] XCom for small metadata only — data passes via Postgres/object storage

## Tests
- [ ] **DAG import test in CI** — parse every DAG file. A broken DAG silently disables scheduling; this is a genuinely nasty failure mode.
- [ ] No cycles; expected task counts and dependency edges
- [ ] All DAGs assert `catchup=False`
- [ ] All tasks assert a non-null `execution_timeout`
- [ ] Quality gate blocks retraining on an injected corrupt window (integration)
- [ ] Backfill is idempotent: run twice → same row count
- [ ] Archival verify step fails closed on a checksum mismatch (drop must not execute)

## Watch out for
- A newly deployed DAG with a past `start_date` and `catchup=True` triggers a stampede of historical runs. Classic first-day incident.
