# 0003 — Airflow 3.x, not 2.x

**Status:** Accepted
**Date:** 2026-08-06

## Context

Phase 4 needs an Airflow scheduler + webserver, and `docker/airflow.Dockerfile`
must install the `marketpulse` package into the same image (phase-4 plan:
"base image + the marketpulse package installed") so DAG task callables can
`import marketpulse.ml.pipeline` etc. directly. That means Airflow's own
dependencies and `marketpulse`'s dependencies have to resolve together in
one environment — not just in CI, but in the real deployed image.

`marketpulse` requires `sqlalchemy>=2.0` (the ORM code throughout
`storage/` uses the 2.0-style `Mapped`/`mapped_column`/`DeclarativeBase`
API; there's no 1.4-compatible fallback and adding one just for Airflow
would be its own maintenance burden).

## Options considered

1. **`apache-airflow` 2.x (any version).** `apache-airflow-core` pins
   `sqlalchemy<2.0,>=1.4.49` all the way through the 2.x line — confirmed by
   attempting to install `apache-airflow==2.11.2` (latest 2.x) alongside
   `marketpulse` and hitting `ResolutionImpossible`. Rejected: a hard,
   unresolvable conflict, not a version-bump-away fix.
2. **Airflow 3.0.x–3.2.x.** Same conflict — `apache-airflow-core` didn't
   move to `sqlalchemy>=2.0` until 3.3.0 (checked directly against each
   version's published metadata).
3. **Airflow 3.3.0 (current latest at time of writing).**
   `apache-airflow-core==3.3.0` requires `sqlalchemy[asyncio]>=2.0.48` —
   installs cleanly alongside `marketpulse`'s own `sqlalchemy>=2.0`.
4. **Run Airflow and `marketpulse` in separate environments/containers**,
   with DAGs calling out to a subprocess or an internal API instead of
   importing `marketpulse.*` directly. Rejected: directly contradicts
   CLAUDE.md rule #9 ("DAG files are thin callables importing from
   `marketpulse.*`") and the phase-4 plan's explicit Dockerfile requirement;
   would also reintroduce cross-process complexity this project's small
   architecture deliberately avoids.

## Decision

Pin `apache-airflow==3.3.0`. This is a newer major version than most
existing Airflow tutorials/examples target — DAG authoring in 3.x goes
through the Task SDK (`airflow.sdk`) rather than 2.x's direct
`airflow.models` imports — but it's the only version that lets one
environment satisfy both Airflow's and `marketpulse`'s dependencies
simultaneously, which the phase-4 architecture requires by construction.

## Consequences

- DAG files in `airflow/dags/` use the Airflow 3 Task SDK import surface
  (`from airflow.sdk import DAG, task`), not the Airflow 2 `airflow.models`
  style — don't copy-paste Airflow 2 examples verbatim.
- `pyproject.toml`'s `apache-airflow==3.3.0` pin and
  `docker/airflow.Dockerfile`'s base image tag must be bumped together, same
  discipline as the existing MLflow client/server pin (see
  `docker/mlflow.Dockerfile`'s comment).
- Airflow 3's split architecture (separate API server / DAG processor /
  scheduler) is heavier than 2.x's — LocalExecutor still applies (per the
  phase-4 plan: "not Celery"), but compose resource limits should assume a
  few more small processes than a 2.x deployment would need.
- If a future `sqlalchemy` major bump ever breaks this pairing again, revisit
  here rather than silently pinning around it.
