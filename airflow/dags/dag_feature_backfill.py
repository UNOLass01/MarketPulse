"""dag_feature_backfill: manual-trigger feature recompute over a date range
(Phase 4).

Recomputes and upserts feature rows from *already-ingested* raw ticks —
distinct from re-ingesting data. Trigger this after a
``feature_set_version`` bump or to fix a detected gap. Chunked by symbol +
date to bound memory, and idempotent (``ON CONFLICT DO NOTHING`` — a
partial rerun after a failure upserts the same rows, never duplicates).
All backfill logic lives in ``marketpulse.jobs.backfill`` (CLAUDE.md rule
#9); this file only wires DAG-run params to it.

Trigger with e.g.::

    {"start": "2026-01-01T00:00:00", "end": "2026-02-01T00:00:00",
     "symbols": ["BTC-USD", "ETH-USD"], "feature_set_version": 1}
"""

from datetime import UTC, datetime, timedelta

from _dag_common import DEFAULT_ARGS, logger, mp_session_factory, mp_settings
from airflow.sdk import DAG, Param, task

from marketpulse.features.registry import FEATURE_SET_VERSION
from marketpulse.jobs.backfill import backfill_features


@task(execution_timeout=timedelta(hours=6), retries=1, retry_delay=timedelta(minutes=10))
def run_backfill(params: dict[str, object]) -> list[dict[str, object]]:
    requested_version = params["feature_set_version"]
    if requested_version != FEATURE_SET_VERSION:
        raise ValueError(
            f"requested feature_set_version={requested_version} does not match the "
            f"current registry version={FEATURE_SET_VERSION} — backfilling a "
            "historical feature set version isn't supported by the current "
            "features.pipeline API"
        )

    settings = mp_settings()
    results = backfill_features(
        mp_session_factory(settings),
        settings,
        start=datetime.fromisoformat(str(params["start"])).replace(tzinfo=UTC),
        end=datetime.fromisoformat(str(params["end"])).replace(tzinfo=UTC),
        symbols=list(params["symbols"]),  # type: ignore[arg-type]
    )
    summary = [{"symbol": r.symbol, "rows_upserted": r.rows_upserted} for r in results]
    logger.info("backfill complete", extra={"extra_fields": {"results": summary}})
    return summary


with DAG(
    dag_id="dag_feature_backfill",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["features", "phase-4", "manual"],
    doc_md=__doc__,
    params={
        "start": Param(default="2026-01-01T00:00:00", type="string"),
        "end": Param(default="2026-01-02T00:00:00", type="string"),
        "symbols": Param(default=["BTC-USD", "ETH-USD"], type="array"),
        "feature_set_version": Param(default=FEATURE_SET_VERSION, type="integer"),
    },
) as dag:
    run_backfill()
