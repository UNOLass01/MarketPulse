"""dag_model_retraining: daily unattended retrain + promotion gate (Phase 4).

Gated on the latest ``dag_data_quality`` checks all passing within the last
:data:`QUALITY_GATE_LOOKBACK` — a corrupted data window must block
training, not get silently trained on. The gate fails closed: a missing or
stale check counts as failing (see
``marketpulse.storage.repositories.quality.latest_checks_passed``).

The training pipeline itself — extract -> train -> evaluate -> promotion
gate -> (promote + archive incumbent + snapshot reference | hold in
Staging) -> persist — is one call into
``marketpulse.ml.pipeline.run_training_pipeline``, which already commits
atomically and is idempotent (a rerun always creates a new training run and
model version, never mutates a prior one). Splitting that into more Airflow
tasks would mean passing a trained booster or a full feature DataFrame
through XCom, which the phase-4 plan explicitly rules out ("XCom for small
metadata only — data passes via Postgres/object storage"). All business
logic lives in ``marketpulse.ml`` (CLAUDE.md rule #9).
"""

from datetime import UTC, datetime, timedelta

from _dag_common import DEFAULT_ARGS, logger, mp_session_factory, mp_settings
from airflow.sdk import DAG, task

from marketpulse.ml.pipeline import run_training_pipeline
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.quality import latest_checks_passed

#: Must cover dag_data_quality's hourly cadence with margin for a delayed
#: run, or every retraining run would spuriously find "no recent check".
QUALITY_GATE_LOOKBACK = timedelta(hours=2)


@task.short_circuit(execution_timeout=timedelta(minutes=1))
def quality_gate() -> bool:
    settings = mp_settings()
    with session_scope(mp_session_factory(settings)) as session:
        passed = latest_checks_passed(session, since=datetime.now(UTC) - QUALITY_GATE_LOOKBACK)
    if not passed:
        logger.error("retraining blocked: data quality gate failed or stale")
    return passed


@task(
    execution_timeout=timedelta(hours=2),
    retries=1,
    retry_delay=timedelta(minutes=15),
)
def train_and_evaluate() -> dict[str, object]:
    settings = mp_settings()
    result = run_training_pipeline(mp_session_factory(settings), settings)
    return {
        "training_run_id": result.training_run_id,
        "mlflow_model_version": result.mlflow_model_version,
        "promoted": result.promoted,
        "rejection_reason": result.rejection_reason,
    }


@task(execution_timeout=timedelta(minutes=1))
def notify(result: dict[str, object]) -> None:
    if result["promoted"]:
        logger.info("model promoted", extra={"extra_fields": result})
    else:
        logger.warning("model held in staging", extra={"extra_fields": result})


with DAG(
    dag_id="dag_model_retraining",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "phase-4"],
    doc_md=__doc__,
) as dag:
    gate = quality_gate()
    training_result = train_and_evaluate()
    gate >> training_result
    notify(training_result)
