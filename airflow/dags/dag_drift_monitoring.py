"""dag_drift_monitoring: PSI/KS drift against the Production model, every 6h.

The comparison is live-window vs. the reference snapshot stored on the
Production model version at promotion time — not vs. yesterday's live data,
which would measure change rather than drift-from-training. All of that logic
lives in ``marketpulse.monitoring.drift`` and the alert decision in
``marketpulse.monitoring.alerts``; this file only schedules them
(CLAUDE.md rule #9).

Six-hourly rather than hourly on purpose: PSI over a window shorter than the
feature set's widest lookback (24h) is dominated by sampling noise, and a
noisy drift signal evaluated often is how alert fatigue starts.
"""

from datetime import datetime, timedelta
from typing import Any

from _dag_common import DEFAULT_ARGS, logger, mp_session_factory, mp_settings
from airflow.sdk import DAG, task

from marketpulse.monitoring.alerts import evaluate_and_record
from marketpulse.monitoring.drift import DriftResult, run_drift_monitoring, worst_severity


def _to_result(row: dict[str, Any]) -> DriftResult:
    return DriftResult(
        feature_name=str(row["feature_name"]),
        metric_name=str(row["metric_name"]),
        metric_value=float(row["metric_value"]),
        p_value=None if row["p_value"] is None else float(row["p_value"]),
        severity=str(row["severity"]),
        sample_size=int(row["sample_size"]),
    )


@task(execution_timeout=timedelta(minutes=20))
def compute_drift() -> list[dict[str, object]]:
    settings = mp_settings()
    results = run_drift_monitoring(
        mp_session_factory(settings),
        settings.monitoring,
        settings.mlflow.registry_model_name,
    )
    logger.info(
        "drift computed",
        extra={
            "extra_fields": {
                "metric_rows": len(results),
                "worst_severity": worst_severity(results),
            }
        },
    )
    return [
        {
            "feature_name": r.feature_name,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "p_value": r.p_value,
            "severity": r.severity,
            "sample_size": r.sample_size,
        }
        for r in results
    ]


@task(execution_timeout=timedelta(minutes=5))
def evaluate_alerts(rows: list[dict[str, object]]) -> None:
    """Feed this run's drift results through the alert rules.

    Reconstructs ``DriftResult`` objects from the XCom payload rather than
    passing them directly: XCom serialises through JSON, so a dataclass does
    not survive the hop intact.
    """
    settings = mp_settings()
    results = [_to_result(row) for row in rows]
    if not results:
        logger.warning("no drift results to evaluate; skipping alert evaluation")
        return

    decisions = evaluate_and_record(
        mp_session_factory(settings), settings.monitoring, drift_results=results
    )
    logger.info(
        "drift alerts evaluated",
        extra={"extra_fields": {"decisions": [d.action for d in decisions]}},
    )


with DAG(
    dag_id="dag_drift_monitoring",
    schedule="0 */6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["monitoring", "drift", "phase-6"],
    doc_md=__doc__,
) as dag:
    evaluate_alerts(compute_drift())
