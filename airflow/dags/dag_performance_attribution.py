"""dag_performance_attribution: score served predictions against reality, hourly.

Runs hourly but is *inherently* lagged by the horizon ``H``: a prediction is
only resolvable once ``feature_ts + H <= now``, so each run picks up whatever
crossed that line since the last one. Nothing here needs to know the lag —
``monitoring.performance`` enforces it in one place, and the DAG just runs
often enough that the backlog stays small.

``catchup=False``: replaying old intervals would re-resolve predictions that
are already resolved (harmless, the writes are idempotent) while producing no
new information, since resolution depends on wall-clock time rather than on
the logical interval.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from _dag_common import DEFAULT_ARGS, logger, mp_session_factory, mp_settings
from airflow.sdk import DAG, task
from sqlalchemy import select

from marketpulse.monitoring.alerts import evaluate_and_record
from marketpulse.monitoring.performance import PerformanceSlice, run_performance_attribution
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import TrainingRun


def _to_slice(row: dict[str, Any]) -> PerformanceSlice:
    return PerformanceSlice(
        model_version=str(row["model_version"]),
        resolved_count=int(row["resolved_count"]),
        accuracy=float(row["accuracy"]),
        macro_f1=float(row["macro_f1"]),
        per_class_f1=dict(row["per_class_f1"]),
        confusion_matrix=dict(row["confusion_matrix"]),
        predicted_distribution=dict(row["predicted_distribution"]),
        window_start=datetime.fromisoformat(str(row["window_start"])).astimezone(UTC),
        window_end=datetime.fromisoformat(str(row["window_end"])).astimezone(UTC),
    )


@task(execution_timeout=timedelta(minutes=30))
def resolve_and_score() -> dict[str, object]:
    settings = mp_settings()
    summary, slices = run_performance_attribution(
        mp_session_factory(settings),
        settings.monitoring,
        # Bounded by the tick pipeline's own gap threshold, not by the
        # horizon: a wider tolerance would let a prediction next to a data
        # gap be scored against a price from far past the horizon it
        # actually claimed anything about.
        price_tolerance=timedelta(seconds=settings.features.gap_threshold_seconds),
    )
    logger.info(
        "performance attribution complete",
        extra={
            "extra_fields": {
                "resolved": summary.resolved,
                "pending": summary.pending,
                "skipped_no_future_price": summary.skipped_no_future_price,
                "model_versions": [s.model_version for s in slices],
            }
        },
    )
    return {
        "slices": [
            {
                "model_version": s.model_version,
                "resolved_count": s.resolved_count,
                "accuracy": s.accuracy,
                "macro_f1": s.macro_f1,
                "per_class_f1": s.per_class_f1,
                "confusion_matrix": s.confusion_matrix,
                "predicted_distribution": s.predicted_distribution,
                "window_start": s.window_start.isoformat(),
                "window_end": s.window_end.isoformat(),
            }
            for s in slices
        ],
        "resolved": summary.resolved,
        "pending": summary.pending,
    }


@task(execution_timeout=timedelta(minutes=5))
def evaluate_alerts(payload: dict[str, object]) -> None:
    """Evaluate accuracy and prediction-distribution rules on this run's slices.

    The training prior comes from the most recent training run's recorded
    class distribution — the same prior the model was fit against. Comparing
    the live predicted mix to anything else would answer a different question.
    """
    settings = mp_settings()
    session_factory = mp_session_factory(settings)

    raw_slices = payload.get("slices") or []
    if not isinstance(raw_slices, list) or not raw_slices:
        logger.warning("no resolved predictions this run; skipping alert evaluation")
        return

    # Rebuilt from the XCom payload rather than passed directly: XCom
    # serialises through JSON, so a dataclass does not survive the hop.
    slices = [_to_slice(row) for row in raw_slices]

    with session_scope(session_factory) as session:
        distribution = session.execute(
            select(TrainingRun.train_class_distribution)
            .order_by(TrainingRun.finished_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    total = sum(distribution.values()) if distribution else 0
    prior = {k: v / total for k, v in distribution.items()} if total else {}

    decisions = evaluate_and_record(
        session_factory,
        settings.monitoring,
        performance_slices=slices,
        training_prior=prior,
    )
    logger.info(
        "performance alerts evaluated",
        extra={"extra_fields": {"decisions": [d.action for d in decisions]}},
    )


with DAG(
    dag_id="dag_performance_attribution",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["monitoring", "performance", "phase-6"],
    doc_md=__doc__,
) as dag:
    evaluate_alerts(resolve_and_score())
