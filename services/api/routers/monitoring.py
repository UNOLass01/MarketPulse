"""Monitoring reads.

Every number served here was **precomputed** by an Airflow DAG and stored.
Nothing on this router computes a statistic, and that is the same rule the
dashboard follows: metrics must exist whether or not anyone is looking at
them, so that a scrape or a page load is never the thing that produces them.

The one shape decision worth reading twice: "no breaches" and "never
evaluated" are different answers and are rendered differently.
``computed_at is None`` means drift has never run; an empty ``metrics`` list
with a real ``computed_at`` means it ran and found nothing. Collapsing those
two into one empty response would let a monitoring pipeline that silently
died read as a clean bill of health.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from marketpulse.contracts.api import (
    AlertItem,
    DriftMetricItem,
    DriftResponse,
    PerformanceResponse,
    PerformanceSliceItem,
    PipelineResponse,
    QualityCheckItem,
    QualityResponse,
)
from marketpulse.ml.config import load_training_config
from marketpulse.monitoring.drift import SEVERITY_RANK, SEVERITY_STABLE
from marketpulse.monitoring.performance import compute_performance_slices
from marketpulse.storage.models import Alert, ModelVersion, TrainingRun
from marketpulse.storage.repositories.alerts import list_open_alerts
from marketpulse.storage.repositories.drift import (
    latest_drift_computed_at,
    list_drift_metrics,
)
from marketpulse.storage.repositories.predictions import count_pending
from marketpulse.storage.repositories.quality import latest_checked_at, list_checks_since
from services.api.state import AppState, get_session, get_state

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _to_alert_item(row: Alert) -> AlertItem:
    return AlertItem(
        alert_name=row.alert_name,
        severity=row.severity,
        status=row.status,
        runbook=row.runbook,
        details=row.details,
        consecutive_breaches=row.consecutive_breaches,
        first_breached_at=row.first_breached_at,
        fired_at=row.fired_at,
        resolved_at=row.resolved_at,
    )


@router.get("/drift", response_model=DriftResponse)
def drift(
    session: Session = Depends(get_session),
    hours: float = Query(default=24.0, gt=0),
) -> DriftResponse:
    since = datetime.now(UTC) - timedelta(hours=hours)
    rows = list_drift_metrics(session, since=since)
    computed_at = latest_drift_computed_at(session)

    severities = [row.severity for row in rows]
    worst = (
        max(severities, key=lambda s: SEVERITY_RANK.get(s, 0)) if severities else SEVERITY_STABLE
    )

    return DriftResponse(
        metrics=[
            DriftMetricItem(
                feature_name=row.feature_name,
                metric_name=row.metric_name,
                metric_value=row.metric_value,
                p_value=row.p_value,
                severity=row.severity,
                reference_model_version=row.reference_model_version,
                window_start=row.window_start,
                window_end=row.window_end,
                computed_at=row.computed_at,
            )
            for row in rows
        ],
        worst_severity=worst,
        computed_at=computed_at,
    )


@router.get("/performance", response_model=PerformanceResponse)
def performance(
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
) -> PerformanceResponse:
    """Rolling accuracy sliced by model version, plus the pending count.

    ``compute_performance_slices`` reads already-resolved outcome rows and
    aggregates them — it does not resolve anything, which stays the
    attribution DAG's job. Serving an unsliced number would smear a
    promotion boundary into a weighted average of two models.
    """
    horizon = load_training_config().labeling.horizon
    now = datetime.now(UTC)
    slices = compute_performance_slices(state.session_factory, state.settings.monitoring, now=now)

    return PerformanceResponse(
        slices=[
            PerformanceSliceItem(
                model_version=s.model_version,
                resolved_count=s.resolved_count,
                accuracy=s.accuracy,
                macro_f1=s.macro_f1,
                per_class_f1=s.per_class_f1,
                confusion_matrix=s.confusion_matrix,
                predicted_distribution=s.predicted_distribution,
                window_start=s.window_start,
                window_end=s.window_end,
            )
            for s in slices
        ],
        pending_count=count_pending(session, horizon=horizon, now=now),
        horizon_minutes=horizon.total_seconds() / 60.0,
    )


@router.get("/quality", response_model=QualityResponse)
def quality(
    session: Session = Depends(get_session),
    hours: float = Query(default=24.0, gt=0),
) -> QualityResponse:
    rows = list_checks_since(session, since=datetime.now(UTC) - timedelta(hours=hours))
    return QualityResponse(
        checks=[
            QualityCheckItem(
                check_name=row.check_name,
                symbol=row.symbol,
                passed=row.passed,
                details=row.details,
                window_start=row.window_start,
                window_end=row.window_end,
                checked_at=row.checked_at,
            )
            for row in rows
        ],
        # ``all(...)`` over an empty list is True, which would report "all
        # passed" for a system where checks have never run. Require at least
        # one result before claiming anything passed.
        all_passed=bool(rows) and all(row.passed for row in rows),
    )


@router.get("/pipeline", response_model=PipelineResponse)
def pipeline(
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
) -> PipelineResponse:
    model_name = state.settings.mlflow.registry_model_name

    last_run = session.execute(
        select(TrainingRun.finished_at).order_by(TrainingRun.finished_at.desc()).limit(1)
    ).scalar_one_or_none()
    last_promotion = session.execute(
        select(ModelVersion.promoted_at)
        .where(
            ModelVersion.mlflow_model_name == model_name,
            ModelVersion.promoted_at.is_not(None),
        )
        .order_by(ModelVersion.promoted_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    loaded = state.model_cache.current
    return PipelineResponse(
        model_version=loaded.version if loaded else None,
        last_training_run_at=last_run,
        last_promotion_at=last_promotion,
        last_quality_check_at=latest_checked_at(session),
        last_drift_check_at=latest_drift_computed_at(session),
        open_alerts=[_to_alert_item(row) for row in list_open_alerts(session)],
    )
