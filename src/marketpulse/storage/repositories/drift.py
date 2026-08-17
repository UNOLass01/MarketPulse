"""Persistence and reads for ``drift_metrics`` (Phase 6).

Writes are upserts on ``(feature, metric, window, reference_model_version)``
so re-running ``dag_drift_monitoring`` over a window it already covered
corrects that window's rows rather than duplicating them — the same
idempotency property every other DAG in this project has.
"""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.storage.models import DriftMetric


def record_drift_metric(
    session: Session,
    *,
    feature_name: str,
    metric_name: str,
    metric_value: float,
    p_value: float | None,
    severity: str,
    reference_model_version: str,
    sample_size: int,
    window_start: datetime,
    window_end: datetime,
    computed_at: datetime,
) -> None:
    stmt = pg_insert(DriftMetric).values(
        feature_name=feature_name,
        metric_name=metric_name,
        metric_value=metric_value,
        p_value=p_value,
        severity=severity,
        reference_model_version=reference_model_version,
        sample_size=sample_size,
        window_start=window_start,
        window_end=window_end,
        computed_at=computed_at,
    )
    session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_drift_metrics_feature_metric_window",
            set_={
                "metric_value": stmt.excluded.metric_value,
                "p_value": stmt.excluded.p_value,
                "severity": stmt.excluded.severity,
                "sample_size": stmt.excluded.sample_size,
                "computed_at": stmt.excluded.computed_at,
            },
        )
    )


def list_drift_metrics(
    session: Session,
    *,
    since: datetime,
    feature_name: str | None = None,
    metric_name: str | None = None,
) -> list[DriftMetric]:
    """Drift rows computed at or after ``since``, newest window first."""
    stmt = (
        select(DriftMetric)
        .where(DriftMetric.computed_at >= since)
        .order_by(DriftMetric.window_end.desc(), DriftMetric.feature_name.asc())
    )
    if feature_name is not None:
        stmt = stmt.where(DriftMetric.feature_name == feature_name)
    if metric_name is not None:
        stmt = stmt.where(DriftMetric.metric_name == metric_name)
    return list(session.execute(stmt).scalars())


def latest_drift_computed_at(session: Session) -> datetime | None:
    """When drift last ran at all.

    The API reports this separately from the metric list so "no breaches"
    and "never evaluated" render differently — an empty result set must
    never be presented as a clean bill of health.
    """
    stmt = select(DriftMetric.computed_at).order_by(DriftMetric.computed_at.desc()).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def latest_window_metrics(session: Session, *, metric_name: str = "psi") -> Sequence[DriftMetric]:
    """Every feature's metric for the most recently computed window."""
    newest = select(DriftMetric.window_end).order_by(DriftMetric.window_end.desc()).limit(1)
    window_end = session.execute(newest).scalar_one_or_none()
    if window_end is None:
        return []
    stmt = (
        select(DriftMetric)
        .where(DriftMetric.window_end == window_end, DriftMetric.metric_name == metric_name)
        .order_by(DriftMetric.feature_name.asc())
    )
    return list(session.execute(stmt).scalars())
