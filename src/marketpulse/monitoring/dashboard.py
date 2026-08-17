"""Panel data for the Phase 6 dashboard.

The dashboard **computes nothing and writes nothing**. Every number it shows
was precomputed by an Airflow DAG and stored, so the metrics exist whether or
not anyone has the page open — a dashboard that computes on page load is a
dashboard whose history disappears the moment you close the tab.

That constraint is why this module exists at all instead of the queries
living in the Streamlit file: these are read-only aggregations returning
plain dataclasses, so the panels can be asserted on in a unit test and the
Streamlit layer stays a rendering shell. It also means the dashboard runs
happily against read-only database credentials.

Every builder returns an **explicit empty state** rather than raising or
returning ``None`` on no data. A fresh install with nothing ingested yet must
render "no data yet", not a stack trace (phase-6 plan).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import MonitoringSettings, Settings
from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from marketpulse.monitoring.drift import SEVERITY_RANK, SEVERITY_STABLE
from marketpulse.monitoring.performance import PerformanceSlice, compute_performance_slices
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import ModelVersion, QualityCheck
from marketpulse.storage.repositories.alerts import list_open_alerts
from marketpulse.storage.repositories.drift import list_drift_metrics
from marketpulse.storage.repositories.features import (
    latest_feature_ts_per_symbol,
    list_feature_rows_in_range,
)
from marketpulse.storage.repositories.predictions import latest_prediction_at
from marketpulse.storage.repositories.quality import list_checks_since
from marketpulse.storage.repositories.symbols import list_symbols
from marketpulse.storage.repositories.ticks import (
    latest_observed_at_per_symbol,
    list_ticks_in_range,
)

#: Default look-back for every panel. Bounded on purpose: an unbounded
#: dashboard query is the classic way a "read-only" dashboard takes the
#: database down during an incident, exactly when it is most needed.
DEFAULT_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class SymbolFreshness:
    symbol: str
    last_tick_at: datetime | None
    last_feature_ts: datetime | None
    tick_age_seconds: float | None
    feature_age_seconds: float | None
    is_stale: bool


@dataclass(frozen=True)
class SystemHealthPanel:
    symbols: list[SymbolFreshness]
    last_prediction_at: datetime | None
    open_alert_count: int
    critical_alert_count: int
    is_empty: bool
    empty_reason: str | None = None


@dataclass(frozen=True)
class DataPipelinePanel:
    #: (bucket start, tick count) — ingestion rate over the window.
    ingestion_rate: list[tuple[datetime, int]]
    completeness_by_symbol: dict[str, float]
    null_rate_by_feature: dict[str, float]
    quality_history: list[tuple[datetime, str, bool]]
    is_empty: bool
    empty_reason: str | None = None


@dataclass(frozen=True)
class PromotionEvent:
    """A promotion boundary, drawn as a vertical annotation on the accuracy
    chart. A visible accuracy step at one of these is the single most
    persuasive artifact this project produces (phase-6 plan) — which is why
    they are first-class panel data rather than something to eyeball.
    """

    model_version: str
    promoted_at: datetime


@dataclass(frozen=True)
class ModelPerformancePanel:
    slices: list[PerformanceSlice]
    promotions: list[PromotionEvent]
    #: Drawn as a horizontal reference line on the accuracy chart. Without it
    #: a 0.41 accuracy looks bad; against a 3-class majority baseline it may
    #: be the whole result.
    baseline_accuracy: float | None
    pending_count: int
    is_empty: bool
    empty_reason: str | None = None


@dataclass(frozen=True)
class DriftCell:
    feature_name: str
    window_end: datetime
    psi: float
    severity: str


@dataclass(frozen=True)
class DriftPanel:
    #: Feature x time cells for the PSI heatmap.
    heatmap: list[DriftCell]
    ks_by_feature: dict[str, float]
    severity_timeline: list[tuple[datetime, str]]
    worst_severity: str
    is_empty: bool
    empty_reason: str | None = None


@dataclass(frozen=True)
class Dashboard:
    system_health: SystemHealthPanel
    data_pipeline: DataPipelinePanel
    model_performance: ModelPerformancePanel
    drift: DriftPanel
    generated_at: datetime
    window: timedelta = DEFAULT_WINDOW
    warnings: list[str] = field(default_factory=list)


def _age(value: datetime | None, now: datetime) -> float | None:
    return None if value is None else max((now - value).total_seconds(), 0.0)


def build_system_health(
    session: Session, *, now: datetime, stale_after: timedelta
) -> SystemHealthPanel:
    ticks = latest_observed_at_per_symbol(session)
    features = latest_feature_ts_per_symbol(session)
    alerts = list_open_alerts(session)

    codes = sorted(set(ticks) | set(features))
    if not codes:
        return SystemHealthPanel(
            symbols=[],
            last_prediction_at=None,
            open_alert_count=len(alerts),
            critical_alert_count=sum(1 for a in alerts if a.severity == "critical"),
            is_empty=True,
            empty_reason="no symbols have reported a tick or a feature row yet",
        )

    threshold = stale_after.total_seconds()
    rows = []
    for code in codes:
        tick_age = _age(ticks.get(code), now)
        feature_age = _age(features.get(code), now)
        rows.append(
            SymbolFreshness(
                symbol=code,
                last_tick_at=ticks.get(code),
                last_feature_ts=features.get(code),
                tick_age_seconds=tick_age,
                feature_age_seconds=feature_age,
                # A symbol with no feature row at all counts as stale --
                # "never produced" is not a healthier state than "produced
                # and went quiet".
                is_stale=feature_age is None or feature_age > threshold,
            )
        )

    return SystemHealthPanel(
        symbols=rows,
        last_prediction_at=latest_prediction_at(session),
        open_alert_count=len(alerts),
        critical_alert_count=sum(1 for a in alerts if a.severity == "critical"),
        is_empty=False,
    )


def build_data_pipeline(
    session: Session,
    *,
    now: datetime,
    window: timedelta,
    expected_interval: timedelta,
    bucket: timedelta = timedelta(hours=1),
) -> DataPipelinePanel:
    start = now - window
    symbols = list_symbols(session)
    if not symbols:
        return DataPipelinePanel([], {}, {}, [], True, "no symbols exist yet")

    buckets: dict[datetime, int] = {}
    completeness: dict[str, float] = {}
    null_counts = dict.fromkeys(FEATURE_NAMES, 0)
    total_rows = 0

    expected_per_symbol = max(int(window / expected_interval), 1)
    for symbol in symbols:
        ticks = list_ticks_in_range(session, symbol.id, start, now)
        completeness[symbol.code] = min(len(ticks) / expected_per_symbol, 1.0)
        for tick in ticks:
            key = _floor_to(tick.observed_at, bucket)
            buckets[key] = buckets.get(key, 0) + 1

        for row in list_feature_rows_in_range(session, symbol.id, FEATURE_SET_VERSION, start, now):
            total_rows += 1
            for name in FEATURE_NAMES:
                if row.feature_values.get(name) is None:
                    null_counts[name] += 1

    null_rate = (
        {name: count / total_rows for name, count in null_counts.items()} if total_rows else {}
    )
    history = [
        (row.checked_at, row.check_name, row.passed)
        for row in list_checks_since(session, since=start)
    ]

    is_empty = not buckets and total_rows == 0
    return DataPipelinePanel(
        ingestion_rate=sorted(buckets.items()),
        completeness_by_symbol=completeness,
        null_rate_by_feature=null_rate,
        quality_history=history,
        is_empty=is_empty,
        empty_reason="no ticks or feature rows in the selected window" if is_empty else None,
    )


def _floor_to(value: datetime, bucket: timedelta) -> datetime:
    seconds = int(bucket.total_seconds())
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)


def build_model_performance(
    session: Session,
    session_factory: sessionmaker[Session],
    monitoring: MonitoringSettings,
    *,
    now: datetime,
    model_name: str,
    pending_count: int = 0,
    baseline_accuracy: float | None = None,
) -> ModelPerformancePanel:
    slices = compute_performance_slices(session_factory, monitoring, now=now)

    promotions = [
        PromotionEvent(model_version=row.mlflow_model_version, promoted_at=row.promoted_at)
        for row in session.execute(
            select(ModelVersion)
            .where(
                ModelVersion.mlflow_model_name == model_name,
                ModelVersion.promoted_at.is_not(None),
            )
            .order_by(ModelVersion.promoted_at.asc())
        ).scalars()
        if row.promoted_at is not None
    ]

    is_empty = not slices
    return ModelPerformancePanel(
        slices=slices,
        promotions=promotions,
        baseline_accuracy=baseline_accuracy,
        pending_count=pending_count,
        is_empty=is_empty,
        empty_reason=(
            "no predictions have been resolved yet — outcomes only become "
            "available one horizon after a prediction is made"
            if is_empty
            else None
        ),
    )


def build_drift(session: Session, *, now: datetime, window: timedelta) -> DriftPanel:
    rows = list_drift_metrics(session, since=now - window)
    if not rows:
        return DriftPanel([], {}, [], SEVERITY_STABLE, True, "drift monitoring has not run yet")

    heatmap = [
        DriftCell(row.feature_name, row.window_end, row.metric_value, row.severity)
        for row in rows
        if row.metric_name == "psi"
    ]
    ks = {row.feature_name: row.metric_value for row in rows if row.metric_name == "ks"}

    by_window: dict[datetime, str] = {}
    for row in rows:
        current = by_window.get(row.window_end, SEVERITY_STABLE)
        if SEVERITY_RANK[row.severity] > SEVERITY_RANK[current]:
            by_window[row.window_end] = row.severity
        else:
            by_window.setdefault(row.window_end, current)

    timeline = sorted(by_window.items())
    worst = (
        max((s for _, s in timeline), key=lambda s: SEVERITY_RANK[s])
        if timeline
        else SEVERITY_STABLE
    )
    return DriftPanel(
        heatmap=heatmap,
        ks_by_feature=ks,
        severity_timeline=timeline,
        worst_severity=worst,
        is_empty=False,
    )


def build_dashboard(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    now: datetime | None = None,
    window: timedelta = DEFAULT_WINDOW,
) -> Dashboard:
    """Assemble all four panels in one pass.

    Every panel is built inside a single read-only session so the four
    panels describe the same instant rather than four slightly different
    ones — a dashboard whose panels disagree about "now" is how an incident
    gets misdiagnosed.
    """
    now = now or datetime.now(UTC)
    monitoring = settings.monitoring
    mlflow_settings = settings.mlflow
    serving = settings.serving

    warnings: list[str] = []
    with session_scope(session_factory) as session:
        system_health = build_system_health(
            session,
            now=now,
            stale_after=timedelta(seconds=serving.max_feature_age_seconds),
        )
        data_pipeline = build_data_pipeline(
            session,
            now=now,
            window=window,
            expected_interval=timedelta(seconds=monitoring.expected_tick_interval_seconds),
        )
        model_performance = build_model_performance(
            session,
            session_factory,
            monitoring,
            now=now,
            model_name=mlflow_settings.registry_model_name,
        )
        drift = build_drift(session, now=now, window=window)

    if drift.is_empty:
        # Stated as a warning, never rendered as "no drift detected" -- an
        # empty drift table means the monitor has not run, which is a
        # different and more worrying thing than a clean one.
        warnings.append("drift monitoring has produced no metrics in this window")
    if system_health.is_empty:
        warnings.append("no ingestion data yet")

    return Dashboard(
        system_health=system_health,
        data_pipeline=data_pipeline,
        model_performance=model_performance,
        drift=drift,
        generated_at=now,
        window=window,
        warnings=warnings,
    )


def stale_symbols(panel: SystemHealthPanel) -> Sequence[str]:
    return [row.symbol for row in panel.symbols if row.is_stale]


def latest_quality_status(session: Session, *, since: datetime) -> dict[str, bool]:
    """Most recent pass/fail per check name.

    A check with no recent row is simply absent from the result — the caller
    renders "not run", never "passed". Absence of a failure is not a pass
    (the same rule ``latest_checks_passed`` enforces for the retraining gate).
    """
    stmt = (
        select(QualityCheck.check_name, QualityCheck.passed, QualityCheck.checked_at)
        .where(QualityCheck.checked_at >= since)
        .order_by(QualityCheck.checked_at.desc())
    )
    status: dict[str, bool] = {}
    for check_name, passed, _checked_at in session.execute(stmt):
        status.setdefault(check_name, passed)
    return status
