"""Data-quality checks for ``dag_data_quality`` (Phase 4).

Four checks per symbol, each producing one persisted :class:`CheckResult`:
freshness (newest tick within the expected interval — "the highest-value
task in the whole portfolio", phase-4 plan, since it catches a silently-dead
producer), completeness (actual vs. expected tick count in a window),
validity (nulls where there shouldn't be any, non-positive prices,
non-monotonic timestamps), and distribution sanity (a feature's mean moving
an implausible amount). The check functions themselves are pure — given
already-fetched data, no I/O — so they're testable without a database;
:func:`run_data_quality_checks` is the only thing here that touches
Postgres, fetching via ``storage.repositories`` and persisting via
``storage.repositories.quality``.

Distribution sanity here is deliberately simple (relative shift in a raw
mean), not PSI-based drift detection — that's Phase 6's job, comparing live
traffic against a *promoted model's* reference distribution
(``model_versions.reference_feature_stats``). This only asks "does anything
look implausible right now", independent of any model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import MonitoringSettings
from marketpulse.features.registry import FEATURE_SET_VERSION
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import Feature, RawTick
from marketpulse.storage.repositories.features import list_feature_rows_in_range
from marketpulse.storage.repositories.quality import record_quality_check
from marketpulse.storage.repositories.symbols import list_symbols
from marketpulse.storage.repositories.ticks import latest_observed_at, list_ticks_in_range


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    symbol: str | None
    passed: bool
    details: dict[str, object]
    window_start: datetime
    window_end: datetime


def check_freshness(
    latest_tick_at: datetime | None,
    *,
    now: datetime,
    max_lag: timedelta,
    symbol: str,
) -> CheckResult:
    """Fails if the newest tick is older than ``max_lag``, or if there is no
    tick at all — a producer that has never reported is exactly as bad as
    one that just went silent.
    """
    window_start = now - max_lag
    if latest_tick_at is None:
        return CheckResult(
            "freshness", symbol, False, {"reason": "no_ticks_observed"}, window_start, now
        )
    lag_seconds = (now - latest_tick_at).total_seconds()
    passed = latest_tick_at >= window_start
    return CheckResult(
        "freshness",
        symbol,
        passed,
        {
            "latest_tick_at": latest_tick_at.isoformat(),
            "lag_seconds": lag_seconds,
            "max_lag_seconds": max_lag.total_seconds(),
        },
        window_start,
        now,
    )


def check_completeness(
    observed_ats: Sequence[datetime],
    *,
    window_start: datetime,
    window_end: datetime,
    expected_interval: timedelta,
    min_ratio: float,
    symbol: str,
) -> CheckResult:
    """Fails if actual/expected tick count in the window drops below
    ``min_ratio`` — catches partial outages a pure freshness check would
    miss (the producer is alive, just dropping most ticks).
    """
    expected = max(int((window_end - window_start) / expected_interval), 1)
    actual = len(observed_ats)
    ratio = actual / expected
    passed = ratio >= min_ratio
    return CheckResult(
        "completeness",
        symbol,
        passed,
        {"expected": expected, "actual": actual, "ratio": ratio, "min_ratio": min_ratio},
        window_start,
        window_end,
    )


def check_validity(
    ticks: Sequence[RawTick],
    features: Sequence[Feature],
    *,
    window_start: datetime,
    window_end: datetime,
    symbol: str,
) -> CheckResult:
    """Fails on any of: a non-positive price, a non-monotonic
    ``observed_at`` sequence, or a feature row claiming sufficient history
    (``insufficient_history=False``) while still holding a null value —
    that combination should be structurally impossible and means the
    pipeline itself is broken, not just the data.
    """
    issues: list[str] = []

    non_positive = sum(1 for tick in ticks if tick.price <= 0)
    if non_positive:
        issues.append(f"{non_positive} tick(s) with non-positive price")

    observed_ats = [tick.observed_at for tick in ticks]
    if any(a > b for a, b in zip(observed_ats, observed_ats[1:], strict=False)):
        issues.append("non-monotonic observed_at sequence")

    unexpected_nulls = sum(
        1
        for row in features
        if not row.insufficient_history and any(v is None for v in row.feature_values.values())
    )
    if unexpected_nulls:
        issues.append(
            f"{unexpected_nulls} feature row(s) with null values despite sufficient history"
        )

    return CheckResult("validity", symbol, not issues, {"issues": issues}, window_start, window_end)


def _feature_means(rows: Sequence[Feature]) -> dict[str, float]:
    """Per-feature-name mean over rows with sufficient history, ignoring
    individual nulls within an otherwise-valid row. Empty input (or a
    feature with no non-null observations) simply contributes no key —
    never a fabricated 0.0 (CLAUDE.md rule #7's spirit applies to
    monitoring aggregates too, not just stored feature rows).
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        if row.insufficient_history:
            continue
        for name, value in row.feature_values.items():
            if value is None:
                continue
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def check_distribution(
    current_means: Mapping[str, float],
    reference_means: Mapping[str, float],
    *,
    max_relative_shift: float,
    window_start: datetime,
    window_end: datetime,
    symbol: str,
) -> CheckResult:
    """Fails if any feature present in both windows has moved by more than
    ``max_relative_shift`` relative to its reference mean. A feature with no
    reference observations (new symbol, cold start) or a reference mean of
    exactly zero is skipped, not treated as an infinite shift.
    """
    shifted: dict[str, float] = {}
    for name, reference in reference_means.items():
        current = current_means.get(name)
        if current is None or reference == 0:
            continue
        relative_shift = abs(current - reference) / abs(reference)
        if relative_shift > max_relative_shift:
            shifted[name] = relative_shift

    return CheckResult(
        "distribution",
        symbol,
        not shifted,
        {"shifted_features": shifted, "max_relative_shift": max_relative_shift},
        window_start,
        window_end,
    )


def run_data_quality_checks(
    session_factory: sessionmaker[Session],
    monitoring: MonitoringSettings,
    *,
    now: datetime | None = None,
) -> list[CheckResult]:
    """Run all four checks for every known symbol and persist each result.

    One committed transaction for the whole run — a partial write (some
    symbols checked, others not, on a mid-run failure) would let
    ``latest_checks_passed`` see a stale pass from before this run instead
    of correctly seeing nothing recent for the missing symbols.
    """
    now = now or datetime.now(UTC)
    max_lag = timedelta(minutes=monitoring.freshness_max_lag_minutes)
    expected_interval = timedelta(seconds=monitoring.expected_tick_interval_seconds)
    completeness_window = timedelta(hours=monitoring.completeness_window_hours)
    distribution_window = timedelta(hours=monitoring.distribution_window_hours)
    reference_window = timedelta(hours=monitoring.distribution_reference_window_hours)

    results: list[CheckResult] = []
    with session_scope(session_factory) as session:
        for symbol in list_symbols(session):
            results.append(
                check_freshness(
                    latest_observed_at(session, symbol.id),
                    now=now,
                    max_lag=max_lag,
                    symbol=symbol.code,
                )
            )

            completeness_start = now - completeness_window
            ticks = list_ticks_in_range(session, symbol.id, completeness_start, now)
            results.append(
                check_completeness(
                    [tick.observed_at for tick in ticks],
                    window_start=completeness_start,
                    window_end=now,
                    expected_interval=expected_interval,
                    min_ratio=monitoring.completeness_min_ratio,
                    symbol=symbol.code,
                )
            )

            distribution_start = now - distribution_window
            current_rows = list_feature_rows_in_range(
                session, symbol.id, FEATURE_SET_VERSION, distribution_start, now
            )
            results.append(
                check_validity(
                    ticks,
                    current_rows,
                    window_start=distribution_start,
                    window_end=now,
                    symbol=symbol.code,
                )
            )

            reference_start = distribution_start - reference_window
            reference_rows = list_feature_rows_in_range(
                session, symbol.id, FEATURE_SET_VERSION, reference_start, distribution_start
            )
            results.append(
                check_distribution(
                    _feature_means(current_rows),
                    _feature_means(reference_rows),
                    max_relative_shift=monitoring.distribution_max_relative_shift,
                    window_start=distribution_start,
                    window_end=now,
                    symbol=symbol.code,
                )
            )

        for result in results:
            record_quality_check(
                session,
                check_name=result.check_name,
                symbol=result.symbol,
                passed=result.passed,
                details=result.details,
                window_start=result.window_start,
                window_end=result.window_end,
            )

    return results
