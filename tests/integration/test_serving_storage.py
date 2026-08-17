"""Prediction logging and outcome resolution against real Postgres (Phases 5-6).

The horizon-lag rule is a SQL predicate, so it is tested here rather than in
the unit tier — a fake would only prove the fake works. This is the phase-6
plan's "off-by-one here is the likeliest bug in the phase", pinned from both
sides: a prediction exactly ``H`` old resolves, one a second younger does not.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.monitoring.performance import compute_performance_slices, resolve_outcomes
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.predictions import (
    count_pending,
    list_predictions,
    list_resolved,
    price_at,
    price_at_or_after,
    resolvable_predictions,
    upsert_prediction,
)
from marketpulse.storage.repositories.ticks import upsert_tick

pytestmark = pytest.mark.integration

SYMBOL = "BTC-USD"
HORIZON = timedelta(minutes=15)
TOLERANCE = timedelta(seconds=120)
THETA = 0.002

# Anchored to "now" so rows land inside the monthly partitions the `engine`
# fixture creates, while every offset below stays exact.
NOW = datetime.now(UTC).replace(microsecond=0)


def _tick(session: Session, *, observed_at: datetime, price: str) -> None:
    upsert_tick(
        session,
        TickEnvelope(
            emitted_at=observed_at,
            symbol=SYMBOL,
            payload=TickPayload(
                price=Decimal(price),
                volume=Decimal("1"),
                provider_observed_at=observed_at,
            ),
        ),
    )


def _prediction(
    session: Session,
    *,
    feature_ts: datetime,
    label: str,
    model_version: str = "1",
) -> int | None:
    return upsert_prediction(
        session,
        symbol=SYMBOL,
        model_version=model_version,
        feature_set_version=1,
        feature_ts=feature_ts,
        predicted_at=feature_ts,
        label=label,
        probabilities={"DOWN": 0.2, "STABLE": 0.3, "UP": 0.5},
        latency_ms=4.2,
        correlation_id=uuid4(),
    )


# --- prediction logging idempotency --------------------------------------


def test_the_same_prediction_logged_twice_produces_one_row(
    session_factory: sessionmaker[Session],
) -> None:
    # One row per (symbol, feature_ts, model_version), not per HTTP request:
    # otherwise rolling accuracy would be weighted by request volume.
    feature_ts = NOW - timedelta(minutes=30)

    with session_scope(session_factory) as session:
        first = _prediction(session, feature_ts=feature_ts, label="UP")
        second = _prediction(session, feature_ts=feature_ts, label="UP")

    assert first is not None
    assert second is None  # the duplicate collapsed to a no-op

    with session_factory() as session:
        rows, _ = list_predictions(
            session,
            symbol=SYMBOL,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(minutes=1),
            limit=10,
        )
    assert len(rows) == 1


def test_a_different_model_version_is_a_different_prediction(
    session_factory: sessionmaker[Session],
) -> None:
    feature_ts = NOW - timedelta(minutes=30)

    with session_scope(session_factory) as session:
        assert _prediction(session, feature_ts=feature_ts, label="UP", model_version="1")
        assert _prediction(session, feature_ts=feature_ts, label="DOWN", model_version="2")

    with session_factory() as session:
        rows, _ = list_predictions(
            session,
            symbol=SYMBOL,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(minutes=1),
            limit=10,
        )
    assert {row.model_version for row in rows} == {"1", "2"}


# --- the horizon lag ------------------------------------------------------


def test_predictions_younger_than_the_horizon_are_not_resolvable(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _prediction(session, feature_ts=NOW - timedelta(minutes=1), label="UP")
        _prediction(session, feature_ts=NOW - timedelta(minutes=14), label="UP")

    with session_factory() as session:
        assert resolvable_predictions(session, horizon=HORIZON, now=NOW) == []
        assert count_pending(session, horizon=HORIZON, now=NOW) == 2


def test_the_resolvable_boundary_is_exactly_the_horizon(
    session_factory: sessionmaker[Session],
) -> None:
    """The off-by-one, pinned from both sides.

    A prediction at ``t`` claims something about ``t + H``, so it becomes
    resolvable exactly when ``t + H <= now``. One second younger must not.
    """
    at_boundary = NOW - HORIZON
    one_second_younger = NOW - HORIZON + timedelta(seconds=1)

    with session_scope(session_factory) as session:
        _prediction(session, feature_ts=at_boundary, label="UP")
        _prediction(session, feature_ts=one_second_younger, label="DOWN")

    with session_factory() as session:
        resolvable = resolvable_predictions(session, horizon=HORIZON, now=NOW)

    assert [row[2] for row in resolvable] == [at_boundary]


def test_already_resolved_predictions_do_not_come_back(
    session_factory: sessionmaker[Session],
) -> None:
    # Re-running the attribution DAG must be idempotent.
    feature_ts = NOW - timedelta(minutes=30)
    with session_scope(session_factory) as session:
        _tick(session, observed_at=feature_ts, price="100.00")
        _tick(session, observed_at=feature_ts + HORIZON, price="105.00")
        _prediction(session, feature_ts=feature_ts, label="UP")

    first = resolve_outcomes(
        session_factory, horizon=HORIZON, theta=THETA, price_tolerance=TOLERANCE, now=NOW
    )
    second = resolve_outcomes(
        session_factory, horizon=HORIZON, theta=THETA, price_tolerance=TOLERANCE, now=NOW
    )

    assert first.resolved == 1
    assert second.resolved == 0

    with session_factory() as session:
        assert len(list_resolved(session, start=NOW - timedelta(days=1), end=NOW)) == 1


# --- outcome scoring ------------------------------------------------------


@pytest.mark.parametrize(
    ("future_price", "predicted", "expected_actual", "expected_correct"),
    [
        ("105.00", "UP", "UP", True),  # +5% >> theta
        ("95.00", "UP", "DOWN", False),  # -5% << -theta
        ("100.05", "UP", "STABLE", False),  # +0.05% inside the deadband
        ("95.00", "DOWN", "DOWN", True),
    ],
)
def test_outcomes_are_scored_against_the_trained_deadband(
    session_factory: sessionmaker[Session],
    future_price: str,
    predicted: str,
    expected_actual: str,
    expected_correct: bool,
) -> None:
    feature_ts = NOW - timedelta(minutes=30)
    with session_scope(session_factory) as session:
        _tick(session, observed_at=feature_ts, price="100.00")
        _tick(session, observed_at=feature_ts + HORIZON, price=future_price)
        _prediction(session, feature_ts=feature_ts, label=predicted)

    summary = resolve_outcomes(
        session_factory, horizon=HORIZON, theta=THETA, price_tolerance=TOLERANCE, now=NOW
    )

    assert summary.resolved == 1
    with session_factory() as session:
        rows = list_resolved(session, start=NOW - timedelta(days=1), end=NOW)
    _version, _symbol, logged_predicted, actual, _at = rows[0]
    assert logged_predicted == predicted
    assert actual == expected_actual
    assert (logged_predicted == actual) is expected_correct


def test_a_prediction_with_no_price_at_the_horizon_stays_unresolved(
    session_factory: sessionmaker[Session],
) -> None:
    # Widening the search to "whatever came next" would score a prediction
    # against a horizon it never claimed anything about.
    feature_ts = NOW - timedelta(minutes=60)
    with session_scope(session_factory) as session:
        _tick(session, observed_at=feature_ts, price="100.00")
        # The next tick is far past the horizon + tolerance.
        _tick(session, observed_at=feature_ts + timedelta(minutes=50), price="130.00")
        _prediction(session, feature_ts=feature_ts, label="UP")

    summary = resolve_outcomes(
        session_factory, horizon=HORIZON, theta=THETA, price_tolerance=TOLERANCE, now=NOW
    )

    assert summary.resolved == 0
    assert summary.skipped_no_future_price == 1


def test_as_of_lookup_finds_the_earliest_tick_at_or_after_the_target(
    session_factory: sessionmaker[Session],
) -> None:
    target = NOW - timedelta(minutes=10)
    with session_scope(session_factory) as session:
        _tick(session, observed_at=target - timedelta(seconds=30), price="99.00")
        _tick(session, observed_at=target + timedelta(seconds=20), price="101.00")
        _tick(session, observed_at=target + timedelta(seconds=90), price="103.00")

    with session_factory() as session:
        found = price_at_or_after(session, symbol=SYMBOL, target=target, tolerance=TOLERANCE)
        exact = price_at(session, symbol=SYMBOL, observed_at=target + timedelta(seconds=20))

    assert found is not None
    assert found[1] == Decimal("101.00000000")
    assert exact == Decimal("101.00000000")


# --- slicing across a promotion boundary ---------------------------------


def test_accuracy_slicing_by_model_version_is_correct_across_a_promotion(
    session_factory: sessionmaker[Session],
) -> None:
    """Version 1 gets everything wrong, version 2 gets everything right.

    Unsliced these average out and the step disappears — which is exactly
    why the metric is sliced.
    """
    base = NOW - timedelta(hours=3)

    with session_scope(session_factory) as session:
        for index in range(10):
            feature_ts = base + timedelta(minutes=index)
            _tick(session, observed_at=feature_ts, price="100.00")
            _tick(session, observed_at=feature_ts + HORIZON, price="110.00")  # a real UP
            # v1 says DOWN (wrong), v2 says UP (right) -- same features.
            _prediction(session, feature_ts=feature_ts, label="DOWN", model_version="1")
            _prediction(session, feature_ts=feature_ts, label="UP", model_version="2")

    resolve_outcomes(
        session_factory, horizon=HORIZON, theta=THETA, price_tolerance=TOLERANCE, now=NOW
    )

    from marketpulse.config import MonitoringSettings

    slices = compute_performance_slices(
        session_factory, MonitoringSettings(performance_window_hours=24), now=NOW
    )

    by_version = {s.model_version: s for s in slices}
    assert set(by_version) == {"1", "2"}
    assert by_version["1"].accuracy == pytest.approx(0.0)
    assert by_version["2"].accuracy == pytest.approx(1.0)
    assert by_version["1"].resolved_count == 10
    assert by_version["2"].resolved_count == 10


# --- pagination -----------------------------------------------------------


def test_pagination_reports_has_more_without_a_count_query(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        for index in range(5):
            _prediction(session, feature_ts=NOW - timedelta(minutes=index + 20), label="UP")

    start, end = NOW - timedelta(days=1), NOW + timedelta(minutes=1)
    with session_factory() as session:
        page_one, more_one = list_predictions(session, symbol=SYMBOL, start=start, end=end, limit=2)
        page_two, more_two = list_predictions(
            session, symbol=SYMBOL, start=start, end=end, limit=2, offset=2
        )
        last, more_last = list_predictions(
            session, symbol=SYMBOL, start=start, end=end, limit=2, offset=4
        )

    assert (len(page_one), more_one) == (2, True)
    assert (len(page_two), more_two) == (2, True)
    assert (len(last), more_last) == (1, False)
    # Newest first, and no row appears on two pages.
    assert {row.id for row in page_one} & {row.id for row in page_two} == set()


def test_history_can_be_filtered_by_model_version(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _prediction(session, feature_ts=NOW - timedelta(minutes=20), label="UP", model_version="1")
        _prediction(session, feature_ts=NOW - timedelta(minutes=21), label="UP", model_version="2")

    with session_factory() as session:
        rows, _ = list_predictions(
            session,
            symbol=SYMBOL,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(minutes=1),
            model_version="2",
            limit=10,
        )

    assert [row.model_version for row in rows] == ["2"]
