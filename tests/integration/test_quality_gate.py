"""Quality gate blocking retraining on an injected corrupt window
(phase-4 plan test list: "Quality gate blocks retraining on an injected
corrupt window (integration)").

Exercises the exact mechanism ``dag_model_retraining``'s ``quality_gate``
task relies on — ``storage.repositories.quality.latest_checks_passed``
reading real, persisted ``quality_checks`` rows — against real Postgres,
not a mock. The gate must fail closed: a missing or stale check counts as
failing, same as an explicit failure.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.quality import (
    ALL_CHECK_NAMES,
    latest_checks_passed,
    record_quality_check,
)

pytestmark = pytest.mark.integration

LOOKBACK = timedelta(hours=2)


def _record_all(session: Session, *, passed: bool) -> None:
    now = datetime.now(UTC)
    for check_name in ALL_CHECK_NAMES:
        record_quality_check(
            session,
            check_name=check_name,
            symbol="BTC-USD",
            passed=passed,
            details={},
            window_start=now - timedelta(hours=1),
            window_end=now,
        )


def test_gate_passes_when_every_check_recently_passed(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _record_all(session, passed=True)

    with session_factory() as session:
        assert latest_checks_passed(session, since=datetime.now(UTC) - LOOKBACK) is True


def test_gate_blocks_on_one_failed_check_in_an_otherwise_clean_window(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _record_all(session, passed=True)
        # Inject one corrupt check -- freshness -- into an otherwise clean run.
        record_quality_check(
            session,
            check_name="freshness",
            symbol="BTC-USD",
            passed=False,
            details={"reason": "injected corrupt window"},
            window_start=datetime.now(UTC) - timedelta(hours=1),
            window_end=datetime.now(UTC),
        )

    with session_factory() as session:
        assert latest_checks_passed(session, since=datetime.now(UTC) - LOOKBACK) is False


def test_gate_reads_the_most_recent_result_per_check_not_an_earlier_failure(
    session_factory: sessionmaker[Session],
) -> None:
    # A failure two runs ago must not still block once the check has since
    # passed -- the gate reads latest-per-check, not "has ever failed".
    with session_factory() as session:
        record_quality_check(
            session,
            check_name="freshness",
            symbol="BTC-USD",
            passed=False,
            details={"reason": "earlier corrupt window, since fixed"},
            window_start=datetime.now(UTC) - timedelta(hours=1, minutes=30),
            window_end=datetime.now(UTC) - timedelta(hours=1),
        )
        session.commit()

    with session_scope(session_factory) as session:
        _record_all(session, passed=True)

    with session_factory() as session:
        assert latest_checks_passed(session, since=datetime.now(UTC) - LOOKBACK) is True


def test_gate_fails_closed_when_a_check_never_ran(session_factory: sessionmaker[Session]) -> None:
    with session_scope(session_factory) as session:
        # Only freshness recorded; completeness/validity/distribution never ran.
        record_quality_check(
            session,
            check_name="freshness",
            symbol="BTC-USD",
            passed=True,
            details={},
            window_start=datetime.now(UTC) - timedelta(hours=1),
            window_end=datetime.now(UTC),
        )

    with session_factory() as session:
        assert latest_checks_passed(session, since=datetime.now(UTC) - LOOKBACK) is False


def test_gate_fails_closed_on_a_stale_check_outside_the_lookback(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _record_all(session, passed=True)

    # A `since` cutoff set ahead of when the rows were actually written
    # models staleness without needing to backdate `checked_at` (a
    # server-side default) -- every row above is, relative to this cutoff,
    # "too old", exactly like a dag_data_quality run that silently stopped
    # firing.
    with session_factory() as session:
        stale_cutoff = datetime.now(UTC) + timedelta(hours=1)
        assert latest_checks_passed(session, since=stale_cutoff) is False
