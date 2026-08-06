"""jobs.backfill idempotency (phase-4 plan test list: "Backfill is
idempotent: run twice -> same row count").
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.jobs.backfill import backfill_symbol
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import Feature, RawTick
from marketpulse.storage.partitions import ensure_partitions_covering
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id

pytestmark = pytest.mark.integration

SYMBOL = "BACKFILL-USD"
MAX_HISTORY = timedelta(hours=27)
MAX_BUFFER_POINTS = 10_000
GAP_THRESHOLD = timedelta(seconds=120)


def _seed_ticks(
    session_factory: sessionmaker[Session], *, start: datetime, end: datetime, interval: timedelta
) -> None:
    with session_scope(session_factory) as session:
        symbol_id = get_or_create_symbol_id(session, SYMBOL)
        observed_at = start
        price = Decimal("100")
        while observed_at < end:
            session.add(
                RawTick(
                    observed_at=observed_at,
                    symbol_id=symbol_id,
                    message_id=uuid4(),
                    correlation_id=uuid4(),
                    schema_version=1,
                    emitted_at=observed_at,
                    price=price,
                    volume=Decimal("1"),
                    source="seed",
                )
            )
            observed_at += interval
            price += 1


def _feature_row_count(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as session:
        return session.execute(select(func.count()).select_from(Feature)).scalar_one()


def test_backfill_rerun_produces_the_same_row_count(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    end = start + timedelta(hours=2)
    warm_up_start = (start - timedelta(hours=1)).date()
    with engine.begin() as connection:
        ensure_partitions_covering(connection, "raw_ticks", warm_up_start, months_ahead=1)
        ensure_partitions_covering(connection, "features", warm_up_start, months_ahead=1)
    # Warm-up history before `start` too, so windowed features at the start
    # of the range aren't all insufficient-history nulls.
    _seed_ticks(
        session_factory, start=start - timedelta(hours=1), end=end, interval=timedelta(minutes=1)
    )

    def run_backfill() -> None:
        backfill_symbol(
            session_factory,
            SYMBOL,
            start=start,
            end=end,
            max_history=MAX_HISTORY,
            max_buffer_points=MAX_BUFFER_POINTS,
            gap_threshold=GAP_THRESHOLD,
        )

    run_backfill()
    first_count = _feature_row_count(session_factory)
    assert first_count > 0, "expected the backfill to actually produce feature rows"

    run_backfill()
    second_count = _feature_row_count(session_factory)

    assert second_count == first_count


def test_backfill_rerun_after_partial_failure_still_converges(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    """A backfill that's rerun after only partially completing (e.g. a
    killed task) must converge to the same end state as one continuous run
    -- never duplicate rows for the chunk that already succeeded.
    """
    start = datetime(2025, 7, 1, tzinfo=UTC)
    midpoint = start + timedelta(hours=1)
    end = start + timedelta(hours=2)
    warm_up_start = (start - timedelta(hours=1)).date()
    with engine.begin() as connection:
        ensure_partitions_covering(connection, "raw_ticks", warm_up_start, months_ahead=1)
        ensure_partitions_covering(connection, "features", warm_up_start, months_ahead=1)
    _seed_ticks(
        session_factory, start=start - timedelta(hours=1), end=end, interval=timedelta(minutes=1)
    )

    # Simulate a partial run: only the first half completes.
    backfill_symbol(
        session_factory,
        SYMBOL,
        start=start,
        end=midpoint,
        max_history=MAX_HISTORY,
        max_buffer_points=MAX_BUFFER_POINTS,
        gap_threshold=GAP_THRESHOLD,
    )
    partial_count = _feature_row_count(session_factory)
    assert partial_count > 0

    # Full rerun over the entire original range.
    backfill_symbol(
        session_factory,
        SYMBOL,
        start=start,
        end=end,
        max_history=MAX_HISTORY,
        max_buffer_points=MAX_BUFFER_POINTS,
        gap_threshold=GAP_THRESHOLD,
    )
    full_count = _feature_row_count(session_factory)

    assert full_count > partial_count

    # And a second full rerun changes nothing further.
    backfill_symbol(
        session_factory,
        SYMBOL,
        start=start,
        end=end,
        max_history=MAX_HISTORY,
        max_buffer_points=MAX_BUFFER_POINTS,
        gap_threshold=GAP_THRESHOLD,
    )
    assert _feature_row_count(session_factory) == full_count
