"""``features`` table persistence: idempotent upsert, latest-per-symbol
lookup, and JSONB round-trip of null feature values.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.contracts.features import FeatureVector
from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import Feature
from marketpulse.storage.repositories.features import latest_feature_vector, upsert_feature_vector
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id

pytestmark = pytest.mark.integration


def _vector(feature_ts: datetime, *, symbol: str = "BTC-USD") -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        feature_ts=feature_ts,
        feature_values=dict.fromkeys(FEATURE_NAMES, 1.5),
        insufficient_history=False,
        has_gap=False,
    )


def test_upsert_is_idempotent(session_factory: sessionmaker[Session]) -> None:
    vector = _vector(datetime.now(UTC))
    with session_scope(session_factory) as session:
        upsert_feature_vector(session, vector)
        upsert_feature_vector(session, vector)

    with session_factory() as session:
        count = session.execute(select(func.count()).select_from(Feature)).scalar_one()
    assert count == 1


def test_latest_feature_vector_returns_most_recent(session_factory: sessionmaker[Session]) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        symbol_id = get_or_create_symbol_id(session, "BTC-USD")
        upsert_feature_vector(session, _vector(now - timedelta(minutes=10)))
        upsert_feature_vector(session, _vector(now))

    with session_factory() as session:
        latest = latest_feature_vector(session, symbol_id, feature_set_version=1)

    assert latest is not None
    assert latest.feature_ts == now


def test_latest_feature_vector_is_none_when_no_rows_exist(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        symbol_id = get_or_create_symbol_id(session, "BTC-USD")

    with session_factory() as session:
        latest = latest_feature_vector(session, symbol_id, feature_set_version=1)
    assert latest is None


def test_null_feature_values_round_trip_through_jsonb(
    session_factory: sessionmaker[Session],
) -> None:
    vector = FeatureVector(
        symbol="BTC-USD",
        feature_ts=datetime.now(UTC),
        feature_values=dict.fromkeys(FEATURE_NAMES, None),
        insufficient_history=True,
        has_gap=False,
    )
    with session_scope(session_factory) as session:
        upsert_feature_vector(session, vector)
        symbol_id = get_or_create_symbol_id(session, "BTC-USD")

    with session_factory() as session:
        row = latest_feature_vector(session, symbol_id, feature_set_version=1)

    assert row is not None
    assert set(row.feature_values) == set(FEATURE_NAMES)
    assert all(v is None for v in row.feature_values.values())
    assert row.insufficient_history is True


def test_different_symbols_at_the_same_timestamp_are_independent_rows(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        upsert_feature_vector(session, _vector(now, symbol="BTC-USD"))
        upsert_feature_vector(session, _vector(now, symbol="ETH-USD"))

    with session_factory() as session:
        count = session.execute(select(func.count()).select_from(Feature)).scalar_one()
    assert count == 2
