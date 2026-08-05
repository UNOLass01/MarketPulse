"""``fetch_feature_rows``: joins ``features`` to the ``raw_ticks`` price of
the same tick, filtered by ``feature_set_version`` and optionally symbol.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.contracts.features import FeatureVector
from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.ticks import upsert_tick
from marketpulse.storage.repositories.training_data import fetch_feature_rows

pytestmark = pytest.mark.integration


def _seed(session: Session, symbol: str, feature_ts: datetime, price: float) -> None:
    envelope = TickEnvelope(
        emitted_at=feature_ts,
        symbol=symbol,
        payload=TickPayload(
            price=Decimal(str(price)), volume=Decimal("10"), provider_observed_at=feature_ts
        ),
    )
    upsert_tick(session, envelope)
    upsert_feature_vector(
        session,
        FeatureVector(
            symbol=symbol,
            feature_ts=feature_ts,
            feature_values=dict.fromkeys(FEATURE_NAMES, 1.0),
            insufficient_history=False,
            has_gap=False,
        ),
    )


def test_fetch_feature_rows_joins_price_from_the_same_tick(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        _seed(session, "BTC-USD", now - timedelta(minutes=1), 100.0)
        _seed(session, "BTC-USD", now, 101.0)

    with session_factory() as session:
        rows = fetch_feature_rows(session, feature_set_version=1)

    assert len(rows) == 2
    assert rows[0].feature_ts < rows[1].feature_ts  # ascending
    assert rows[0].price == pytest.approx(100.0)
    assert rows[1].price == pytest.approx(101.0)
    assert set(rows[0].feature_values) == set(FEATURE_NAMES)


def test_fetch_feature_rows_filters_by_symbol(session_factory: sessionmaker[Session]) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        _seed(session, "BTC-USD", now, 100.0)
        _seed(session, "ETH-USD", now, 50.0)

    with session_factory() as session:
        btc_only = fetch_feature_rows(session, feature_set_version=1, symbols=["BTC-USD"])

    assert len(btc_only) == 1
    assert btc_only[0].symbol == "BTC-USD"


def test_fetch_feature_rows_filters_by_feature_set_version(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        _seed(session, "BTC-USD", now, 100.0)

    with session_factory() as session:
        rows_v1 = fetch_feature_rows(session, feature_set_version=1)
        rows_v2 = fetch_feature_rows(session, feature_set_version=2)

    assert len(rows_v1) == 1
    assert len(rows_v2) == 0


def test_fetch_feature_rows_empty_when_no_data(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rows = fetch_feature_rows(session, feature_set_version=1)
    assert rows == []
