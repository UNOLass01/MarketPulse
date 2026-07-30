"""Phase-2 exit criterion: features computed online (streaming consumer)
must be byte-identical to features computed offline (batch replay of the
same raw rows). Also covers restart warm-up — the other integration-tier
item on the phase-2 test list.

The "online" path here mirrors ``services/consumer/main.py``'s ``process``
closure exactly, driven through the real ``BaseConsumer`` against a real
broker + Postgres, the same way ``tests/integration/test_pipeline.py``
exercises phase-1 behaviour.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.features.pipeline import compute_feature_vector
from marketpulse.features.windows import Observation, WindowStore
from marketpulse.ingestion.publisher import Publisher
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.consumer import BaseConsumer
from marketpulse.messaging.serialization import deserialize_envelope
from marketpulse.messaging.topology import QUEUE_PERSIST
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import Feature
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id, list_symbols
from marketpulse.storage.repositories.ticks import list_recent_ticks, upsert_tick

pytestmark = pytest.mark.integration

SYMBOL = "BTC-USD"
GAP_THRESHOLD = timedelta(minutes=5)
MAX_HISTORY = timedelta(hours=24)
MAX_BUFFER_POINTS = 10_000


def _synthetic_ticks(
    count: int, *, start: datetime, step_seconds: float = 30.0
) -> list[TickEnvelope]:
    envelopes = []
    for i in range(count):
        observed_at = start + timedelta(seconds=i * step_seconds)
        price = Decimal(str(round(100.0 + 5.0 * ((i % 7) - 3) + 0.01 * i, 4)))
        volume = Decimal(str(10.0 + (i % 4)))
        envelopes.append(
            TickEnvelope(
                emitted_at=observed_at,
                symbol=SYMBOL,
                payload=TickPayload(price=price, volume=volume, provider_observed_at=observed_at),
            )
        )
    return envelopes


def _online_process(
    session_factory: sessionmaker[Session], window_store: WindowStore
) -> Callable[[TickEnvelope], None]:
    """Mirrors services/consumer/main.py's `process` closure exactly."""

    def process(envelope: TickEnvelope) -> None:
        with session_scope(session_factory) as session:
            inserted = upsert_tick(session, envelope)
            if not inserted:
                return
            observed_at = envelope.payload.provider_observed_at
            window_store.push(
                envelope.symbol,
                Observation(
                    observed_at=observed_at,
                    price=float(envelope.payload.price),
                    volume=float(envelope.payload.volume),
                ),
            )
            vector = compute_feature_vector(
                envelope.symbol,
                window_store.snapshot(envelope.symbol),
                as_of=observed_at,
                gap_threshold=GAP_THRESHOLD,
            )
            upsert_feature_vector(session, vector)

    return process


def test_online_and_offline_feature_vectors_are_byte_identical(
    connection_manager: ConnectionManager, session_factory: sessionmaker[Session]
) -> None:
    start = datetime.now(UTC) - timedelta(hours=2, minutes=30)
    ticks = _synthetic_ticks(300, start=start, step_seconds=30.0)  # 2.5h of history

    # --- online: publish through the real broker, consume through the real
    # BaseConsumer + a process closure identical to the production wiring.
    publisher = Publisher(connection_manager)
    for envelope in ticks:
        publisher.publish(envelope)

    online_window = WindowStore(max_history=MAX_HISTORY, max_points=MAX_BUFFER_POINTS)
    consumer = BaseConsumer(
        connection_manager,
        QUEUE_PERSIST,
        deserialize_envelope,
        _online_process(session_factory, online_window),
    )
    channel = connection_manager.channel()
    for _ in range(len(ticks)):
        method, properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=False)
        assert method is not None
        consumer._on_message(channel, method, properties, body)

    with session_scope(session_factory) as session:
        symbol_id = get_or_create_symbol_id(session, SYMBOL)
        online_rows = {
            row.feature_ts: (row.feature_values, row.insufficient_history, row.has_gap)
            for row in session.execute(
                select(Feature).where(Feature.symbol_id == symbol_id)
            ).scalars()
        }

    # --- offline: replay the same raw rows, read back from Postgres (not
    # from the original in-memory `ticks` list), through a *fresh*
    # WindowStore — pure in-process batch recomputation, no broker involved.
    with session_factory() as session:
        raw_rows = list_recent_ticks(session, symbol_id, start - timedelta(minutes=1))

    assert len(raw_rows) == len(ticks)

    offline_window = WindowStore(max_history=MAX_HISTORY, max_points=MAX_BUFFER_POINTS)
    offline_vectors = {}
    for tick in raw_rows:
        offline_window.push(
            SYMBOL,
            Observation(
                observed_at=tick.observed_at, price=float(tick.price), volume=float(tick.volume)
            ),
        )
        vector = compute_feature_vector(
            SYMBOL,
            offline_window.snapshot(SYMBOL),
            as_of=tick.observed_at,
            gap_threshold=GAP_THRESHOLD,
        )
        offline_vectors[tick.observed_at] = (
            vector.feature_values,
            vector.insufficient_history,
            vector.has_gap,
        )

    assert len(online_rows) == len(ticks)
    assert set(online_rows) == set(offline_vectors)
    for observed_at, offline_result in offline_vectors.items():
        assert online_rows[observed_at] == offline_result, f"parity mismatch at {observed_at}"


def test_restart_warm_up_prevents_a_nan_gap_after_a_crash(
    session_factory: sessionmaker[Session],
) -> None:
    """Simulates a consumer restart: raw ticks already exist in Postgres
    from before the crash (2h of them — enough to satisfy every feature
    whose lookback is <= 60m, but not the 24h-lookback ones). A fresh
    `WindowStore` rebuilt via the same warm-up query
    `services.consumer.main._warm_up` uses must let the very next tick's
    short-window features compute immediately, instead of falling back to
    insufficient-history purely because the process restarted.
    """
    start = datetime.now(UTC) - timedelta(hours=2)
    pre_crash_ticks = _synthetic_ticks(240, start=start, step_seconds=30.0)  # 2h of history

    with session_scope(session_factory) as session:
        for envelope in pre_crash_ticks:
            upsert_tick(session, envelope)

    next_observed_at = pre_crash_ticks[-1].payload.provider_observed_at + timedelta(seconds=30)

    # --- without warm-up: a bare restart with an empty WindowStore is the
    # gap this test proves warm-up prevents.
    cold_window = WindowStore(max_history=MAX_HISTORY, max_points=MAX_BUFFER_POINTS)
    cold_window.push(SYMBOL, Observation(observed_at=next_observed_at, price=123.0, volume=5.0))
    cold_vector = compute_feature_vector(
        SYMBOL, cold_window.snapshot(SYMBOL), as_of=next_observed_at, gap_threshold=GAP_THRESHOLD
    )
    assert cold_vector.feature_values["ma_60m"] is None

    # --- with warm-up: rebuild window state from Postgres exactly as
    # services.consumer.main._warm_up does, then push the same new tick.
    warm_window = WindowStore(max_history=MAX_HISTORY, max_points=MAX_BUFFER_POINTS)
    since = datetime.now(UTC) - MAX_HISTORY
    with session_scope(session_factory) as session:
        for symbol in list_symbols(session):
            for tick in list_recent_ticks(session, symbol.id, since):
                warm_window.push(
                    symbol.code,
                    Observation(
                        observed_at=tick.observed_at,
                        price=float(tick.price),
                        volume=float(tick.volume),
                    ),
                )
    warm_window.push(SYMBOL, Observation(observed_at=next_observed_at, price=123.0, volume=5.0))
    warm_vector = compute_feature_vector(
        SYMBOL, warm_window.snapshot(SYMBOL), as_of=next_observed_at, gap_threshold=GAP_THRESHOLD
    )

    # Every feature whose lookback fits inside the warmed-up 2h of history
    # must be populated -- no NaN gap purely because the process restarted.
    for name in (
        "ma_5m",
        "ma_15m",
        "ma_60m",
        "roc_1m",
        "roc_5m",
        "roc_15m",
        "rsi_15m",
        "volatility_15m",
    ):
        assert warm_vector.feature_values[name] is not None, name
    assert warm_vector.has_gap is False

    # The 24h-lookback features genuinely can't be trusted yet with only 2h
    # of real history -- that's correct behaviour, not a gap.
    assert warm_vector.feature_values["realised_vol_24h"] is None
