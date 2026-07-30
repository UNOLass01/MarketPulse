"""End-to-end pipeline behaviour against real Postgres + RabbitMQ.

Covers the phase-1 test list: idempotency, error classification (schema
violation -> DLQ, transient -> retry queue), the retry queue's real TTL/DLX
round-trip, and a full producer -> broker -> consumer -> Postgres row.
"""

import time
from datetime import UTC, datetime
from decimal import Decimal

import pika
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.exceptions import TransientError
from marketpulse.ingestion.publisher import Publisher
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.consumer import (
    DEAD_REASON_HEADER,
    REASON_VALIDATION,
    RETRY_COUNT_HEADER,
    BaseConsumer,
)
from marketpulse.messaging.serialization import deserialize_envelope
from marketpulse.messaging.topology import (
    EXCHANGE_DATA,
    EXCHANGE_RETRY,
    QUEUE_DEAD,
    QUEUE_PERSIST,
    QUEUE_RETRY,
)
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import RawTick
from marketpulse.storage.repositories.ticks import upsert_tick

pytestmark = pytest.mark.integration


def _envelope(symbol: str = "BTC-USD") -> TickEnvelope:
    return TickEnvelope(
        emitted_at=datetime.now(UTC),
        symbol=symbol,
        payload=TickPayload(
            price=Decimal("100.50"), volume=Decimal("2.5"), provider_observed_at=datetime.now(UTC)
        ),
    )


def _process(session_factory: sessionmaker[Session]):
    def process(envelope: TickEnvelope) -> None:
        try:
            with session_scope(session_factory) as session:
                upsert_tick(session, envelope)
        except DBAPIError as exc:
            raise TransientError("database unavailable") from exc

    return process


def _consumer(
    connection_manager: ConnectionManager, session_factory: sessionmaker[Session]
) -> BaseConsumer[TickEnvelope]:
    return BaseConsumer(
        connection_manager, QUEUE_PERSIST, deserialize_envelope, _process(session_factory)
    )


def test_idempotency_publish_same_envelope_twice_yields_one_row(
    connection_manager: ConnectionManager, session_factory: sessionmaker[Session]
) -> None:
    publisher = Publisher(connection_manager)
    envelope = _envelope()
    publisher.publish(envelope)
    publisher.publish(envelope)

    channel = connection_manager.channel()
    consumer = _consumer(connection_manager, session_factory)
    for _ in range(2):
        method, properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=False)
        assert method is not None
        consumer._on_message(channel, method, properties, body)

    with session_factory() as session:
        row_count = session.execute(select(func.count()).select_from(RawTick)).scalar_one()
    assert row_count == 1


def test_same_symbol_and_observed_at_with_different_message_id_still_collapses_to_one_row(
    connection_manager: ConnectionManager, session_factory: sessionmaker[Session]
) -> None:
    """A provider can legitimately report the same ``observed_at`` twice
    across separate poll cycles — a distinct ``message_id`` each time, but
    the same ``(symbol_id, observed_at)``. That must no-op via the second
    unique constraint, not raise an IntegrityError.
    """
    observed_at = datetime.now(UTC)

    def _envelope_at(observed_at: datetime, price: str) -> TickEnvelope:
        return TickEnvelope(
            emitted_at=datetime.now(UTC),
            symbol="BTC-USD",
            payload=TickPayload(
                price=Decimal(price), volume=Decimal("3.0"), provider_observed_at=observed_at
            ),
        )

    publisher = Publisher(connection_manager)
    publisher.publish(_envelope_at(observed_at, "100.50"))
    publisher.publish(_envelope_at(observed_at, "101.00"))

    channel = connection_manager.channel()
    consumer = _consumer(connection_manager, session_factory)
    for _ in range(2):
        method, properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=False)
        assert method is not None
        consumer._on_message(channel, method, properties, body)

    assert channel.basic_get(QUEUE_DEAD, auto_ack=True)[0] is None, (
        "the second message must not dead-letter; a symbol/observed_at collision is a "
        "legitimate no-op, not an error"
    )
    with session_factory() as session:
        row_count = session.execute(select(func.count()).select_from(RawTick)).scalar_one()
    assert row_count == 1


def test_producer_to_consumer_to_postgres_row_appears(
    connection_manager: ConnectionManager, session_factory: sessionmaker[Session]
) -> None:
    publisher = Publisher(connection_manager)
    envelope = _envelope("ETH-USD")
    publisher.publish(envelope)

    channel = connection_manager.channel()
    consumer = _consumer(connection_manager, session_factory)
    method, properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=False)
    assert method is not None
    consumer._on_message(channel, method, properties, body)

    with session_factory() as session:
        row = session.execute(
            select(RawTick).where(RawTick.message_id == envelope.message_id)
        ).scalar_one()
    assert row.price == envelope.payload.price
    assert row.volume == envelope.payload.volume


def test_malformed_message_lands_in_dead_letter_queue(
    connection_manager: ConnectionManager, session_factory: sessionmaker[Session]
) -> None:
    channel = connection_manager.channel()
    channel.confirm_delivery()
    channel.basic_publish(
        exchange=EXCHANGE_DATA,
        routing_key="tick.BTC-USD",
        body=b"not valid json",
        properties=pika.BasicProperties(delivery_mode=2),
    )

    consumer = _consumer(connection_manager, session_factory)
    method, properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=False)
    assert method is not None
    consumer._on_message(channel, method, properties, body)

    dead_method, dead_properties, _dead_body = channel.basic_get(QUEUE_DEAD, auto_ack=True)
    assert dead_method is not None
    assert dead_properties.headers[DEAD_REASON_HEADER] == REASON_VALIDATION


def test_simulated_db_outage_routes_to_retry_queue(
    connection_manager: ConnectionManager,
) -> None:
    publisher = Publisher(connection_manager)
    publisher.publish(_envelope())

    def always_fails(envelope: TickEnvelope) -> None:
        raise TransientError("simulated db outage")

    channel = connection_manager.channel()
    consumer = BaseConsumer(connection_manager, QUEUE_PERSIST, deserialize_envelope, always_fails)
    method, properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=False)
    assert method is not None
    consumer._on_message(channel, method, properties, body)

    retry_method, retry_properties, _retry_body = channel.basic_get(QUEUE_RETRY, auto_ack=True)
    assert retry_method is not None
    assert retry_properties.headers[RETRY_COUNT_HEADER] == 1


def test_retry_queue_ttl_dead_letters_back_to_persist_queue(
    connection_manager: ConnectionManager,
) -> None:
    """Validates the real RabbitMQ TTL + DLX wiring, not consumer logic."""
    channel = connection_manager.channel()
    channel.confirm_delivery()
    probe_body = b'{"probe": "ttl-roundtrip"}'
    channel.basic_publish(
        exchange=EXCHANGE_RETRY,
        routing_key="tick.BTC-USD",
        body=probe_body,
        properties=pika.BasicProperties(delivery_mode=2, headers={RETRY_COUNT_HEADER: 1}),
    )

    deadline = time.monotonic() + 40
    method = None
    body = b""
    while time.monotonic() < deadline:
        method, _properties, body = channel.basic_get(QUEUE_PERSIST, auto_ack=True)
        if method is not None:
            break
        time.sleep(1)

    assert method is not None, "message did not dead-letter back to q.ticks.persist in time"
    assert body == probe_body
