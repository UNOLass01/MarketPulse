"""Publisher: bounded in-memory buffer sheds the oldest entry when full and
the broker is unreachable — it never grows unbounded.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.ingestion.publisher import Publisher

pytestmark = pytest.mark.unit


class UnreachableConnectionManager:
    def channel(self) -> None:
        raise ConnectionError("broker down")


def _envelope(symbol: str) -> TickEnvelope:
    return TickEnvelope(
        emitted_at=datetime.now(UTC),
        symbol=symbol,
        payload=TickPayload(
            price=Decimal("1"), volume=Decimal("1"), provider_observed_at=datetime.now(UTC)
        ),
    )


def test_publish_buffers_when_broker_unreachable() -> None:
    publisher = Publisher(UnreachableConnectionManager(), max_buffer_size=10)  # type: ignore[arg-type]

    publisher.publish(_envelope("BTC-USD"))

    assert publisher.buffered_count() == 1


def test_buffer_sheds_oldest_when_full() -> None:
    publisher = Publisher(UnreachableConnectionManager(), max_buffer_size=2)  # type: ignore[arg-type]

    first, second, third = _envelope("A"), _envelope("B"), _envelope("C")
    publisher.publish(first)
    publisher.publish(second)
    publisher.publish(third)

    assert publisher.buffered_count() == 2
    remaining = list(publisher._buffer)
    assert first not in remaining
    assert remaining == [second, third]
