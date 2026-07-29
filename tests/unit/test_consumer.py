"""BaseConsumer error classification: retry-count header increments and caps,
schema violations dead-letter immediately, unhandled errors nack without
requeue. Uses a fake channel double — no broker needed.
"""

from types import SimpleNamespace
from typing import Any

import pika
import pytest

from marketpulse.exceptions import PermanentError, TransientError
from marketpulse.messaging.consumer import (
    DEAD_REASON_HEADER,
    REASON_EXHAUSTED,
    REASON_VALIDATION,
    RETRY_COUNT_HEADER,
    BaseConsumer,
)
from marketpulse.messaging.topology import EXCHANGE_DLX, EXCHANGE_RETRY

pytestmark = pytest.mark.unit


class FakeChannel:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.acked: list[int] = []
        self.nacked: list[tuple[int, bool]] = []

    def basic_publish(
        self, exchange: str, routing_key: str, body: bytes, properties: pika.BasicProperties
    ) -> None:
        self.published.append(
            {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
            }
        )

    def basic_ack(self, delivery_tag: int) -> None:
        self.acked.append(delivery_tag)

    def basic_nack(self, delivery_tag: int, requeue: bool) -> None:
        self.nacked.append((delivery_tag, requeue))


def _method(routing_key: str = "tick.BTC-USD", delivery_tag: int = 1) -> Any:
    return SimpleNamespace(routing_key=routing_key, delivery_tag=delivery_tag)


def _properties(retry_count: int | None = None) -> pika.BasicProperties:
    headers = {RETRY_COUNT_HEADER: retry_count} if retry_count is not None else {}
    return pika.BasicProperties(headers=headers)


def _consumer(deserialize: Any, process: Any) -> BaseConsumer[Any]:
    return BaseConsumer(
        connection_manager=object(),  # type: ignore[arg-type]
        queue_name="q.test",
        deserialize=deserialize,
        process=process,
    )


def test_success_acks() -> None:
    channel = FakeChannel()
    consumer = _consumer(deserialize=lambda b: b, process=lambda m: None)

    consumer._on_message(channel, _method(delivery_tag=7), _properties(), b"body")  # type: ignore[arg-type]

    assert channel.acked == [7]
    assert channel.published == []


def test_schema_violation_dead_letters_and_acks() -> None:
    def deserialize(body: bytes) -> Any:
        raise PermanentError("bad schema")

    channel = FakeChannel()
    consumer = _consumer(deserialize=deserialize, process=lambda m: None)

    consumer._on_message(channel, _method(delivery_tag=1), _properties(), b"body")  # type: ignore[arg-type]

    assert channel.acked == [1]
    assert len(channel.published) == 1
    published = channel.published[0]
    assert published["exchange"] == EXCHANGE_DLX
    assert published["routing_key"] == REASON_VALIDATION
    assert published["properties"].headers[DEAD_REASON_HEADER] == REASON_VALIDATION


def test_transient_error_below_cap_requeues_with_incremented_header() -> None:
    def process(m: Any) -> None:
        raise TransientError("db down")

    channel = FakeChannel()
    consumer = _consumer(deserialize=lambda b: b, process=process)

    consumer._on_message(channel, _method(delivery_tag=1), _properties(retry_count=1), b"body")  # type: ignore[arg-type]

    assert channel.acked == [1]
    published = channel.published[0]
    assert published["exchange"] == EXCHANGE_RETRY
    assert published["routing_key"] == "tick.BTC-USD"
    assert published["properties"].headers[RETRY_COUNT_HEADER] == 2


def test_transient_error_at_cap_dead_letters() -> None:
    def process(m: Any) -> None:
        raise TransientError("db down")

    channel = FakeChannel()
    consumer = _consumer(deserialize=lambda b: b, process=process)

    consumer._on_message(channel, _method(delivery_tag=1), _properties(retry_count=3), b"body")  # type: ignore[arg-type]

    assert channel.acked == [1]
    published = channel.published[0]
    assert published["exchange"] == EXCHANGE_DLX
    assert published["routing_key"] == REASON_EXHAUSTED


def test_unhandled_exception_nacks_without_requeue() -> None:
    def process(m: Any) -> None:
        raise RuntimeError("boom")

    channel = FakeChannel()
    consumer = _consumer(deserialize=lambda b: b, process=process)

    consumer._on_message(channel, _method(delivery_tag=9), _properties(), b"body")  # type: ignore[arg-type]

    assert channel.nacked == [(9, False)]
    assert channel.acked == []
    assert channel.published == []


def test_retry_count_caps_at_three_before_dead_lettering() -> None:
    """Header increments 0->1->2->3, then the 4th delivery (retry_count=3) dead-letters."""

    def process(m: Any) -> None:
        raise TransientError("db down")

    channel = FakeChannel()
    consumer = _consumer(deserialize=lambda b: b, process=process)

    for retry_count in (0, 1, 2):
        channel.published.clear()
        consumer._on_message(  # type: ignore[arg-type]
            channel, _method(delivery_tag=1), _properties(retry_count=retry_count), b"body"
        )
        assert channel.published[0]["exchange"] == EXCHANGE_RETRY
        assert channel.published[0]["properties"].headers[RETRY_COUNT_HEADER] == retry_count + 1

    channel.published.clear()
    consumer._on_message(channel, _method(delivery_tag=1), _properties(retry_count=3), b"body")  # type: ignore[arg-type]
    assert channel.published[0]["exchange"] == EXCHANGE_DLX
    assert channel.published[0]["routing_key"] == REASON_EXHAUSTED
