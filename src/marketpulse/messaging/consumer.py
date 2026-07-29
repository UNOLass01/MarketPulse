"""Generic base consumer: QoS prefetch, ack/nack, retry-count header, error classification.

Classification (CLAUDE.md rule #10 — retry only what can succeed):

- deserialize raises ``PermanentError`` -> dead-letter immediately (``dead.validation``)
- ``process`` raises ``TransientError``  -> retry queue, capped at ``max_retries``,
  then dead-letter (``dead.exhausted``)
- ``process`` raises anything else       -> ``nack(requeue=False)``; the persist
  queue's own dead-letter-exchange is the safety net for anything unclassified
- success                                -> ack (an idempotent ``process`` makes
  redelivered duplicates a silent no-op, no special-casing needed here)
"""

import logging
from collections.abc import Callable
from typing import Generic, TypeVar

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from marketpulse.exceptions import PermanentError, TransientError
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.topology import EXCHANGE_DLX, EXCHANGE_RETRY

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRY_COUNT_HEADER = "x-retry-count"
DEAD_REASON_HEADER = "x-dead-reason"
DEFAULT_MAX_RETRIES = 3
REASON_VALIDATION = "dead.validation"
REASON_EXHAUSTED = "dead.exhausted"


class BaseConsumer(Generic[T]):
    """Consumes from ``queue_name``, deserializing with ``deserialize`` and handing
    each message to ``process``. Neither callable needs to know about RabbitMQ.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        queue_name: str,
        deserialize: Callable[[bytes], T],
        process: Callable[[T], None],
        *,
        prefetch_count: int = 10,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._connection_manager = connection_manager
        self._queue_name = queue_name
        self._deserialize = deserialize
        self._process = process
        self._prefetch_count = prefetch_count
        self._max_retries = max_retries
        self._channel: BlockingChannel | None = None

    def start(self) -> None:
        """Blocking consume loop; returns once ``stop()`` is called or the
        connection is closed."""
        channel = self._connection_manager.channel()
        channel.basic_qos(prefetch_count=self._prefetch_count)
        channel.basic_consume(queue=self._queue_name, on_message_callback=self._on_message)
        self._channel = channel
        channel.start_consuming()

    def stop(self) -> None:
        if self._channel is not None and self._channel.is_open:
            self._channel.stop_consuming()

    def _on_message(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        try:
            message = self._deserialize(body)
        except PermanentError:
            logger.warning("schema violation, dead-lettering", exc_info=True)
            self._dead_letter(channel, body, properties, reason=REASON_VALIDATION)
            channel.basic_ack(method.delivery_tag)
            return

        try:
            self._process(message)
        except TransientError:
            self._handle_transient(channel, method.routing_key, body, properties)
            channel.basic_ack(method.delivery_tag)
        except Exception:
            logger.exception("unhandled error processing message, nacking to safety-net DLX")
            channel.basic_nack(method.delivery_tag, requeue=False)
        else:
            channel.basic_ack(method.delivery_tag)

    def _handle_transient(
        self, channel: BlockingChannel, routing_key: str, body: bytes, properties: BasicProperties
    ) -> None:
        retry_count = int((properties.headers or {}).get(RETRY_COUNT_HEADER, 0))
        if retry_count < self._max_retries:
            next_count = retry_count + 1
            logger.warning(
                "transient error, requeuing for retry",
                extra={"extra_fields": {"retry_count": next_count}},
            )
            channel.basic_publish(
                exchange=EXCHANGE_RETRY,
                routing_key=routing_key,
                body=body,
                properties=self._with_headers(properties, {RETRY_COUNT_HEADER: next_count}),
            )
        else:
            logger.error(
                "retry cap reached, dead-lettering",
                extra={"extra_fields": {"retry_count": retry_count}},
            )
            self._dead_letter(channel, body, properties, reason=REASON_EXHAUSTED)

    def _dead_letter(
        self, channel: BlockingChannel, body: bytes, properties: BasicProperties, *, reason: str
    ) -> None:
        channel.basic_publish(
            exchange=EXCHANGE_DLX,
            routing_key=reason,
            body=body,
            properties=self._with_headers(properties, {DEAD_REASON_HEADER: reason}),
        )

    @staticmethod
    def _with_headers(properties: BasicProperties, extra: dict[str, object]) -> BasicProperties:
        headers = dict(properties.headers or {})
        headers.update(extra)
        return BasicProperties(
            content_type=properties.content_type,
            delivery_mode=2,
            message_id=properties.message_id,
            correlation_id=properties.correlation_id,
            headers=headers,
        )
