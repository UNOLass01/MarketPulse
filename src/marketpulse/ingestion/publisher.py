"""Publishes tick envelopes with persistent delivery and publisher confirms.

Persistence alone does not guarantee the broker actually saved the message —
only publisher confirms do. Both are required together, or the producer can
believe a message was saved when it wasn't (see phase-1 plan, "watch out for").
"""

import logging
from collections import deque

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPError, NackError, UnroutableError

from marketpulse.contracts.messages import TickEnvelope
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.serialization import CONTENT_TYPE, serialize_envelope
from marketpulse.messaging.topology import EXCHANGE_DATA

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUFFER_SIZE = 10_000


class Publisher:
    """Publishes envelopes to ``market.data`` with persistence + confirms.

    When the broker is unreachable, envelopes queue up in an in-memory
    buffer bounded at ``max_buffer_size``. Once full, the oldest buffered
    envelope is dropped (and logged) to make room — the buffer never grows
    unbounded.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        *,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
    ) -> None:
        self._connection_manager = connection_manager
        self._max_buffer_size = max_buffer_size
        self._buffer: deque[TickEnvelope] = deque()
        self._confirms_channel: BlockingChannel | None = None

    def publish(self, envelope: TickEnvelope) -> None:
        """Queue an envelope for delivery and attempt to flush the buffer."""
        self._buffer.append(envelope)
        self._enforce_buffer_bound()
        self._flush()

    def buffered_count(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        """Best-effort final flush before shutdown. Never blocks indefinitely."""
        self._flush()

    def _enforce_buffer_bound(self) -> None:
        while len(self._buffer) > self._max_buffer_size:
            dropped = self._buffer.popleft()
            logger.warning(
                "publisher buffer full, dropping oldest envelope",
                extra={
                    "extra_fields": {
                        "message_id": str(dropped.message_id),
                        "symbol": dropped.symbol,
                    }
                },
            )

    def _flush(self) -> None:
        try:
            channel = self._connection_manager.channel()
            if channel is not self._confirms_channel:
                channel.confirm_delivery()
                self._confirms_channel = channel
        except Exception:
            logger.warning("broker unavailable, buffering envelopes", exc_info=True)
            return

        while self._buffer:
            envelope = self._buffer[0]
            try:
                channel.basic_publish(
                    exchange=EXCHANGE_DATA,
                    routing_key=f"tick.{envelope.symbol}",
                    body=serialize_envelope(envelope),
                    properties=pika.BasicProperties(
                        content_type=CONTENT_TYPE,
                        delivery_mode=2,
                        message_id=str(envelope.message_id),
                        correlation_id=str(envelope.correlation_id),
                        headers={"x-retry-count": 0},
                    ),
                    mandatory=True,
                )
            except UnroutableError:
                logger.error(
                    "envelope unroutable, dropping",
                    extra={"extra_fields": {"message_id": str(envelope.message_id)}},
                )
                self._buffer.popleft()
                continue
            except (NackError, AMQPError):
                logger.warning("publish not confirmed, will retry from buffer", exc_info=True)
                return
            else:
                self._buffer.popleft()
