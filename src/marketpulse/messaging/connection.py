"""RabbitMQ connection/channel lifecycle with reconnect-on-demand backoff."""

import random
import time
from collections.abc import Callable

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from marketpulse.exceptions import ConnectionUnavailableError
from marketpulse.messaging.topology import declare_topology

_MAX_BACKOFF_SECONDS = 30.0


class ConnectionManager:
    """Owns a single blocking connection/channel pair, reconnecting lazily.

    Topology is (re)declared once per physical connection — right after
    ``_connect()`` — rather than by every caller on every use, since it's an
    idempotent no-op only until the connection drops and a fresh channel
    needs it again.

    Not thread-safe — matches ``pika.BlockingConnection``, which is itself
    single-threaded. Each producer/consumer process owns its own instance.
    """

    def __init__(
        self,
        url: str,
        *,
        max_connect_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = url
        self._max_connect_attempts = max_connect_attempts
        self._sleep = sleep
        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None

    def channel(self) -> BlockingChannel:
        """Return a live channel, transparently reconnecting if needed."""
        if not self._is_usable():
            self._connect()
        assert self._channel is not None
        return self._channel

    def close(self) -> None:
        if self._connection is not None and self._connection.is_open:
            self._connection.close()
        self._connection = None
        self._channel = None

    def _is_usable(self) -> bool:
        return (
            self._connection is not None
            and self._connection.is_open
            and self._channel is not None
            and self._channel.is_open
        )

    def _connect(self) -> None:
        attempt = 0
        while True:
            attempt += 1
            try:
                self._connection = pika.BlockingConnection(pika.URLParameters(self._url))
                self._channel = self._connection.channel()
                declare_topology(self._channel)
                return
            except (AMQPConnectionError, OSError) as exc:
                if attempt >= self._max_connect_attempts:
                    raise ConnectionUnavailableError(
                        f"could not connect to RabbitMQ after {attempt} attempts"
                    ) from exc
                self._sleep(self._backoff_delay(attempt))

    def _backoff_delay(self, attempt: int) -> float:
        base = min(float(2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
        return base + random.uniform(0, base * 0.25)
