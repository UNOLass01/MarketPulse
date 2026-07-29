"""Interval poll loop: fetch observations from a provider, publish each as a tick."""

import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.ingestion.providers.base import MarketDataProvider, Observation
from marketpulse.ingestion.publisher import Publisher

logger = logging.getLogger(__name__)


class Poller:
    """Fetches on an interval (with jitter) and publishes each observation.

    A single fetch/publish failure never kills the loop — it's logged and
    the next iteration proceeds on schedule.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        publisher: Publisher,
        symbols: list[str],
        *,
        interval_seconds: float = 10.0,
        jitter_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider = provider
        self._publisher = publisher
        self._symbols = symbols
        self._interval_seconds = interval_seconds
        self._jitter_seconds = jitter_seconds
        self._sleep = sleep
        self._stop_event = threading.Event()
        self.heartbeat_count = 0

    def stop(self) -> None:
        """Signal the loop to exit after the current iteration completes."""
        self._stop_event.set()

    def run(self) -> None:
        """Blocking loop; returns once ``stop()`` has been called."""
        while not self._stop_event.is_set():
            self._run_once()
            self.heartbeat_count += 1
            logger.info(
                "poller heartbeat", extra={"extra_fields": {"heartbeat": self.heartbeat_count}}
            )
            if self._stop_event.is_set():
                break
            self._sleep(self._interval_seconds + random.uniform(0, self._jitter_seconds))

    def _run_once(self) -> None:
        try:
            observations = self._provider.fetch(self._symbols)
        except Exception:
            logger.exception("provider fetch failed")
            return

        for observation in observations:
            envelope = self._to_envelope(observation)
            try:
                self._publisher.publish(envelope)
            except Exception:
                logger.exception(
                    "publish failed",
                    extra={"extra_fields": {"message_id": str(envelope.message_id)}},
                )

    @staticmethod
    def _to_envelope(observation: Observation) -> TickEnvelope:
        return TickEnvelope(
            emitted_at=datetime.now(UTC),
            symbol=observation.symbol,
            payload=TickPayload(
                price=observation.price,
                volume=observation.volume,
                provider_observed_at=observation.observed_at,
            ),
        )
