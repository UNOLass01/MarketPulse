"""Poller: heartbeat increments every iteration, and a fetch failure never
kills the loop.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from marketpulse.ingestion.poller import Poller
from marketpulse.ingestion.providers.base import Observation

pytestmark = pytest.mark.unit


class FakeProvider:
    def __init__(self, observations: list[Observation]) -> None:
        self._observations = observations
        self.calls = 0

    def fetch(self, symbols: list[str]) -> list[Observation]:
        self.calls += 1
        return self._observations


class FailingProvider:
    def fetch(self, symbols: list[str]) -> list[Observation]:
        raise RuntimeError("provider down")


class RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, envelope: Any) -> None:
        self.published.append(envelope)


def _observation(symbol: str = "BTC-USD") -> Observation:
    return Observation(
        symbol=symbol, price=Decimal("1"), volume=Decimal("1"), observed_at=datetime.now(UTC)
    )


def test_heartbeat_increments_and_publishes_each_observation() -> None:
    provider = FakeProvider([_observation("BTC-USD"), _observation("ETH-USD")])
    publisher = RecordingPublisher()
    sleeps: list[float] = []

    def stop_after_three(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) >= 3:
            poller.stop()

    poller = Poller(provider, publisher, ["BTC-USD", "ETH-USD"], sleep=stop_after_three)  # type: ignore[arg-type]
    poller.run()

    assert poller.heartbeat_count == 3
    assert len(publisher.published) == 6  # 2 observations x 3 iterations


def test_provider_failure_does_not_stop_the_loop() -> None:
    publisher = RecordingPublisher()
    sleeps: list[float] = []

    def stop_after_one(delay: float) -> None:
        sleeps.append(delay)
        poller.stop()

    poller = Poller(FailingProvider(), publisher, ["BTC-USD"], sleep=stop_after_one)  # type: ignore[arg-type]
    poller.run()  # must not raise

    assert poller.heartbeat_count == 1
    assert publisher.published == []
