"""CoinGeckoProvider retry/backoff behaviour, mocked at the requests.Session
boundary — the live API is never contacted in tests.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from marketpulse.exceptions import PermanentError, TransientError
from marketpulse.ingestion.providers.coingecko import CoinGeckoProvider

pytestmark = pytest.mark.unit


@dataclass
class FakeResponse:
    status_code: int
    _json: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self._json


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(
        self, url: str, params: dict[str, str] | None = None, timeout: float | None = None
    ) -> FakeResponse:
        self.calls += 1
        return self._responses.pop(0)


def _price_response() -> FakeResponse:
    return FakeResponse(
        200,
        {"bitcoin": {"usd": 65000.5, "usd_24h_vol": 1000.0, "last_updated_at": 1_700_000_000}},
    )


def test_fetch_parses_observations() -> None:
    session = FakeSession([_price_response()])
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None)

    observations = provider.fetch(["BTC-USD"])

    assert len(observations) == 1
    obs = observations[0]
    assert obs.symbol == "BTC-USD"
    assert obs.price == Decimal("65000.5")
    assert obs.observed_at == datetime.fromtimestamp(1_700_000_000, tz=UTC)
    assert obs.observed_at.tzinfo is UTC


def test_fetch_with_no_known_symbols_skips_request() -> None:
    session = FakeSession([])
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None)

    assert provider.fetch(["ETH-USD"]) == []
    assert session.calls == 0


def test_retry_after_header_is_honoured_on_429() -> None:
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "5"}), _price_response()])
    sleeps: list[float] = []
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=sleeps.append)

    observations = provider.fetch(["BTC-USD"])

    assert sleeps == [5.0]
    assert len(observations) == 1


def test_5xx_backs_off_with_jitter_then_succeeds() -> None:
    session = FakeSession([FakeResponse(503), _price_response()])
    sleeps: list[float] = []
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=sleeps.append)

    provider.fetch(["BTC-USD"])

    assert len(sleeps) == 1
    # attempt 1: base = 2**0 = 1s, jitter adds up to 25% -> [1.0, 1.25)
    assert 1.0 <= sleeps[0] < 1.25


def test_exhausting_retries_raises_transient_error() -> None:
    session = FakeSession([FakeResponse(503) for _ in range(4)])
    provider = CoinGeckoProvider(
        {"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None, max_retries=3
    )

    with pytest.raises(TransientError):
        provider.fetch(["BTC-USD"])


def test_4xx_error_raises_permanent_error() -> None:
    session = FakeSession([FakeResponse(404, text="not found")])
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None)

    with pytest.raises(PermanentError):
        provider.fetch(["BTC-USD"])


def _market_chart_response() -> FakeResponse:
    return FakeResponse(
        200,
        {
            "prices": [[1_700_000_000_000, 65000.5], [1_700_003_600_000, 65100.0]],
            "total_volumes": [[1_700_000_000_000, 1000.0], [1_700_003_600_000, 1100.0]],
        },
    )


def test_fetch_history_parses_price_and_volume_series() -> None:
    session = FakeSession([_market_chart_response()])
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None)

    observations = provider.fetch_history("BTC-USD", days=1)

    assert len(observations) == 2
    first = observations[0]
    assert first.symbol == "BTC-USD"
    assert first.price == Decimal("65000.5")
    assert first.volume == Decimal("1000.0")
    assert first.observed_at == datetime.fromtimestamp(1_700_000_000, tz=UTC)
    assert first.observed_at.tzinfo is UTC


def test_fetch_history_unknown_symbol_skips_request() -> None:
    session = FakeSession([])
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None)

    assert provider.fetch_history("ETH-USD", days=1) == []
    assert session.calls == 0


def test_fetch_history_mismatched_series_lengths_zips_to_shortest() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "prices": [[1_700_000_000_000, 100.0], [1_700_003_600_000, 101.0]],
                    "total_volumes": [[1_700_000_000_000, 10.0]],
                },
            )
        ]
    )
    provider = CoinGeckoProvider({"BTC-USD": "bitcoin"}, session=session, sleep=lambda s: None)

    observations = provider.fetch_history("BTC-USD", days=1)

    assert len(observations) == 1
