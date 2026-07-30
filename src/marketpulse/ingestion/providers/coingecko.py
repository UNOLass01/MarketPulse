"""CoinGecko implementation of :class:`MarketDataProvider`.

Timestamps are normalised to UTC here, at the provider boundary, per
CLAUDE.md rule: never downstream. Tests must mock ``requests`` (or the
provider itself) — never hit the live API.
"""

import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import requests

from marketpulse.exceptions import PermanentError, TransientError
from marketpulse.ingestion.providers.base import Observation

DEFAULT_BASE_URL = "https://api.coingecko.com/api/v3"
_MAX_BACKOFF_SECONDS = 30.0


class CoinGeckoProvider:
    """Fetches spot prices for a fixed set of symbols via CoinGecko's REST API."""

    def __init__(
        self,
        symbol_to_coin_id: dict[str, str],
        *,
        vs_currency: str = "usd",
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 10.0,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._symbol_to_coin_id = symbol_to_coin_id
        self._coin_id_to_symbol = {v: k for k, v in symbol_to_coin_id.items()}
        self._vs_currency = vs_currency
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep

    def fetch(self, symbols: list[str]) -> list[Observation]:
        coin_ids = [self._symbol_to_coin_id[s] for s in symbols if s in self._symbol_to_coin_id]
        if not coin_ids:
            return []

        data = self._get_with_retry(
            "/simple/price",
            params={
                "ids": ",".join(coin_ids),
                "vs_currencies": self._vs_currency,
                "include_last_updated_at": "true",
                "include_24hr_vol": "true",
            },
        )
        return self._parse_observations(data)

    def fetch_history(self, symbol: str, *, days: int) -> list[Observation]:
        """Historical price+volume series for ``symbol`` over the trailing ``days``.

        Used only by the offline backfill path (``scripts/seed_historical.py``)
        — the live poller always calls ``fetch``. CoinGecko's ``market_chart``
        endpoint returns ``prices`` and ``total_volumes`` as separate
        ``[timestamp_ms, value]`` series sharing the same timestamps and
        length for a given request, so they're zipped by position.
        """
        coin_id = self._symbol_to_coin_id.get(symbol)
        if coin_id is None:
            return []

        data = self._get_with_retry(
            f"/coins/{coin_id}/market_chart",
            params={"vs_currency": self._vs_currency, "days": str(days)},
        )
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        observations = []
        for (timestamp_ms, price), (_, volume) in zip(prices, volumes, strict=False):
            observations.append(
                Observation(
                    symbol=symbol,
                    price=Decimal(str(price)),
                    volume=Decimal(str(volume)),
                    observed_at=datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
                )
            )
        return observations

    def _parse_observations(self, data: dict[str, Any]) -> list[Observation]:
        observations: list[Observation] = []
        volume_key = f"{self._vs_currency}_24h_vol"
        for coin_id, values in data.items():
            symbol = self._coin_id_to_symbol.get(coin_id)
            if symbol is None or self._vs_currency not in values:
                continue
            last_updated_at = values.get("last_updated_at")
            observed_at = (
                datetime.fromtimestamp(last_updated_at, tz=UTC)
                if last_updated_at is not None
                else datetime.now(UTC)
            )
            observations.append(
                Observation(
                    symbol=symbol,
                    price=Decimal(str(values[self._vs_currency])),
                    volume=Decimal(str(values.get(volume_key, 0))),
                    observed_at=observed_at,
                )
            )
        return observations

    def _get_with_retry(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._session.get(
                    f"{self._base_url}{path}", params=params, timeout=self._timeout
                )
            except requests.RequestException as exc:
                self._retry_or_raise(attempt, exc=exc)
                continue

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._backoff_delay(attempt)
                self._retry_or_raise(attempt, delay=delay)
                continue

            if 500 <= response.status_code < 600:
                self._retry_or_raise(attempt)
                continue

            if response.status_code >= 400:
                raise PermanentError(
                    f"CoinGecko request failed: {response.status_code} {response.text}"
                )

            result: dict[str, Any] = response.json()
            return result

    def _retry_or_raise(
        self, attempt: int, *, delay: float | None = None, exc: Exception | None = None
    ) -> None:
        if attempt > self._max_retries:
            raise TransientError(
                f"CoinGecko unavailable after {self._max_retries} retries"
            ) from exc
        self._sleep(delay if delay is not None else self._backoff_delay(attempt))

    def _backoff_delay(self, attempt: int) -> float:
        base = min(float(2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
        return base + random.uniform(0, base * 0.25)
