"""Provider-facing interface. All providers normalise to UTC at this boundary."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, Field, field_validator


class Observation(BaseModel):
    """A single symbol observation as returned by a provider."""

    symbol: str = Field(min_length=1)
    price: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def _tz_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(UTC)


class MarketDataProvider(Protocol):
    """Anything that can fetch current observations for a set of symbols."""

    def fetch(self, symbols: list[str]) -> list[Observation]:
        """Return the latest observation for each available symbol.

        Symbols the provider has no data for are simply omitted — callers
        must not assume the result covers every requested symbol.
        """
        ...
