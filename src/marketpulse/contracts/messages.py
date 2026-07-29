"""Message envelope contracts exchanged over RabbitMQ.

``schema_version`` is bumped whenever the payload shape changes in a
backward-incompatible way; consumers can branch on it during deserialization.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

CURRENT_SCHEMA_VERSION = 1


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class TickPayload(BaseModel):
    """Market observation carried inside a :class:`TickEnvelope`."""

    model_config = ConfigDict(frozen=True)

    price: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    provider_observed_at: datetime

    @field_validator("provider_observed_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_utc(value)


class TickEnvelope(BaseModel):
    """The unit of work published to and consumed from RabbitMQ."""

    model_config = ConfigDict(frozen=True)

    message_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID = Field(default_factory=uuid4)
    schema_version: int = CURRENT_SCHEMA_VERSION
    emitted_at: datetime
    symbol: str = Field(min_length=1)
    payload: TickPayload

    @field_validator("emitted_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_utc(value)
