"""Feature vector contract: the output of the feature pipeline, and the row
shape persisted to and read back from the ``features`` table.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marketpulse.features.registry import FEATURE_SET_VERSION


class FeatureVector(BaseModel):
    """One symbol's computed features as of ``feature_ts``.

    ``feature_values`` is keyed by name, not positionally ordered — callers
    that need a stable column order (training, serving) must go through
    ``features.registry.ordered_values``, never rely on dict iteration order.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    feature_ts: datetime
    feature_set_version: int = FEATURE_SET_VERSION
    feature_values: dict[str, float | None]
    insufficient_history: bool
    has_gap: bool

    @field_validator("feature_ts")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("feature_ts must be timezone-aware")
        return value.astimezone(UTC)
