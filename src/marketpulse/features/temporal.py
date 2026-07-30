"""Cyclical time-of-day / day-of-week encodings.

Plain sine/cosine so hour 23 and hour 0 (or Sunday and Monday) sit next to
each other in encoded space instead of jumping across the number line — a
raw integer hour would teach the model a false discontinuity at midnight.
"""

import math
from datetime import UTC, datetime


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def hour_of_day_sin_cos(observed_at: datetime) -> tuple[float, float]:
    """Sine/cosine encoding of the second-of-day, period 24h."""
    dt = _require_utc(observed_at)
    seconds_into_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    angle = 2 * math.pi * seconds_into_day / 86400
    return math.sin(angle), math.cos(angle)


def day_of_week_sin_cos(observed_at: datetime) -> tuple[float, float]:
    """Sine/cosine encoding of the ISO weekday (Monday=0), period 7 days."""
    dt = _require_utc(observed_at)
    angle = 2 * math.pi * dt.weekday() / 7
    return math.sin(angle), math.cos(angle)
