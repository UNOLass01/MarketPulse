"""Canonical, versioned feature ordering.

Serving and training both read column order from here — never from dict or
DataFrame iteration order, which is not guaranteed to stay stable across
Python versions, pandas versions, or code paths. Bump ``FEATURE_SET_VERSION``
whenever ``FEATURE_NAMES`` changes; old and new versions coexist in storage
(see ``storage.models.Feature``'s unique constraint).
"""

from collections.abc import Mapping

FEATURE_SET_VERSION = 1

FEATURE_NAMES: tuple[str, ...] = (
    "ma_5m",
    "ma_15m",
    "ma_60m",
    "price_to_ma_5m",
    "price_to_ma_15m",
    "price_to_ma_60m",
    "ema_15m",
    "volatility_15m",
    "volatility_60m",
    "high_low_range_15m",
    "realised_vol_1h",
    "realised_vol_24h",
    "roc_1m",
    "roc_5m",
    "roc_15m",
    "rsi_15m",
    "direction_streak",
    "volume_change",
    "volume_relative_to_ma_15m",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)


def ordered_values(feature_values: Mapping[str, float | None]) -> list[float | None]:
    """Project a name -> value mapping into the canonical column order.

    Raises ``KeyError`` if ``feature_values`` is missing a registered name —
    silently filling a gap with ``None`` would make a pipeline bug (a feature
    that was never computed) indistinguishable from a legitimate
    insufficient-history null.
    """
    return [feature_values[name] for name in FEATURE_NAMES]
