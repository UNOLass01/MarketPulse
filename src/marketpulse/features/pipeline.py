"""Compose window state into a :class:`~marketpulse.contracts.features.FeatureVector`.

This is the only place that decides *which* window feeds *which* feature —
``features.technical`` and ``features.temporal`` know nothing about minutes
or hours, only about the plain lists of numbers they're handed. Keeping that
split means every indicator is testable in isolation with plain floats,
while the leakage-sensitive windowing logic is concentrated in one function
that both the streaming consumer and the offline seed script call
identically, guaranteeing online/offline parity by construction.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

from marketpulse.contracts.features import FeatureVector
from marketpulse.features import technical, temporal
from marketpulse.features.registry import FEATURE_SET_VERSION
from marketpulse.features.windows import Observation, detect_gap, has_full_coverage, slice_window

WINDOW_1M = timedelta(minutes=1)
WINDOW_5M = timedelta(minutes=5)
WINDOW_15M = timedelta(minutes=15)
WINDOW_60M = timedelta(minutes=60)
WINDOW_1H = timedelta(hours=1)
WINDOW_24H = timedelta(hours=24)

# The widest lookback any feature uses — gap detection covers at least this much.
GAP_DETECTION_WINDOW = WINDOW_24H


def _windowed(
    observations: Sequence[Observation],
    *,
    as_of: datetime,
    window: timedelta,
) -> list[Observation] | None:
    """The observations inside ``window``, or ``None`` if coverage is insufficient.

    No call site here overrides ``has_full_coverage``'s ``min_points`` — this
    used to redeclare its own default, which shadowed a change to
    ``has_full_coverage``'s default without anyone noticing. Not repeating
    the parameter is deliberate: one definition of "enough points" for the
    whole pipeline.
    """
    if not has_full_coverage(observations, as_of=as_of, window=window):
        return None
    return slice_window(observations, as_of=as_of, window=window)


def compute_feature_vector(
    symbol: str,
    observations: Sequence[Observation],
    *,
    as_of: datetime,
    gap_threshold: timedelta,
) -> FeatureVector:
    """Pure: only ever considers observations with ``observed_at <= as_of``.

    ``observations`` need not be pre-truncated by the caller — filtering
    happens here, so a buffer that already holds points beyond ``as_of``
    (e.g. because the caller is replaying history) can never leak into the
    result. ``as_of`` must match the timestamp of the latest observation in
    ``observations`` — that's the tick this vector is being computed for.
    """
    history = [obs for obs in observations if obs.observed_at <= as_of]
    if not history or history[-1].observed_at != as_of:
        raise ValueError("as_of must match the latest observation at or before it")

    current_price = history[-1].price
    current_volume = history[-1].volume
    previous_volume = history[-2].volume if len(history) >= 2 else None

    prices_1m = _windowed(history, as_of=as_of, window=WINDOW_1M)
    prices_5m = _windowed(history, as_of=as_of, window=WINDOW_5M)
    prices_15m = _windowed(history, as_of=as_of, window=WINDOW_15M)
    prices_60m = _windowed(history, as_of=as_of, window=WINDOW_60M)
    prices_1h = _windowed(history, as_of=as_of, window=WINDOW_1H)
    prices_24h = _windowed(history, as_of=as_of, window=WINDOW_24H)
    volumes_15m = _windowed(history, as_of=as_of, window=WINDOW_15M)

    def prices_of(window: list[Observation] | None) -> list[float] | None:
        return None if window is None else [obs.price for obs in window]

    def volumes_of(window: list[Observation] | None) -> list[float] | None:
        return None if window is None else [obs.volume for obs in window]

    p5, p15, p60, p1m, p1h, p24h = (
        prices_of(prices_5m),
        prices_of(prices_15m),
        prices_of(prices_60m),
        prices_of(prices_1m),
        prices_of(prices_1h),
        prices_of(prices_24h),
    )
    v15 = volumes_of(volumes_15m)

    ma_5m = technical.moving_average(p5) if p5 is not None else None
    ma_15m = technical.moving_average(p15) if p15 is not None else None
    ma_60m = technical.moving_average(p60) if p60 is not None else None
    ma_volume_15m = technical.moving_average(v15) if v15 is not None else None

    hour_sin, hour_cos = temporal.hour_of_day_sin_cos(as_of)
    dow_sin, dow_cos = temporal.day_of_week_sin_cos(as_of)

    feature_values: dict[str, float | None] = {
        "ma_5m": ma_5m,
        "ma_15m": ma_15m,
        "ma_60m": ma_60m,
        "price_to_ma_5m": technical.ratio_to_baseline(current_price, ma_5m),
        "price_to_ma_15m": technical.ratio_to_baseline(current_price, ma_15m),
        "price_to_ma_60m": technical.ratio_to_baseline(current_price, ma_60m),
        "ema_15m": technical.exponential_moving_average(p15) if p15 is not None else None,
        "volatility_15m": technical.return_volatility(p15) if p15 is not None else None,
        "volatility_60m": technical.return_volatility(p60) if p60 is not None else None,
        "high_low_range_15m": technical.high_low_range(p15) if p15 is not None else None,
        "realised_vol_1h": technical.realised_volatility(p1h) if p1h is not None else None,
        "realised_vol_24h": technical.realised_volatility(p24h) if p24h is not None else None,
        "roc_1m": technical.rate_of_change(p1m) if p1m is not None else None,
        "roc_5m": technical.rate_of_change(p5) if p5 is not None else None,
        "roc_15m": technical.rate_of_change(p15) if p15 is not None else None,
        "rsi_15m": technical.rsi(p15) if p15 is not None else None,
        "direction_streak": (float(technical.direction_streak(p60)) if p60 is not None else None),
        "volume_change": technical.ratio_to_baseline(current_volume, previous_volume),
        "volume_relative_to_ma_15m": technical.ratio_to_baseline(current_volume, ma_volume_15m),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }

    base_windows_sufficient = all(w is not None for w in (p1m, p5, p15, p60, p1h, p24h, v15))
    insufficient_history = not (base_windows_sufficient and previous_volume is not None)

    has_gap = detect_gap(history, as_of=as_of, window=GAP_DETECTION_WINDOW, threshold=gap_threshold)

    return FeatureVector(
        symbol=symbol,
        feature_ts=as_of,
        feature_set_version=FEATURE_SET_VERSION,
        feature_values=feature_values,
        insufficient_history=insufficient_history,
        has_gap=has_gap,
    )
