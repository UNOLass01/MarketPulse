"""Cyclical time encodings: hour-of-day and day-of-week sine/cosine."""

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from marketpulse.features.temporal import day_of_week_sin_cos, hour_of_day_sin_cos

pytestmark = pytest.mark.unit


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_hour_of_day_midnight_encodes_to_zero_angle() -> None:
    sin, cos = hour_of_day_sin_cos(datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC))
    assert sin == pytest.approx(0.0, abs=1e-9)
    assert cos == pytest.approx(1.0, abs=1e-9)


def test_hour_of_day_noon_is_opposite_midnight() -> None:
    sin, cos = hour_of_day_sin_cos(datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))
    assert sin == pytest.approx(0.0, abs=1e-9)
    assert cos == pytest.approx(-1.0, abs=1e-9)


def test_hour_23_and_hour_0_are_close_in_encoded_space() -> None:
    late = hour_of_day_sin_cos(datetime(2026, 1, 1, 23, 59, 0, tzinfo=UTC))
    early = hour_of_day_sin_cos(datetime(2026, 1, 2, 0, 1, 0, tzinfo=UTC))
    midday = hour_of_day_sin_cos(datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))

    # Two minutes apart across midnight should be far closer than half a day apart.
    assert _euclidean(late, early) < _euclidean(late, midday)
    assert _euclidean(late, early) < 0.05


def test_hour_of_day_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        hour_of_day_sin_cos(datetime(2026, 1, 1, 0, 0, 0))  # noqa: DTZ001


def test_hour_of_day_normalises_non_utc_timezone_before_encoding() -> None:
    plus_two = timezone(timedelta(hours=2))
    # 01:00+02:00 is 23:00 UTC the previous day.
    sin, cos = hour_of_day_sin_cos(datetime(2026, 1, 2, 1, 0, 0, tzinfo=plus_two))
    expected_sin, expected_cos = hour_of_day_sin_cos(datetime(2026, 1, 1, 23, 0, 0, tzinfo=UTC))
    assert sin == pytest.approx(expected_sin)
    assert cos == pytest.approx(expected_cos)


def test_day_of_week_monday_encodes_to_zero_angle() -> None:
    # 2026-01-05 is a Monday.
    sin, cos = day_of_week_sin_cos(datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC))
    assert sin == pytest.approx(0.0, abs=1e-9)
    assert cos == pytest.approx(1.0, abs=1e-9)


def test_sunday_and_monday_are_close_in_encoded_space() -> None:
    sunday = day_of_week_sin_cos(datetime(2026, 1, 4, 0, 0, 0, tzinfo=UTC))
    monday = day_of_week_sin_cos(datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC))
    thursday = day_of_week_sin_cos(datetime(2026, 1, 8, 0, 0, 0, tzinfo=UTC))

    assert _euclidean(sunday, monday) < _euclidean(sunday, thursday)


def test_day_of_week_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        day_of_week_sin_cos(datetime(2026, 1, 1))  # noqa: DTZ001
