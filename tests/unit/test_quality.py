"""monitoring.quality's pure check functions -- no DB, no I/O."""

from datetime import UTC, datetime, timedelta

import pytest

from marketpulse.monitoring.quality import (
    check_completeness,
    check_distribution,
    check_freshness,
    check_validity,
)
from marketpulse.storage.models import Feature, RawTick

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)


def _tick(observed_at: datetime, *, price: float = 100.0) -> RawTick:
    return RawTick(observed_at=observed_at, price=price, volume=1.0)


def _feature_row(*, insufficient_history: bool, values: dict[str, float | None]) -> Feature:
    return Feature(
        feature_ts=NOW,
        feature_values=values,
        insufficient_history=insufficient_history,
        has_gap=False,
    )


# --- freshness ---------------------------------------------------------------


def test_freshness_passes_within_max_lag() -> None:
    result = check_freshness(
        NOW - timedelta(minutes=1), now=NOW, max_lag=timedelta(minutes=10), symbol="BTC-USD"
    )
    assert result.passed


def test_freshness_fails_beyond_max_lag() -> None:
    result = check_freshness(
        NOW - timedelta(minutes=30), now=NOW, max_lag=timedelta(minutes=10), symbol="BTC-USD"
    )
    assert not result.passed
    assert result.details["lag_seconds"] == pytest.approx(1800.0)


def test_freshness_fails_closed_with_no_ticks_ever() -> None:
    result = check_freshness(None, now=NOW, max_lag=timedelta(minutes=10), symbol="BTC-USD")
    assert not result.passed
    assert result.details["reason"] == "no_ticks_observed"


# --- completeness --------------------------------------------------------------


def test_completeness_passes_at_full_coverage() -> None:
    window_start = NOW - timedelta(hours=1)
    observed_ats = [window_start + timedelta(seconds=10 * i) for i in range(360)]
    result = check_completeness(
        observed_ats,
        window_start=window_start,
        window_end=NOW,
        expected_interval=timedelta(seconds=10),
        min_ratio=0.95,
        symbol="BTC-USD",
    )
    assert result.passed


def test_completeness_fails_on_gap() -> None:
    window_start = NOW - timedelta(hours=1)
    # Half the expected ticks missing -- a real gap, not noise.
    observed_ats = [window_start + timedelta(seconds=20 * i) for i in range(180)]
    result = check_completeness(
        observed_ats,
        window_start=window_start,
        window_end=NOW,
        expected_interval=timedelta(seconds=10),
        min_ratio=0.95,
        symbol="BTC-USD",
    )
    assert not result.passed
    assert result.details["ratio"] == pytest.approx(0.5, abs=0.01)


# --- validity --------------------------------------------------------------------


def test_validity_passes_clean_data() -> None:
    ticks = [_tick(NOW + timedelta(seconds=i)) for i in range(5)]
    features = [_feature_row(insufficient_history=False, values={"ma_5m": 1.0})]
    result = check_validity(ticks, features, window_start=NOW, window_end=NOW, symbol="BTC-USD")
    assert result.passed
    assert result.details["issues"] == []


def test_validity_fails_on_non_positive_price() -> None:
    ticks = [_tick(NOW, price=0.0)]
    result = check_validity(ticks, [], window_start=NOW, window_end=NOW, symbol="BTC-USD")
    assert not result.passed
    assert any("non-positive price" in issue for issue in result.details["issues"])


def test_validity_fails_on_non_monotonic_timestamps() -> None:
    ticks = [_tick(NOW), _tick(NOW - timedelta(seconds=5))]
    result = check_validity(ticks, [], window_start=NOW, window_end=NOW, symbol="BTC-USD")
    assert not result.passed
    assert any("non-monotonic" in issue for issue in result.details["issues"])


def test_validity_fails_on_null_despite_sufficient_history() -> None:
    features = [_feature_row(insufficient_history=False, values={"ma_5m": None})]
    result = check_validity([], features, window_start=NOW, window_end=NOW, symbol="BTC-USD")
    assert not result.passed
    assert any("null values" in issue for issue in result.details["issues"])


def test_validity_ignores_null_when_history_is_flagged_insufficient() -> None:
    # A legitimate insufficient-history null (CLAUDE.md rule #7) is not a bug.
    features = [_feature_row(insufficient_history=True, values={"ma_5m": None})]
    result = check_validity([], features, window_start=NOW, window_end=NOW, symbol="BTC-USD")
    assert result.passed


# --- distribution sanity --------------------------------------------------------


def test_distribution_passes_within_threshold() -> None:
    result = check_distribution(
        {"ma_5m": 105.0},
        {"ma_5m": 100.0},
        max_relative_shift=0.5,
        window_start=NOW,
        window_end=NOW,
        symbol="BTC-USD",
    )
    assert result.passed


def test_distribution_fails_on_implausible_shift() -> None:
    result = check_distribution(
        {"ma_5m": 200.0},
        {"ma_5m": 100.0},
        max_relative_shift=0.5,
        window_start=NOW,
        window_end=NOW,
        symbol="BTC-USD",
    )
    assert not result.passed
    assert "ma_5m" in result.details["shifted_features"]


def test_distribution_skips_feature_with_no_reference_data() -> None:
    # Cold start (new symbol / new feature): nothing to compare against yet,
    # not an automatic failure.
    result = check_distribution(
        {"ma_5m": 999.0},
        {},
        max_relative_shift=0.5,
        window_start=NOW,
        window_end=NOW,
        symbol="BTC-USD",
    )
    assert result.passed


def test_distribution_skips_zero_reference_mean_rather_than_dividing_by_zero() -> None:
    result = check_distribution(
        {"roc_1m": 0.01},
        {"roc_1m": 0.0},
        max_relative_shift=0.5,
        window_start=NOW,
        window_end=NOW,
        symbol="BTC-USD",
    )
    assert result.passed
