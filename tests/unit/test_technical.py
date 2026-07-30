"""Pure technical indicator functions: everything here is testable with
plain lists of floats — no timestamps, no I/O. Covers the phase-2 test list
items "zero-denominator returns null, not inf/NaN" and provides the
building blocks the pipeline-level leakage tests rely on being correct.
"""

import math

import pytest

from marketpulse.features import technical

pytestmark = pytest.mark.unit


def test_moving_average_of_empty_is_none() -> None:
    assert technical.moving_average([]) is None


def test_moving_average() -> None:
    assert technical.moving_average([1.0, 2.0, 3.0]) == 2.0


def test_ratio_to_baseline_zero_denominator_returns_none() -> None:
    assert technical.ratio_to_baseline(100.0, 0.0) is None


def test_ratio_to_baseline_missing_baseline_returns_none() -> None:
    assert technical.ratio_to_baseline(100.0, None) is None


def test_ratio_to_baseline_computes_relative_change() -> None:
    assert technical.ratio_to_baseline(110.0, 100.0) == pytest.approx(0.10)


def test_exponential_moving_average_empty_is_none() -> None:
    assert technical.exponential_moving_average([]) is None


def test_exponential_moving_average_single_value_is_itself() -> None:
    assert technical.exponential_moving_average([42.0]) == 42.0


def test_exponential_moving_average_deterministic_for_same_input() -> None:
    values = [10.0, 11.0, 12.0, 11.5, 13.0]
    assert technical.exponential_moving_average(values) == technical.exponential_moving_average(
        values
    )


def test_exponential_moving_average_matches_hand_computed_value() -> None:
    # n=3 -> alpha = 2/4 = 0.5
    # ema0 = 10; ema1 = 0.5*20 + 0.5*10 = 15; ema2 = 0.5*30 + 0.5*15 = 22.5
    assert technical.exponential_moving_average([10.0, 20.0, 30.0]) == pytest.approx(22.5)


def test_return_volatility_needs_at_least_three_prices() -> None:
    assert technical.return_volatility([1.0, 2.0]) is None


def test_return_volatility_zero_for_constant_prices() -> None:
    assert technical.return_volatility([100.0, 100.0, 100.0]) == pytest.approx(0.0)


def test_return_volatility_zero_price_guards_to_none() -> None:
    assert technical.return_volatility([0.0, 1.0, 2.0]) is None


def test_realised_volatility_needs_two_prices() -> None:
    assert technical.realised_volatility([1.0]) is None


def test_realised_volatility_zero_for_constant_prices() -> None:
    assert technical.realised_volatility([100.0, 100.0, 100.0]) == pytest.approx(0.0)


def test_realised_volatility_zero_price_guards_to_none() -> None:
    assert technical.realised_volatility([0.0, 5.0]) is None


def test_high_low_range_empty_is_none() -> None:
    assert technical.high_low_range([]) is None


def test_high_low_range_computes_relative_spread() -> None:
    assert technical.high_low_range([100.0, 110.0, 90.0]) == pytest.approx((110 - 90) / 90)


def test_high_low_range_zero_low_guards_to_none() -> None:
    assert technical.high_low_range([0.0, 5.0]) is None


def test_rate_of_change_needs_two_prices() -> None:
    assert technical.rate_of_change([1.0]) is None


def test_rate_of_change_computes_relative_move() -> None:
    assert technical.rate_of_change([100.0, 90.0, 120.0]) == pytest.approx(0.20)


def test_rate_of_change_zero_first_price_guards_to_none() -> None:
    assert technical.rate_of_change([0.0, 5.0]) is None


def test_rsi_needs_two_prices() -> None:
    assert technical.rsi([1.0]) is None


def test_rsi_all_gains_is_100() -> None:
    assert technical.rsi([1.0, 2.0, 3.0, 4.0]) == pytest.approx(100.0)


def test_rsi_all_losses_is_0() -> None:
    assert technical.rsi([4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.0)


def test_rsi_flat_prices_is_neutral_50() -> None:
    assert technical.rsi([5.0, 5.0, 5.0]) == pytest.approx(50.0)


def test_rsi_is_bounded_0_100() -> None:
    prices = [10.0, 12.0, 9.0, 15.0, 8.0, 20.0, 5.0]
    value = technical.rsi(prices)
    assert value is not None
    assert 0.0 <= value <= 100.0
    assert not math.isnan(value)
    assert not math.isinf(value)


def test_direction_streak_too_short_is_zero() -> None:
    assert technical.direction_streak([1.0]) == 0


def test_direction_streak_flat_last_move_is_zero() -> None:
    assert technical.direction_streak([1.0, 2.0, 2.0]) == 0


def test_direction_streak_counts_consecutive_up_moves() -> None:
    assert technical.direction_streak([1.0, 2.0, 3.0, 4.0]) == 3


def test_direction_streak_counts_consecutive_down_moves_as_negative() -> None:
    assert technical.direction_streak([4.0, 3.0, 2.0, 1.0]) == -3


def test_direction_streak_stops_at_direction_change() -> None:
    # up, up, down -> most recent move is down, streak length 1
    assert technical.direction_streak([1.0, 2.0, 3.0, 2.0]) == -1
