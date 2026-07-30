"""Bounded window state: ring buffer memory bounds, no-look-ahead snapshot
filtering, coverage/gap detection helpers.
"""

from datetime import UTC, datetime, timedelta

import pytest

from marketpulse.features.windows import (
    Observation,
    SymbolWindow,
    WindowStore,
    detect_gap,
    has_full_coverage,
    slice_window,
)

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _obs(minutes: float, price: float = 100.0, volume: float = 1.0) -> Observation:
    return Observation(observed_at=T0 + timedelta(minutes=minutes), price=price, volume=volume)


def test_window_memory_stays_bounded_after_10k_observations() -> None:
    window = SymbolWindow(max_history=timedelta(hours=24), max_points=5000)
    for i in range(10_000):
        window.push(_obs(i * 0.01))  # 1s apart -> well within 24h for all 10k
    assert len(window) <= 5000


def test_time_based_eviction_drops_points_older_than_max_history() -> None:
    window = SymbolWindow(max_history=timedelta(minutes=10), max_points=10_000)
    window.push(_obs(0))
    window.push(_obs(5))
    window.push(_obs(20))  # more than 10 minutes past the first point

    snapshot = window.snapshot()
    assert all(obs.observed_at >= T0 + timedelta(minutes=10) for obs in snapshot)
    assert len(snapshot) == 1


def test_snapshot_filters_to_as_of_even_if_buffer_holds_future_points() -> None:
    window = SymbolWindow()
    window.push(_obs(0))
    window.push(_obs(5))
    window.push(_obs(10))

    snapshot = window.snapshot(as_of=T0 + timedelta(minutes=5))
    assert [obs.observed_at for obs in snapshot] == [T0, T0 + timedelta(minutes=5)]


def test_window_store_isolates_symbols() -> None:
    store = WindowStore()
    store.push("BTC-USD", _obs(0, price=100.0))
    store.push("ETH-USD", _obs(0, price=2000.0))

    assert [o.price for o in store.snapshot("BTC-USD")] == [100.0]
    assert [o.price for o in store.snapshot("ETH-USD")] == [2000.0]


def test_window_store_snapshot_for_unknown_symbol_is_empty() -> None:
    store = WindowStore()
    assert store.snapshot("UNKNOWN") == []


def test_slice_window_is_half_open_on_the_lower_bound() -> None:
    observations = [_obs(0), _obs(5), _obs(10)]
    as_of = T0 + timedelta(minutes=10)
    sliced = slice_window(observations, as_of=as_of, window=timedelta(minutes=10))
    # T0 itself is exactly `window` before as_of -> excluded (start < observed_at)
    assert [o.observed_at for o in sliced] == [T0 + timedelta(minutes=5), as_of]


def test_has_full_coverage_false_when_symbol_too_new() -> None:
    # Only 3 minutes of history exists at all -> a 15m window can't be trusted yet.
    observations = [_obs(0), _obs(1), _obs(2), _obs(3)]
    assert not has_full_coverage(
        observations, as_of=T0 + timedelta(minutes=3), window=timedelta(minutes=15)
    )


def test_has_full_coverage_false_when_too_few_points_in_window() -> None:
    observations = [_obs(-20), _obs(0)]
    assert not has_full_coverage(observations, as_of=T0, window=timedelta(minutes=15), min_points=2)


def test_has_full_coverage_true_when_window_is_fully_populated() -> None:
    observations = [_obs(-20), _obs(-10), _obs(-5), _obs(0)]
    assert has_full_coverage(observations, as_of=T0, window=timedelta(minutes=15), min_points=2)


def test_has_full_coverage_empty_observations_is_false() -> None:
    assert not has_full_coverage([], as_of=T0, window=timedelta(minutes=15))


def test_detect_gap_false_for_evenly_spaced_observations() -> None:
    observations = [_obs(i) for i in range(0, 30, 5)]
    assert not detect_gap(
        observations,
        as_of=T0 + timedelta(minutes=25),
        window=timedelta(hours=1),
        threshold=timedelta(minutes=6),
    )


def test_detect_gap_true_across_an_injected_time_hole() -> None:
    observations = [_obs(0), _obs(1), _obs(2), _obs(50), _obs(51)]
    assert detect_gap(
        observations,
        as_of=T0 + timedelta(minutes=51),
        window=timedelta(hours=1),
        threshold=timedelta(minutes=5),
    )


def test_default_retention_leaves_margin_for_the_widest_feature_window() -> None:
    """Regression guard: default retention must exceed the widest feature
    window (24h, see ``features.pipeline.WINDOW_24H``) with real margin, or
    eviction always discards the one observation coverage needs to prove a
    24h window is fully populated -- see ``DEFAULT_MAX_HISTORY``'s docstring.

    Steps of 11 minutes (not a divisor of 24h) deliberately avoid landing
    exactly on the eviction cutoff, matching irregular real-world tick
    timestamps rather than a coincidentally-aligned round number.
    """
    window = SymbolWindow()  # DEFAULT_MAX_HISTORY / DEFAULT_MAX_POINTS
    step_minutes, count = 11, 170  # ~31 hours of history

    for i in range(count):
        window.push(_obs(i * step_minutes))

    as_of = T0 + timedelta(minutes=(count - 1) * step_minutes)
    assert has_full_coverage(window.snapshot(), as_of=as_of, window=timedelta(hours=24))


def test_retention_exactly_equal_to_window_cannot_reliably_report_sufficient_coverage() -> None:
    """Documents *why* retention must exceed the window: with retention
    exactly equal to it, eviction discards the observation that would prove
    the window is fully populated in the same push that would have made it
    available, so coverage never reliably reports sufficient -- exactly the
    failure this project hit backfilling real (non-round-number-spaced)
    history before ``DEFAULT_MAX_HISTORY`` was widened past 24h.
    """
    window = SymbolWindow(max_history=timedelta(hours=24), max_points=10_000)
    step_minutes, count = 11, 170  # ~31 hours of history

    for i in range(count):
        window.push(_obs(i * step_minutes))

    as_of = T0 + timedelta(minutes=(count - 1) * step_minutes)
    assert not has_full_coverage(window.snapshot(), as_of=as_of, window=timedelta(hours=24))


def test_detect_gap_ignores_holes_outside_the_window() -> None:
    # The gap is between minute 0 and minute 50; a 5-minute window ending at
    # minute 51 never sees minute 0, so it shouldn't flag anything.
    observations = [_obs(0), _obs(50), _obs(51)]
    assert not detect_gap(
        observations,
        as_of=T0 + timedelta(minutes=51),
        window=timedelta(minutes=5),
        threshold=timedelta(minutes=5),
    )
