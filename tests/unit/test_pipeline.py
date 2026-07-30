"""Feature pipeline composition.

Covers the two "tests that matter most" from the phase-2 plan: look-ahead
leakage (parametrised across every registered feature) and the
insufficient-history / gap flags, plus the invariants the online/offline
parity guarantee in ``tests/integration`` depends on (determinism, purity).
"""

from datetime import UTC, datetime, timedelta

import pytest

from marketpulse.features.pipeline import compute_feature_vector
from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.features.windows import Observation

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, tzinfo=UTC)
GAP_THRESHOLD = timedelta(minutes=5)


def _synthetic_series(count: int, *, step_minutes: float = 10.0) -> list[Observation]:
    observations = []
    for i in range(count):
        price = 100.0 + 5.0 * ((i % 7) - 3) + 0.1 * i
        volume = 10.0 + (i % 4)
        observations.append(
            Observation(
                observed_at=T0 + timedelta(minutes=i * step_minutes),
                price=price,
                volume=volume,
            )
        )
    return observations


@pytest.fixture(scope="module")
def full_series() -> list[Observation]:
    # 3200 points, 30 seconds apart -> ~26.7 hours of history: comfortably
    # past every feature's widest lookback (realised_vol_24h) while staying
    # dense enough that even the 1-minute window has real coverage.
    return _synthetic_series(3200, step_minutes=0.5)


@pytest.fixture(scope="module")
def leakage_pair(
    full_series: list[Observation],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    as_of_index = 3000  # 25 hours in
    as_of = full_series[as_of_index].observed_at

    truncated = full_series[: as_of_index + 1]
    vector_from_truncated = compute_feature_vector(
        "BTC-USD", truncated, as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    vector_from_full = compute_feature_vector(
        "BTC-USD", full_series, as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    return vector_from_truncated.feature_values, vector_from_full.feature_values


@pytest.mark.parametrize("feature_name", FEATURE_NAMES)
def test_no_look_ahead_leakage(
    leakage_pair: tuple[dict[str, float | None], dict[str, float | None]], feature_name: str
) -> None:
    """Truncating the sequence after ``t`` must not change the value at ``t``,
    even when the *untruncated* buffer (with points after ``t`` already in
    it) is handed to the same function — it must filter internally, not rely
    on the caller having pre-truncated.
    """
    truncated_values, full_values = leakage_pair
    assert truncated_values[feature_name] == full_values[feature_name]


def test_leakage_pair_uses_a_sequence_long_enough_to_be_meaningful(
    full_series: list[Observation],
) -> None:
    assert len(full_series) == 3200


def test_all_registered_features_are_present_in_the_vector(
    full_series: list[Observation],
) -> None:
    as_of = full_series[3000].observed_at
    vector = compute_feature_vector(
        "BTC-USD", full_series[:3001], as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    assert set(vector.feature_values) == set(FEATURE_NAMES)


def test_insufficient_history_flag_true_and_values_null_not_zero_for_new_symbol() -> None:
    observations = _synthetic_series(3, step_minutes=1.0)
    as_of = observations[-1].observed_at

    vector = compute_feature_vector(
        "BTC-USD", observations, as_of=as_of, gap_threshold=GAP_THRESHOLD
    )

    assert vector.insufficient_history is True
    assert vector.feature_values["ma_60m"] is None
    assert vector.feature_values["realised_vol_24h"] is None
    # A missing feature must never be silently represented as a zero.
    assert vector.feature_values["ma_60m"] != 0.0


def test_temporal_features_are_always_available_despite_insufficient_history() -> None:
    observations = _synthetic_series(1, step_minutes=1.0)
    as_of = observations[-1].observed_at

    vector = compute_feature_vector(
        "BTC-USD", observations, as_of=as_of, gap_threshold=GAP_THRESHOLD
    )

    assert vector.insufficient_history is True
    for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert vector.feature_values[name] is not None


def test_sufficient_history_clears_the_flag(full_series: list[Observation]) -> None:
    as_of = full_series[3000].observed_at
    vector = compute_feature_vector(
        "BTC-USD", full_series[:3001], as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    assert vector.insufficient_history is False


def test_has_gap_false_for_a_regular_series(full_series: list[Observation]) -> None:
    as_of = full_series[3000].observed_at
    vector = compute_feature_vector(
        "BTC-USD", full_series[:3001], as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    assert vector.has_gap is False


def test_has_gap_true_across_an_injected_time_hole(full_series: list[Observation]) -> None:
    observations = list(full_series[:100])
    last = observations[-1]
    gap_point = Observation(
        observed_at=last.observed_at + timedelta(hours=2),
        price=last.price,
        volume=last.volume,
    )
    observations.append(gap_point)

    vector = compute_feature_vector(
        "BTC-USD", observations, as_of=gap_point.observed_at, gap_threshold=GAP_THRESHOLD
    )
    assert vector.has_gap is True


def test_as_of_must_match_the_latest_observation() -> None:
    observations = _synthetic_series(5, step_minutes=1.0)
    with pytest.raises(ValueError, match="as_of"):
        compute_feature_vector(
            "BTC-USD",
            observations,
            as_of=observations[-1].observed_at + timedelta(minutes=1),
            gap_threshold=GAP_THRESHOLD,
        )


def test_deterministic_given_identical_input(full_series: list[Observation]) -> None:
    as_of = full_series[3000].observed_at
    first = compute_feature_vector(
        "BTC-USD", full_series[:3001], as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    second = compute_feature_vector(
        "BTC-USD", full_series[:3001], as_of=as_of, gap_threshold=GAP_THRESHOLD
    )
    assert first.feature_values == second.feature_values
    assert first.insufficient_history == second.insufficient_history
    assert first.has_gap == second.has_gap
