"""FeatureVector schema contract: round-trips and rejects a naive timestamp."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from marketpulse.contracts.features import FeatureVector
from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION

pytestmark = pytest.mark.contract


def _vector() -> FeatureVector:
    return FeatureVector(
        symbol="BTC-USD",
        feature_ts=datetime.now(UTC),
        feature_values=dict.fromkeys(FEATURE_NAMES, 1.0),
        insufficient_history=False,
        has_gap=False,
    )


def test_round_trips_through_json_unchanged() -> None:
    original = _vector()
    restored = FeatureVector.model_validate_json(original.model_dump_json())
    assert restored == original


def test_defaults_to_current_feature_set_version() -> None:
    assert _vector().feature_set_version == FEATURE_SET_VERSION


def test_is_frozen() -> None:
    vector = _vector()
    with pytest.raises(ValidationError):
        vector.has_gap = True  # type: ignore[misc]


def test_naive_feature_ts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureVector(
            symbol="BTC-USD",
            feature_ts=datetime(2026, 1, 1),  # naive, no tzinfo
            feature_values={},
            insufficient_history=False,
            has_gap=False,
        )


def test_non_utc_feature_ts_is_normalised_to_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    vector = FeatureVector(
        symbol="BTC-USD",
        feature_ts=datetime(2026, 1, 1, 2, 0, tzinfo=plus_two),
        feature_values={},
        insufficient_history=False,
        has_gap=False,
    )
    assert vector.feature_ts == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def test_feature_values_preserves_null_entries() -> None:
    values = dict.fromkeys(FEATURE_NAMES, None)
    vector = FeatureVector(
        symbol="BTC-USD",
        feature_ts=datetime.now(UTC),
        feature_values=values,
        insufficient_history=True,
        has_gap=False,
    )
    restored = FeatureVector.model_validate_json(vector.model_dump_json())
    assert all(v is None for v in restored.feature_values.values())
