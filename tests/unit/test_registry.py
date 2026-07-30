"""Feature registry: ordering is the single source of truth for both
training and serving column order."""

import pytest

from marketpulse.features.registry import FEATURE_NAMES, ordered_values

pytestmark = pytest.mark.unit


def test_feature_names_has_no_duplicates() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_ordered_values_respects_registry_order_not_dict_order() -> None:
    # Deliberately insert keys in the reverse of FEATURE_NAMES order.
    shuffled = dict.fromkeys(reversed(FEATURE_NAMES), 1.0)
    assert ordered_values(shuffled) == [1.0] * len(FEATURE_NAMES)


def test_ordered_values_preserves_none() -> None:
    values = dict.fromkeys(FEATURE_NAMES, None)
    values[FEATURE_NAMES[0]] = 42.0
    result = ordered_values(values)
    assert result[0] == 42.0
    assert all(v is None for v in result[1:])


def test_ordered_values_raises_on_missing_feature() -> None:
    incomplete = dict.fromkeys(FEATURE_NAMES[:-1], 1.0)
    with pytest.raises(KeyError):
        ordered_values(incomplete)
