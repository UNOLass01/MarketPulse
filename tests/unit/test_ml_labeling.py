"""Forward-return labeling: known price sequence -> expected labels, exact
deadband boundary cases (the phase-3 plan's most leakage-adjacent unit)."""

import pytest

from marketpulse.ml.labeling import (
    INDEX_TO_LABEL,
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_TO_INDEX,
    LABEL_UP,
    LABELS,
    derive_theta,
    forward_return,
    label_from_return,
)

pytestmark = pytest.mark.unit


def test_forward_return_up() -> None:
    assert forward_return(100.0, 110.0) == pytest.approx(0.10)


def test_forward_return_down() -> None:
    assert forward_return(100.0, 90.0) == pytest.approx(-0.10)


def test_forward_return_flat() -> None:
    assert forward_return(100.0, 100.0) == 0.0


def test_forward_return_rejects_non_positive_current_price() -> None:
    with pytest.raises(ValueError):
        forward_return(0.0, 100.0)
    with pytest.raises(ValueError):
        forward_return(-5.0, 100.0)


@pytest.mark.parametrize(
    ("forward_ret", "theta", "expected"),
    [
        (0.05, 0.02, LABEL_UP),
        (-0.05, 0.02, LABEL_DOWN),
        (0.01, 0.02, LABEL_STABLE),
        (-0.01, 0.02, LABEL_STABLE),
        (0.0, 0.02, LABEL_STABLE),
    ],
)
def test_label_from_return_known_cases(forward_ret: float, theta: float, expected: str) -> None:
    assert label_from_return(forward_ret, theta) == expected


def test_label_from_return_deadband_boundary_is_exact() -> None:
    theta = 0.02
    # Exactly at the boundary is STABLE (strict >/< only), on both sides.
    assert label_from_return(theta, theta) == LABEL_STABLE
    assert label_from_return(-theta, theta) == LABEL_STABLE
    # A single float ULP beyond the boundary flips the label.
    epsilon = 1e-12
    assert label_from_return(theta + epsilon, theta) == LABEL_UP
    assert label_from_return(-theta - epsilon, theta) == LABEL_DOWN


def test_label_from_return_rejects_negative_theta() -> None:
    with pytest.raises(ValueError):
        label_from_return(0.01, -0.01)


def test_derive_theta_known_distribution() -> None:
    # |returns| = [0, 1, 2, ..., 9] (as fractions); the 0.4 quantile of a
    # 0..9 uniform grid via numpy's default (linear) interpolation is 3.6.
    returns = [float(i) / 100 for i in range(10)]
    theta = derive_theta(returns, quantile=0.4)
    assert theta == pytest.approx(0.036)


def test_derive_theta_uses_absolute_value() -> None:
    positive = derive_theta([0.01, 0.02, 0.03, 0.04, 0.05], quantile=0.5)
    mixed_sign = derive_theta([-0.01, -0.02, 0.03, 0.04, 0.05], quantile=0.5)
    assert positive == pytest.approx(mixed_sign)


def test_derive_theta_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        derive_theta([])


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1])
def test_derive_theta_rejects_quantile_outside_unit_interval(quantile: float) -> None:
    with pytest.raises(ValueError):
        derive_theta([0.01, 0.02], quantile=quantile)


def test_label_index_round_trip_is_alphabetical_and_consistent() -> None:
    assert LABELS == (LABEL_DOWN, LABEL_STABLE, LABEL_UP)
    for label in LABELS:
        assert INDEX_TO_LABEL[LABEL_TO_INDEX[label]] == label
    assert LABEL_TO_INDEX == {"DOWN": 0, "STABLE": 1, "UP": 2}
