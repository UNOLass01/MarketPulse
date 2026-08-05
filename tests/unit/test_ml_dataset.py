"""Dataset assembly: chronological join/label/split with an embargo gap.

These are the phase-3 "five tests that matter most" for leakage: a
contiguous split (no embargo) would let a training example's label -- which
reads price at t+H -- peek at data that's chronologically inside validation.
"""

import math
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.ml.dataset import (
    Dataset,
    FeatureRow,
    InsufficientDataError,
    SplitConfig,
    assemble_dataset,
)
from marketpulse.ml.labeling import forward_return, label_from_return

pytestmark = pytest.mark.unit

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=1)
HORIZON = timedelta(minutes=15)
TOLERANCE = timedelta(seconds=30)


def _make_rows(
    symbol: str,
    n: int,
    *,
    start: datetime = BASE_TIME,
    step: timedelta = STEP,
    price_fn: Callable[[int], float] | None = None,
    insufficient_history: Callable[[int], bool] | None = None,
    has_gap: Callable[[int], bool] | None = None,
) -> list[FeatureRow]:
    # A sine wave (not a monotonic walk) so forward returns span both signs --
    # a monotonically drifting price never produces a DOWN label, which would
    # make the default class-balance guard (min_class_frac) unsatisfiable.
    price_fn = price_fn or (lambda i: 100.0 + 5.0 * math.sin(i * 0.3))
    insufficient_history = insufficient_history or (lambda i: False)
    has_gap = has_gap or (lambda i: False)
    return [
        FeatureRow(
            symbol=symbol,
            feature_ts=start + i * step,
            price=price_fn(i),
            feature_values=dict.fromkeys(FEATURE_NAMES, 1.0),
            insufficient_history=insufficient_history(i),
            has_gap=has_gap(i),
        )
        for i in range(n)
    ]


def _lenient_split(**overrides: object) -> SplitConfig:
    base = {"train_frac": 0.6, "val_frac": 0.2, "min_rows_per_split": 1, "min_class_frac": 0.0}
    base.update(overrides)
    return SplitConfig(**base)  # type: ignore[arg-type]


def _assemble(rows: list[FeatureRow], **kwargs: object) -> Dataset:
    kwargs.setdefault("horizon", HORIZON)
    kwargs.setdefault("feature_set_version", 1)
    kwargs.setdefault("forward_price_tolerance", TOLERANCE)
    kwargs.setdefault("split", _lenient_split())
    return assemble_dataset(rows, **kwargs)  # type: ignore[arg-type]


# --- label correctness through the full join -------------------------------


def test_labels_match_hand_computed_forward_return() -> None:
    n = 300
    horizon_steps = HORIZON // STEP

    def price_fn(i: int) -> float:
        # A kink midway so both up- and down-moving regions are exercised,
        # staying well clear of zero throughout.
        return 100.0 + i if i < 150 else 100.0 + 150 - (i - 150) * 0.5

    rows = _make_rows("BTC-USD", n, price_fn=price_fn)
    theta = 0.01
    dataset = _assemble(rows, theta=theta)

    all_rows = pd.concat([dataset.train, dataset.validation, dataset.test], ignore_index=True)
    prices = {i: price_fn(i) for i in range(n)}
    ts_to_index = {pd.Timestamp(rows[i].feature_ts): i for i in range(n)}

    assert len(all_rows) > 0
    for _, row in all_rows.iterrows():
        i = ts_to_index[row["feature_ts"]]
        expected_future = prices[i + horizon_steps]
        expected_return = forward_return(prices[i], expected_future)
        expected_label = label_from_return(expected_return, theta)
        assert row["forward_return"] == pytest.approx(expected_return)
        assert row["label"] == expected_label


def test_theta_is_derived_when_not_given() -> None:
    rows = _make_rows("BTC-USD", 300)
    dataset = _assemble(rows, theta=None, theta_quantile=0.4)
    assert dataset.theta > 0


# --- chronology + embargo ---------------------------------------------------


def test_split_is_chronological_and_non_overlapping() -> None:
    rows = _make_rows("BTC-USD", 1000)
    dataset = _assemble(rows, split=_lenient_split(train_frac=0.7, val_frac=0.15))

    for frame in (dataset.train, dataset.validation, dataset.test):
        assert frame["feature_ts"].is_monotonic_increasing

    assert dataset.train["feature_ts"].max() < dataset.validation["feature_ts"].min()
    assert dataset.validation["feature_ts"].max() < dataset.test["feature_ts"].min()


def test_embargo_gap_is_at_least_horizon() -> None:
    rows = _make_rows("BTC-USD", 1000)
    dataset = _assemble(rows, split=_lenient_split(train_frac=0.7, val_frac=0.15))

    gap_train_val = dataset.validation["feature_ts"].min() - dataset.train["feature_ts"].max()
    gap_val_test = dataset.test["feature_ts"].min() - dataset.validation["feature_ts"].max()
    assert gap_train_val >= HORIZON
    assert gap_val_test >= HORIZON


def test_assert_not_shuffled_split_never_reorders_input() -> None:
    rows = _make_rows("BTC-USD", 500)
    ordered = _assemble(list(rows), split=_lenient_split(train_frac=0.7, val_frac=0.15), theta=0.01)

    shuffled_rows = list(rows)
    random.Random(1234).shuffle(shuffled_rows)
    shuffled_split = _lenient_split(train_frac=0.7, val_frac=0.15)
    from_shuffled = _assemble(shuffled_rows, split=shuffled_split, theta=0.01)

    for frame in (from_shuffled.train, from_shuffled.validation, from_shuffled.test):
        assert frame["feature_ts"].is_monotonic_increasing

    for split_name in ("train", "validation", "test"):
        left = getattr(ordered, split_name)["feature_ts"].tolist()
        right = getattr(from_shuffled, split_name)["feature_ts"].tolist()
        assert left == right


# --- exclusions --------------------------------------------------------------


def test_recent_horizon_tail_is_excluded_from_every_split() -> None:
    n = 200
    rows = _make_rows("BTC-USD", n)
    dataset = _assemble(rows, split=_lenient_split(train_frac=0.7, val_frac=0.15))

    cutoff = rows[-1].feature_ts - HORIZON
    combined_max = max(
        dataset.train["feature_ts"].max(),
        dataset.validation["feature_ts"].max(),
        dataset.test["feature_ts"].max(),
    )
    assert combined_max <= cutoff


def test_insufficient_history_and_gap_rows_are_dropped() -> None:
    n = 200
    flagged_insufficient = {5, 15, 25}
    flagged_gap = {6, 16, 26}
    rows = _make_rows(
        "BTC-USD",
        n,
        insufficient_history=lambda i: i in flagged_insufficient,
        has_gap=lambda i: i in flagged_gap,
    )
    dataset = _assemble(rows, split=_lenient_split(train_frac=0.7, val_frac=0.15))

    flagged_timestamps = {rows[i].feature_ts for i in flagged_insufficient | flagged_gap}
    all_ts = (
        set(dataset.train["feature_ts"])
        | set(dataset.validation["feature_ts"])
        | set(dataset.test["feature_ts"])
    )
    assert not (flagged_timestamps & all_ts)


# --- sufficiency guards --------------------------------------------------------


def test_empty_input_raises() -> None:
    with pytest.raises(InsufficientDataError):
        assemble_dataset(
            [], horizon=HORIZON, feature_set_version=1, forward_price_tolerance=TOLERANCE
        )


def test_too_few_rows_raises_rather_than_training() -> None:
    rows = _make_rows("BTC-USD", 30)
    with pytest.raises(InsufficientDataError):
        _assemble(rows, split=SplitConfig())  # default min_rows_per_split=200 far exceeds n=30


def test_class_collapse_raises() -> None:
    # A perfectly flat price series -> every forward_return is 0 -> STABLE
    # only, with an explicit theta > 0. Row count is generous but one class
    # is 100% of the data, which must trip the class-balance guard.
    rows = _make_rows("BTC-USD", 400, price_fn=lambda i: 100.0)
    with pytest.raises(InsufficientDataError):
        _assemble(
            rows,
            theta=0.001,
            split=SplitConfig(
                train_frac=0.6, val_frac=0.2, min_rows_per_split=10, min_class_frac=0.05
            ),
        )


# --- multi-symbol --------------------------------------------------------------


def test_multi_symbol_rows_split_independently_then_combined() -> None:
    btc = _make_rows("BTC-USD", 500)
    eth = _make_rows("ETH-USD", 500, price_fn=lambda i: 50.0 + 3.0 * math.sin(i * 0.25 + 1.0))
    dataset = _assemble(btc + eth, split=_lenient_split(train_frac=0.7, val_frac=0.15))

    assert dataset.symbols == ("BTC-USD", "ETH-USD")
    for frame in (dataset.train, dataset.validation, dataset.test):
        assert set(frame["symbol"].unique()) == {"BTC-USD", "ETH-USD"}
        for _, group in frame.groupby("symbol"):
            assert group["feature_ts"].is_monotonic_increasing


def test_row_counts_and_class_distribution_helpers() -> None:
    rows = _make_rows("BTC-USD", 500)
    dataset = _assemble(rows, split=_lenient_split(train_frac=0.7, val_frac=0.15))
    counts = dataset.row_counts()
    assert counts["train"] == len(dataset.train)
    assert counts["validation"] == len(dataset.validation)
    assert counts["test"] == len(dataset.test)
    distribution = dataset.class_distribution("train")
    assert sum(distribution.values()) == len(dataset.train)


def test_reference_feature_stats_covers_every_feature_with_finite_values() -> None:
    rows = _make_rows("BTC-USD", 500)
    dataset = _assemble(rows, split=_lenient_split(train_frac=0.7, val_frac=0.15))
    stats = dataset.reference_feature_stats()
    assert set(stats) == set(FEATURE_NAMES)
    for name, feature_stats in stats.items():
        assert feature_stats["min"] <= feature_stats["p50"] <= feature_stats["max"]
        assert feature_stats["std"] >= 0.0
        # All synthetic rows use a constant 1.0 for every feature.
        assert feature_stats["mean"] == pytest.approx(1.0), name
