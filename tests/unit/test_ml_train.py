"""LightGBM training: class weighting, column/label ordering, no scaler
needed (LightGBM is scale-invariant) -- fits fast on small synthetic data."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.ml.config import LightGBMConfig
from marketpulse.ml.dataset import Dataset
from marketpulse.ml.labeling import LABEL_TO_INDEX, LABELS
from marketpulse.ml.train import class_weights, predict_labels, predict_proba, to_xy, train_model

pytestmark = pytest.mark.unit

FAST_CONFIG = LightGBMConfig(
    objective="multiclass",
    learning_rate=0.2,
    num_leaves=15,
    max_depth=-1,
    min_data_in_leaf=5,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    lambda_l1=0.0,
    lambda_l2=0.0,
    num_boost_round=60,
    early_stopping_rounds=15,
    seed=42,
    verbosity=-1,
)


def _synthetic_frame(n: int, *, seed: int, signal_column: str = "roc_5m") -> pd.DataFrame:
    """Rows where ``signal_column`` deterministically decides the label --
    lets a test assert the model actually learned something, not just that
    it ran.
    """
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(size=n) for name in FEATURE_NAMES}
    signal = data[signal_column]
    labels = np.where(signal > 0.3, "UP", np.where(signal < -0.3, "DOWN", "STABLE"))
    frame = pd.DataFrame(data)
    frame["label"] = labels
    frame["feature_ts"] = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    frame["symbol"] = "BTC-USD"
    return frame


def _dataset(train_n: int, val_n: int, test_n: int = 10) -> Dataset:
    return Dataset(
        train=_synthetic_frame(train_n, seed=1),
        validation=_synthetic_frame(val_n, seed=2),
        test=_synthetic_frame(test_n, seed=3),
        theta=0.3,
        horizon=timedelta(minutes=15),
        feature_set_version=1,
        symbols=("BTC-USD",),
    )


# --- column / label ordering -------------------------------------------------


def test_to_xy_orders_features_by_registry_regardless_of_frame_column_order() -> None:
    frame = _synthetic_frame(20, seed=0)
    shuffled = frame[list(reversed(list(frame.columns)))]
    features, labels = to_xy(shuffled)
    assert list(features.columns) == list(FEATURE_NAMES)
    assert labels.dtype.kind in "iu"
    assert set(labels.tolist()) <= set(LABEL_TO_INDEX.values())


def test_to_xy_maps_labels_via_fixed_index() -> None:
    frame = pd.DataFrame({name: [0.0, 1.0, 2.0] for name in FEATURE_NAMES})
    frame["label"] = ["DOWN", "STABLE", "UP"]
    _, labels = to_xy(frame)
    assert labels.tolist() == [0, 1, 2]


# --- class weighting -----------------------------------------------------------


def test_class_weights_are_inverse_frequency() -> None:
    # 8 DOWN, 1 STABLE, 1 UP -- STABLE/UP should be weighted ~8x DOWN.
    y = np.array([0] * 8 + [1] + [2])
    weights = class_weights(y)
    down_weight = weights[y == 0][0]
    stable_weight = weights[y == 1][0]
    up_weight = weights[y == 2][0]
    assert stable_weight == pytest.approx(down_weight * 8, rel=0.05)
    assert up_weight == pytest.approx(down_weight * 8, rel=0.05)


def test_class_weights_handles_a_class_absent_from_this_split() -> None:
    y = np.array([1, 1, 1, 2, 2])  # class 0 (DOWN) never occurs
    weights = class_weights(y)
    assert np.all(np.isfinite(weights))
    assert len(weights) == len(y)


# --- training -------------------------------------------------------------------


def test_train_model_produces_a_valid_probability_simplex() -> None:
    dataset = _dataset(train_n=400, val_n=150)
    model = train_model(dataset, FAST_CONFIG)

    proba = predict_proba(model, dataset.test)
    assert proba.shape == (len(dataset.test), len(LABELS))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()

    labels = predict_labels(model, dataset.test)
    assert set(labels.tolist()) <= set(LABELS)


def test_train_model_learns_the_separable_signal() -> None:
    dataset = _dataset(train_n=600, val_n=200, test_n=300)
    model = train_model(dataset, FAST_CONFIG)

    predicted = predict_labels(model, dataset.test)
    accuracy = float((predicted == dataset.test["label"].to_numpy()).mean())
    # Chance on a 3-class problem is ~1/3; the signal column deterministically
    # decides the label, so a model that learned anything should clear it
    # comfortably without needing to be near-perfect.
    assert accuracy > 0.6


def test_train_model_sets_best_iteration_within_bounds() -> None:
    dataset = _dataset(train_n=400, val_n=150)
    model = train_model(dataset, FAST_CONFIG)
    assert 0 < model.best_iteration <= FAST_CONFIG.num_boost_round


def test_train_model_is_scale_invariant_across_wildly_different_feature_scales() -> None:
    """No scaler is fit anywhere in this module -- LightGBM is tree-based and
    splits on raw thresholds, so wildly different feature magnitudes must
    not break (or even affect) training."""
    dataset = _dataset(train_n=400, val_n=150)
    scaled_train = dataset.train.copy()
    scaled_train["roc_5m"] = scaled_train["roc_5m"] * 1_000_000.0
    scaled_val = dataset.validation.copy()
    scaled_val["roc_5m"] = scaled_val["roc_5m"] * 1_000_000.0
    scaled_test = dataset.test.copy()
    scaled_test["roc_5m"] = scaled_test["roc_5m"] * 1_000_000.0
    scaled_dataset = Dataset(
        train=scaled_train,
        validation=scaled_val,
        test=scaled_test,
        theta=dataset.theta,
        horizon=dataset.horizon,
        feature_set_version=dataset.feature_set_version,
        symbols=dataset.symbols,
    )

    model = train_model(scaled_dataset, FAST_CONFIG)
    predicted = predict_labels(model, scaled_dataset.test)
    accuracy = float((predicted == scaled_dataset.test["label"].to_numpy()).mean())
    assert accuracy > 0.6
