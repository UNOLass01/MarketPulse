"""LightGBM multiclass training: fits on ``dataset.train``, early-stops on
``dataset.validation`` logloss.

No scaler/normaliser anywhere in this module -- LightGBM is tree-based and
scale-invariant, and CLAUDE.md is explicit that fitting one here would be
adding a whole class of "scaler wasn't persisted" bugs for no benefit.
Hyperparameters come from :class:`marketpulse.ml.config.LightGBMConfig`,
never hardcoded, so every run is reproducible from its logged config alone.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.ml.config import LightGBMConfig
from marketpulse.ml.dataset import Dataset
from marketpulse.ml.labeling import INDEX_TO_LABEL, LABEL_TO_INDEX, LABELS


@dataclass(frozen=True)
class TrainedModel:
    """A fitted booster plus the metadata needed to predict on new frames."""

    booster: lgb.Booster
    best_iteration: int
    feature_names: tuple[str, ...]


def to_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split an assembled split-frame into feature matrix + integer label
    array, in the registry's canonical column order and the labeling
    module's fixed class-index order -- never incidental frame column order.
    """
    features = frame[list(FEATURE_NAMES)]
    labels = frame["label"].map(LABEL_TO_INDEX).to_numpy()
    return features, labels


def class_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample inverse-frequency weights so the majority class doesn't
    dominate the loss (phase-3 plan: "class weighting for imbalance").

    A class entirely absent from ``y`` gets a weight denominator of 1 rather
    than 0 -- it simply can't occur in this split, so there is no sample to
    actually receive that weight; the guard just avoids a division by zero.
    """
    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    counts[counts == 0] = 1.0
    weight_by_class = counts.sum() / (len(LABELS) * counts)
    return np.asarray(weight_by_class[y])


def train_model(dataset: Dataset, config: LightGBMConfig) -> TrainedModel:
    """Fit a LightGBM multiclass booster, early-stopping on validation logloss."""
    x_train, y_train = to_xy(dataset.train)
    x_val, y_val = to_xy(dataset.validation)

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=class_weights(y_train),
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        x_val,
        label=y_val,
        weight=class_weights(y_val),
        reference=train_set,
        free_raw_data=False,
    )

    params = config.booster_params(num_class=len(LABELS))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[val_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = booster.best_iteration or config.num_boost_round
    return TrainedModel(booster=booster, best_iteration=best_iteration, feature_names=FEATURE_NAMES)


def predict_proba(model: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    """Class probabilities in :data:`marketpulse.ml.labeling.LABELS` order."""
    features = frame[list(model.feature_names)]
    result = model.booster.predict(features, num_iteration=model.best_iteration)
    return np.asarray(result)


def predict_labels(model: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    """Argmax class label string per row."""
    proba = predict_proba(model, frame)
    indices = proba.argmax(axis=1)
    return np.array([INDEX_TO_LABEL[i] for i in indices])
