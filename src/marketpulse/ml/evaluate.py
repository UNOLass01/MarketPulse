"""Naive baselines, evaluation metrics, and the promotion gate.

Pure functions operating on arrays of predictions/probabilities -- nothing
here touches MLflow or Postgres. ``ml.pipeline`` is the only caller that
resolves an incumbent model and passes its predictions in; this module just
computes metrics and compares them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from marketpulse.ml.labeling import LABEL_TO_INDEX, LABELS

# --- metrics ------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    precision: dict[str, float]
    recall: dict[str, float]
    support: dict[str, int]
    logloss: float
    brier: float
    confusion_matrix: dict[str, dict[str, int]]
    time_bucket_accuracy: dict[str, float] | None = None


def _multiclass_brier(y_true_idx: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between one-hot true labels and predicted
    probabilities, summed over classes -- the standard multiclass Brier
    score (sklearn only ships a binary ``brier_score_loss``).
    """
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def uniform_brier(num_classes: int = len(LABELS)) -> float:
    """Brier score of a baseline that always predicts a uniform ``1/K`` per
    class -- the calibration sanity floor a candidate's probabilities must
    beat (algebraically ``(K - 1) / K``, independent of the data).
    """
    return (num_classes - 1) / num_classes


def _bucket_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.Series, buckets: int
) -> dict[str, float]:
    order = np.argsort(timestamps.to_numpy())
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    result: dict[str, float] = {}
    for bucket_index, positions in enumerate(np.array_split(np.arange(len(sorted_true)), buckets)):
        if len(positions) == 0:
            continue
        accuracy = float((sorted_true[positions] == sorted_pred[positions]).mean())
        result[f"bucket_{bucket_index}"] = accuracy
    return result


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: np.ndarray,
    *,
    timestamps: pd.Series | None = None,
    time_buckets: int = 5,
) -> Metrics:
    """All headline metrics for one set of predictions against ground truth.

    ``y_proba`` must be shaped ``(n, len(LABELS))`` with columns in
    :data:`marketpulse.ml.labeling.LABELS` order.
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    y_true_idx = np.array([LABEL_TO_INDEX[label] for label in y_true_arr])

    precision, recall, _, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(LABELS), zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(LABELS))
    cm_dict = {
        true_label: {pred_label: int(cm[i][j]) for j, pred_label in enumerate(LABELS)}
        for i, true_label in enumerate(LABELS)
    }

    bucket_accuracy = None
    if timestamps is not None:
        bucket_accuracy = _bucket_accuracy(y_true_arr, y_pred_arr, timestamps, time_buckets)

    return Metrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        macro_f1=float(
            f1_score(y_true_arr, y_pred_arr, labels=list(LABELS), average="macro", zero_division=0)
        ),
        precision=dict(zip(LABELS, precision.tolist(), strict=True)),
        recall=dict(zip(LABELS, recall.tolist(), strict=True)),
        support=dict(zip(LABELS, support.tolist(), strict=True)),
        logloss=float(log_loss(y_true_idx, y_proba, labels=list(range(len(LABELS))))),
        brier=_multiclass_brier(y_true_idx, y_proba),
        confusion_matrix=cm_dict,
        time_bucket_accuracy=bucket_accuracy,
    )


# --- baselines ------------------------------------------------------------------


@dataclass(frozen=True)
class BaselinePrediction:
    labels: np.ndarray
    proba: np.ndarray


def majority_class(labels: Sequence[str]) -> str:
    counts = pd.Series(list(labels)).value_counts()
    if counts.empty:
        raise ValueError("cannot compute a majority class from an empty label series")
    return str(counts.idxmax())


def majority_baseline(train_labels: Sequence[str], test_size: int) -> BaselinePrediction:
    """Always predict the training set's most frequent class."""
    majority = majority_class(train_labels)
    labels = np.full(test_size, majority)
    proba = np.zeros((test_size, len(LABELS)))
    proba[:, LABEL_TO_INDEX[majority]] = 1.0
    return BaselinePrediction(labels=labels, proba=proba)


def random_stratified_baseline(
    train_labels: Sequence[str], test_size: int, *, seed: int
) -> BaselinePrediction:
    """Sample a label per row from the training set's own class distribution.

    The probability row for every prediction *is* that distribution -- it's
    the actual generating model this baseline simulates, not a one-hot
    point guess.
    """
    distribution = (
        pd.Series(list(train_labels)).value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
    )
    probs = distribution.to_numpy()
    rng = np.random.default_rng(seed)
    labels = rng.choice(np.array(LABELS), size=test_size, p=probs)
    proba = np.tile(probs, (test_size, 1))
    return BaselinePrediction(labels=labels, proba=proba)


def persistence_baseline(
    history_ts: Sequence[pd.Timestamp],
    history_labels: Sequence[str],
    query_ts: Sequence[pd.Timestamp],
    *,
    horizon: timedelta,
    fallback_label: str,
) -> BaselinePrediction:
    """Predict each query timestamp's label as the most recently *resolved*
    true label as of that time.

    Naively repeating the immediately preceding row's label would itself be
    look-ahead: that row's own outcome isn't knowable until ``H`` after it.
    A history row's label only becomes usable once its own ``ts + horizon``
    has passed, so this is an as-of-backward join on the *resolution* time,
    not on ``history_ts`` directly. ``query_ts`` must already be
    chronologically sorted (true of every dataset.py split).
    """
    history = pd.DataFrame(
        {
            "resolved_ts": pd.to_datetime(list(history_ts)) + horizon,
            "label": list(history_labels),
        }
    ).sort_values("resolved_ts")
    probe = pd.DataFrame({"query_ts": pd.to_datetime(list(query_ts))})
    joined = pd.merge_asof(
        probe, history, left_on="query_ts", right_on="resolved_ts", direction="backward"
    )
    labels = joined["label"].fillna(fallback_label).to_numpy()
    proba = np.zeros((len(labels), len(LABELS)))
    for row_index, label in enumerate(labels):
        proba[row_index, LABEL_TO_INDEX[label]] = 1.0
    return BaselinePrediction(labels=labels, proba=proba)


# --- promotion gate ------------------------------------------------------------


@dataclass(frozen=True)
class PromotionResult:
    promote: bool
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rejection_reason(self) -> str | None:
        """Single joined string for the ``training_runs.rejection_reason`` column."""
        return "; ".join(self.rejection_reasons) if self.rejection_reasons else None


def decide_promotion(
    candidate: Metrics,
    baselines: Mapping[str, Metrics],
    incumbent: Metrics | None,
    *,
    min_class_recall: float = 0.05,
) -> PromotionResult:
    """Promotion gate: beat every naive baseline on macro-F1 AND beat/match
    the incumbent AND no per-class collapse AND pass calibration sanity.
    Otherwise the run stays in Staging with every failed criterion recorded
    (phase-3 plan: "record rejections too").
    """
    reasons: list[str] = []

    for name, baseline in baselines.items():
        if candidate.macro_f1 <= baseline.macro_f1:
            reasons.append(
                f"macro-F1 {candidate.macro_f1:.4f} does not beat baseline "
                f"'{name}' ({baseline.macro_f1:.4f})"
            )

    if incumbent is not None and candidate.macro_f1 < incumbent.macro_f1:
        reasons.append(
            f"macro-F1 {candidate.macro_f1:.4f} does not beat/match incumbent "
            f"({incumbent.macro_f1:.4f})"
        )

    collapsed = [label for label in LABELS if candidate.recall.get(label, 0.0) < min_class_recall]
    if collapsed:
        reasons.append(f"per-class collapse: recall below {min_class_recall:.0%} for {collapsed}")

    floor = uniform_brier()
    if candidate.brier >= floor:
        reasons.append(
            f"calibration sanity failed: Brier {candidate.brier:.4f} is no better than "
            f"the uniform-prior floor ({floor:.4f})"
        )

    return PromotionResult(promote=not reasons, rejection_reasons=tuple(reasons))
