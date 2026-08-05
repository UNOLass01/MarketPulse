"""Baselines, metrics, and the promotion gate -- including the phase-3 plan's
required test: a deliberately worse candidate must not be promoted."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from marketpulse.ml.evaluate import (
    Metrics,
    PromotionResult,
    compute_metrics,
    decide_promotion,
    majority_baseline,
    persistence_baseline,
    random_stratified_baseline,
    uniform_brier,
)
from marketpulse.ml.labeling import LABEL_TO_INDEX, LABELS

pytestmark = pytest.mark.unit


def _one_hot_proba(labels: list[str]) -> np.ndarray:
    proba = np.zeros((len(labels), len(LABELS)))
    for i, label in enumerate(labels):
        proba[i, LABEL_TO_INDEX[label]] = 1.0
    return proba


def _metrics(
    *, macro_f1: float, recall: dict[str, float] | None = None, brier: float = 0.1
) -> Metrics:
    recall = recall or {label: 0.5 for label in LABELS}
    return Metrics(
        accuracy=macro_f1,
        macro_f1=macro_f1,
        precision={label: 0.5 for label in LABELS},
        recall=recall,
        support={label: 10 for label in LABELS},
        logloss=0.5,
        brier=brier,
        confusion_matrix={t: dict.fromkeys(LABELS, 0) for t in LABELS},
    )


# --- compute_metrics -------------------------------------------------------------


def test_compute_metrics_accuracy_and_macro_f1_match_sklearn() -> None:
    y_true = ["UP", "DOWN", "STABLE", "UP", "UP", "DOWN"]
    y_pred = ["UP", "DOWN", "UP", "UP", "STABLE", "DOWN"]
    proba = _one_hot_proba(y_pred)

    metrics = compute_metrics(y_true, y_pred, proba)

    assert metrics.accuracy == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics.macro_f1 == pytest.approx(
        f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    )


def test_compute_metrics_confusion_matrix_counts_are_correct() -> None:
    y_true = ["UP", "UP", "DOWN", "STABLE"]
    y_pred = ["UP", "DOWN", "DOWN", "STABLE"]
    proba = _one_hot_proba(y_pred)

    metrics = compute_metrics(y_true, y_pred, proba)

    assert metrics.confusion_matrix["UP"]["UP"] == 1
    assert metrics.confusion_matrix["UP"]["DOWN"] == 1
    assert metrics.confusion_matrix["DOWN"]["DOWN"] == 1
    assert metrics.confusion_matrix["STABLE"]["STABLE"] == 1
    assert metrics.confusion_matrix["DOWN"]["UP"] == 0


def test_compute_metrics_brier_matches_hand_computation() -> None:
    y_true = ["UP", "DOWN"]
    proba = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])  # columns: DOWN, STABLE, UP
    y_pred = ["UP", "DOWN"]

    metrics = compute_metrics(y_true, y_pred, proba)

    # Row 0: true UP -> one-hot [0,0,1]; sq diffs vs [0.2,0.3,0.5] = .04+.09+.25=.38
    # Row 1: true DOWN -> one-hot [1,0,0]; sq diffs vs [0.6,0.3,0.1] = .16+.09+.01=.26
    expected = (0.38 + 0.26) / 2
    assert metrics.brier == pytest.approx(expected)


def test_compute_metrics_perfect_predictions_have_zero_brier_and_logloss() -> None:
    y_true = ["UP", "DOWN", "STABLE"] * 5
    proba = _one_hot_proba(y_true)
    metrics = compute_metrics(y_true, y_true, proba)
    assert metrics.brier == pytest.approx(0.0, abs=1e-9)
    assert metrics.logloss == pytest.approx(0.0, abs=1e-6)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0


def test_uniform_brier_is_two_thirds_for_three_classes() -> None:
    assert uniform_brier() == pytest.approx(2 / 3)


def test_time_bucket_accuracy_isolates_a_bad_regime() -> None:
    n = 99  # divides evenly into 3 buckets of 33 so the bad regime aligns exactly
    y_true = ["UP"] * n
    # Correct everywhere except a contiguous bad regime in the middle third.
    y_pred = ["DOWN" if 33 <= i < 66 else "UP" for i in range(n)]
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"))

    metrics = compute_metrics(
        y_true, y_pred, _one_hot_proba(y_pred), timestamps=timestamps, time_buckets=3
    )

    assert metrics.time_bucket_accuracy is not None
    buckets = metrics.time_bucket_accuracy
    assert buckets["bucket_0"] == pytest.approx(1.0)
    assert buckets["bucket_1"] < 0.5
    assert buckets["bucket_2"] == pytest.approx(1.0)


# --- baselines ------------------------------------------------------------------


def test_majority_baseline_predicts_the_train_mode_with_full_confidence() -> None:
    train_labels = ["UP"] * 7 + ["DOWN"] * 2 + ["STABLE"] * 1
    result = majority_baseline(train_labels, test_size=5)
    assert (result.labels == "UP").all()
    assert np.allclose(result.proba[:, LABEL_TO_INDEX["UP"]], 1.0)
    assert np.allclose(result.proba.sum(axis=1), 1.0)


def test_random_stratified_baseline_is_deterministic_given_a_seed() -> None:
    train_labels = ["UP"] * 5 + ["DOWN"] * 3 + ["STABLE"] * 2
    first = random_stratified_baseline(train_labels, test_size=50, seed=7)
    second = random_stratified_baseline(train_labels, test_size=50, seed=7)
    assert (first.labels == second.labels).all()


def test_random_stratified_baseline_matches_train_distribution_at_scale() -> None:
    train_labels = ["UP"] * 50 + ["DOWN"] * 30 + ["STABLE"] * 20
    result = random_stratified_baseline(train_labels, test_size=20_000, seed=42)
    empirical = pd.Series(result.labels).value_counts(normalize=True)
    assert empirical["UP"] == pytest.approx(0.5, abs=0.02)
    assert empirical["DOWN"] == pytest.approx(0.3, abs=0.02)
    assert empirical["STABLE"] == pytest.approx(0.2, abs=0.02)


def test_persistence_baseline_uses_only_already_resolved_labels() -> None:
    horizon = timedelta(minutes=15)
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    # History rows at t=0 (label UP, resolves at t=15) and t=5 (label DOWN,
    # resolves at t=20).
    history_ts = [base, base + timedelta(minutes=5)]
    history_labels = ["UP", "DOWN"]

    # Query at t=10: neither history row has resolved yet (15 and 20 both
    # still in the future) -> falls back.
    # Query at t=16: only the t=0/UP row has resolved -> predicts UP.
    # Query at t=21: both resolved, t=5/DOWN is the most recent -> predicts DOWN.
    query_ts = [
        base + timedelta(minutes=10),
        base + timedelta(minutes=16),
        base + timedelta(minutes=21),
    ]

    result = persistence_baseline(
        history_ts, history_labels, query_ts, horizon=horizon, fallback_label="STABLE"
    )
    assert result.labels.tolist() == ["STABLE", "UP", "DOWN"]


# --- promotion gate --------------------------------------------------------------


def test_deliberately_worse_candidate_is_not_promoted() -> None:
    candidate = _metrics(macro_f1=0.30)
    baselines = {"majority": _metrics(macro_f1=0.40), "persistence": _metrics(macro_f1=0.35)}
    result = decide_promotion(candidate, baselines, incumbent=None)
    assert result.promote is False
    assert result.rejection_reason is not None
    assert "majority" in result.rejection_reason


def test_candidate_beating_everything_is_promoted() -> None:
    candidate = _metrics(macro_f1=0.55, brier=0.2)
    baselines = {"majority": _metrics(macro_f1=0.35), "persistence": _metrics(macro_f1=0.40)}
    incumbent = _metrics(macro_f1=0.50)
    result = decide_promotion(candidate, baselines, incumbent)
    assert result.promote is True
    assert result.rejection_reason is None


def test_first_ever_run_has_no_incumbent_to_beat() -> None:
    candidate = _metrics(macro_f1=0.45, brier=0.2)
    baselines = {"majority": _metrics(macro_f1=0.35)}
    result = decide_promotion(candidate, baselines, incumbent=None)
    assert result.promote is True


def test_incumbent_beats_candidate_blocks_promotion() -> None:
    candidate = _metrics(macro_f1=0.45, brier=0.2)
    baselines = {"majority": _metrics(macro_f1=0.35)}
    incumbent = _metrics(macro_f1=0.60)
    result = decide_promotion(candidate, baselines, incumbent)
    assert result.promote is False
    assert "incumbent" in (result.rejection_reason or "")


def test_per_class_collapse_blocks_promotion_even_with_good_macro_f1() -> None:
    recall = {"UP": 0.6, "STABLE": 0.6, "DOWN": 0.0}  # never once predicts DOWN correctly
    candidate = _metrics(macro_f1=0.55, recall=recall, brier=0.2)
    baselines = {"majority": _metrics(macro_f1=0.30)}
    result = decide_promotion(candidate, baselines, incumbent=None)
    assert result.promote is False
    assert "collapse" in (result.rejection_reason or "")


def test_calibration_sanity_blocks_a_miscalibrated_candidate() -> None:
    candidate = _metrics(macro_f1=0.55, brier=uniform_brier() + 0.01)
    baselines = {"majority": _metrics(macro_f1=0.30)}
    result = decide_promotion(candidate, baselines, incumbent=None)
    assert result.promote is False
    assert "calibration" in (result.rejection_reason or "")


def test_promotion_result_reasons_are_a_tuple_not_a_shared_mutable_default() -> None:
    a = PromotionResult(promote=True)
    b = PromotionResult(promote=False, rejection_reasons=("x",))
    assert a.rejection_reasons == ()
    assert b.rejection_reasons == ("x",)
