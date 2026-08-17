"""Performance attribution metrics (Phase 6).

The horizon-lag rule is asserted at the SQL layer in
``tests/integration/test_serving_storage.py`` (it is a query predicate, and a
fake would only prove the fake works). What is covered here is everything
downstream of it: the confusion matrix, the per-class and macro F1, the
version slicing across a promotion boundary, and the class-distribution
shift that fires before accuracy can.
"""

from datetime import UTC, datetime, timedelta

import pytest

from marketpulse.ml.labeling import LABEL_DOWN, LABEL_STABLE, LABEL_UP, LABELS
from marketpulse.monitoring.performance import (
    accuracy_of,
    build_slice,
    class_distribution,
    confusion_matrix,
    distribution_shift,
    macro_f1,
    per_class_f1,
)

pytestmark = pytest.mark.unit

WINDOW_START = datetime(2026, 8, 17, 0, 0, tzinfo=UTC)
WINDOW_END = WINDOW_START + timedelta(hours=24)


# --- confusion matrix -----------------------------------------------------


def test_confusion_matrix_is_indexed_actual_then_predicted() -> None:
    matrix = confusion_matrix(
        actual=[LABEL_UP, LABEL_UP, LABEL_DOWN],
        predicted=[LABEL_UP, LABEL_DOWN, LABEL_DOWN],
    )
    assert matrix[LABEL_UP][LABEL_UP] == 1
    assert matrix[LABEL_UP][LABEL_DOWN] == 1
    assert matrix[LABEL_DOWN][LABEL_DOWN] == 1


def test_confusion_matrix_covers_the_full_label_set_even_when_unseen() -> None:
    # "Never predicted UP" is a finding, not an absence -- it must show as an
    # explicit row/column of zeros rather than vanishing.
    matrix = confusion_matrix([LABEL_DOWN], [LABEL_DOWN])
    assert set(matrix) == set(LABELS)
    assert all(set(row) == set(LABELS) for row in matrix.values())
    assert matrix[LABEL_UP][LABEL_UP] == 0


def test_mismatched_lengths_raise_rather_than_silently_truncating() -> None:
    # ``zip(..., strict=True)`` is what enforces this -- a silent truncation
    # would drop real outcomes out of the accuracy denominator.
    with pytest.raises(ValueError, match="shorter"):
        confusion_matrix([LABEL_UP, LABEL_DOWN], [LABEL_UP])


# --- metrics --------------------------------------------------------------


def test_accuracy_matches_a_hand_count() -> None:
    actual = [LABEL_UP, LABEL_UP, LABEL_DOWN, LABEL_STABLE]
    predicted = [LABEL_UP, LABEL_DOWN, LABEL_DOWN, LABEL_STABLE]
    assert accuracy_of(confusion_matrix(actual, predicted)) == pytest.approx(0.75)


def test_perfect_predictions_score_one_across_the_board() -> None:
    actual = [LABEL_UP, LABEL_DOWN, LABEL_STABLE] * 5
    matrix = confusion_matrix(actual, actual)
    assert accuracy_of(matrix) == pytest.approx(1.0)
    assert macro_f1(matrix) == pytest.approx(1.0)


def test_per_class_f1_matches_a_hand_computed_value() -> None:
    # UP: 2 true positives, 1 false positive, 1 false negative.
    #   precision = 2/3, recall = 2/3, F1 = 2/3
    actual = [LABEL_UP, LABEL_UP, LABEL_UP, LABEL_DOWN]
    predicted = [LABEL_UP, LABEL_UP, LABEL_DOWN, LABEL_UP]

    scores = per_class_f1(confusion_matrix(actual, predicted))

    assert scores[LABEL_UP] == pytest.approx(2 / 3)


def test_a_class_with_no_instances_scores_zero_rather_than_being_dropped() -> None:
    # Omitting it would quietly raise the macro average by shrinking the
    # denominator.
    matrix = confusion_matrix([LABEL_DOWN] * 4, [LABEL_DOWN] * 4)
    scores = per_class_f1(matrix)

    assert scores[LABEL_DOWN] == pytest.approx(1.0)
    assert scores[LABEL_UP] == 0.0
    assert scores[LABEL_STABLE] == 0.0
    assert macro_f1(matrix) == pytest.approx(1 / 3)


def test_empty_matrix_is_zero_accuracy_not_a_division_error() -> None:
    assert accuracy_of(confusion_matrix([], [])) == 0.0


# --- class distribution ---------------------------------------------------


def test_class_distribution_covers_every_label() -> None:
    dist = class_distribution([LABEL_UP, LABEL_UP, LABEL_DOWN, LABEL_STABLE])
    assert dist == {
        LABEL_DOWN: pytest.approx(0.25),
        LABEL_STABLE: pytest.approx(0.25),
        LABEL_UP: pytest.approx(0.5),
    }


def test_empty_distribution_is_zeros_not_an_error() -> None:
    assert class_distribution([]) == dict.fromkeys(LABELS, 0.0)


def test_distribution_shift_is_zero_for_identical_mixes() -> None:
    prior = {LABEL_DOWN: 0.3, LABEL_STABLE: 0.4, LABEL_UP: 0.3}
    assert distribution_shift(prior, prior) == pytest.approx(0.0)


def test_distribution_shift_of_a_total_collapse_is_one() -> None:
    # The failure mode this exists to catch: the model starts predicting one
    # class for everything, which usually means broken features.
    prior = {LABEL_DOWN: 0.0, LABEL_STABLE: 0.0, LABEL_UP: 1.0}
    collapsed = {LABEL_DOWN: 1.0, LABEL_STABLE: 0.0, LABEL_UP: 0.0}
    assert distribution_shift(collapsed, prior) == pytest.approx(1.0)


def test_distribution_shift_matches_hand_computed_total_variation() -> None:
    prior = {LABEL_DOWN: 0.3, LABEL_STABLE: 0.4, LABEL_UP: 0.3}
    live = {LABEL_DOWN: 0.1, LABEL_STABLE: 0.6, LABEL_UP: 0.3}
    # 0.5 * (|0.1-0.3| + |0.6-0.4| + |0.3-0.3|) = 0.2
    assert distribution_shift(live, prior) == pytest.approx(0.2)


# --- slicing by model version --------------------------------------------


def test_slice_carries_every_metric_and_its_window() -> None:
    actual = [LABEL_UP, LABEL_UP, LABEL_DOWN, LABEL_STABLE]
    predicted = [LABEL_UP, LABEL_DOWN, LABEL_DOWN, LABEL_STABLE]

    result = build_slice("7", actual, predicted, window_start=WINDOW_START, window_end=WINDOW_END)

    assert result.model_version == "7"
    assert result.resolved_count == 4
    assert result.accuracy == pytest.approx(0.75)
    assert set(result.per_class_f1) == set(LABELS)
    assert result.window_start == WINDOW_START


def test_slicing_separates_two_model_versions_across_a_promotion() -> None:
    """The persuasive artifact: an accuracy step at a promotion boundary.

    Unsliced, these two would average to 0.60 and the step would vanish
    entirely — which is exactly what makes an unsliced rolling accuracy
    misleading around every promotion.
    """
    old_actual = [LABEL_UP] * 10
    old_predicted = [LABEL_UP] * 2 + [LABEL_DOWN] * 8  # 0.2

    new_actual = [LABEL_UP] * 10
    new_predicted = [LABEL_UP] * 10  # 1.0

    old = build_slice(
        "1", old_actual, old_predicted, window_start=WINDOW_START, window_end=WINDOW_END
    )
    new = build_slice(
        "2", new_actual, new_predicted, window_start=WINDOW_START, window_end=WINDOW_END
    )

    assert old.accuracy == pytest.approx(0.2)
    assert new.accuracy == pytest.approx(1.0)

    combined = build_slice(
        "mixed",
        old_actual + new_actual,
        old_predicted + new_predicted,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )
    assert combined.accuracy == pytest.approx(0.6)
    assert old.accuracy < combined.accuracy < new.accuracy
