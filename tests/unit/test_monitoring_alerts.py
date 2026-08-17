"""Alert decision logic (Phase 6).

``decide`` is pure, so the two rules that are easiest to get subtly wrong —
"a single spike must not fire" and "the same condition fires once" — are
tested exhaustively here rather than inferred from an integration run.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from marketpulse.config import MonitoringSettings
from marketpulse.ml.labeling import LABEL_DOWN, LABEL_STABLE, LABEL_UP
from marketpulse.monitoring.alerts import (
    ACTION_ACCUMULATE,
    ACTION_FIRE,
    ACTION_NONE,
    ACTION_RESOLVE,
    ACTION_SUPPRESS,
    ALL_RULES,
    RULE_ACCURACY_DEGRADED,
    RULE_FEATURE_DRIFT,
    RULE_PREDICTION_DISTRIBUTION,
    AlertState,
    Breach,
    decide,
    evaluate_accuracy,
    evaluate_drift,
    evaluate_prediction_distribution,
)
from marketpulse.monitoring.drift import METRIC_PSI, SEVERITY_SIGNIFICANT, DriftResult
from marketpulse.monitoring.performance import build_slice

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
SUPPRESSION = timedelta(hours=6)
REPO_ROOT = Path(__file__).resolve().parents[2]

MONITORING = MonitoringSettings()


def a_breach(dedup_key: str = "feature_drift") -> Breach:
    return Breach(rule=RULE_FEATURE_DRIFT, dedup_key=dedup_key, details={"x": 1})


# --- every alert names its runbook ---------------------------------------


def test_every_rule_names_a_runbook_that_exists_on_disk() -> None:
    # An alert with no attached action is just anxiety. A rule pointing at a
    # renamed or missing runbook is the same thing with extra steps.
    for rule in ALL_RULES:
        assert rule.runbook, f"{rule.name} has no runbook"
        assert (REPO_ROOT / rule.runbook).is_file(), f"{rule.name} -> missing {rule.runbook}"


def test_rule_names_and_severities_are_distinct_and_valid() -> None:
    assert len({rule.name for rule in ALL_RULES}) == len(ALL_RULES)
    assert {rule.severity for rule in ALL_RULES} <= {"info", "warning", "critical"}


# --- sustained breach -----------------------------------------------------


def test_a_single_spike_accumulates_but_does_not_fire() -> None:
    decision = decide(None, a_breach(), now=NOW, sustained_evaluations=2, suppression=SUPPRESSION)
    assert decision.action == ACTION_ACCUMULATE
    assert decision.should_fire is False
    assert decision.consecutive_breaches == 1


def test_n_consecutive_breaches_fire() -> None:
    first = decide(None, a_breach(), now=NOW, sustained_evaluations=2, suppression=SUPPRESSION)
    assert first.should_fire is False

    state = AlertState("feature_drift", first.consecutive_breaches, NOW, None)
    second = decide(
        state,
        a_breach(),
        now=NOW + timedelta(hours=6),
        sustained_evaluations=2,
        suppression=SUPPRESSION,
    )

    assert second.action == ACTION_FIRE
    assert second.should_fire is True
    assert second.consecutive_breaches == 2


@pytest.mark.parametrize("threshold", [1, 2, 3, 5])
def test_firing_happens_exactly_at_the_configured_threshold(threshold: int) -> None:
    state: AlertState | None = None
    fired_at_evaluation = None

    for evaluation in range(1, threshold + 2):
        decision = decide(
            state,
            a_breach(),
            now=NOW + timedelta(minutes=evaluation),
            sustained_evaluations=threshold,
            suppression=SUPPRESSION,
        )
        if decision.should_fire and fired_at_evaluation is None:
            fired_at_evaluation = evaluation
        state = AlertState(
            "feature_drift",
            decision.consecutive_breaches,
            NOW,
            NOW if decision.should_fire else None,
        )

    assert fired_at_evaluation == threshold


def test_a_cleared_condition_resets_the_counter() -> None:
    # The counter is *consecutive*: an intermittent breach must not
    # accumulate its way to an alert over a week.
    state = AlertState("feature_drift", 1, NOW, None)

    cleared = decide(state, None, now=NOW, sustained_evaluations=2, suppression=SUPPRESSION)
    assert cleared.action == ACTION_RESOLVE
    assert cleared.consecutive_breaches == 0

    # With the alert resolved, the next breach starts from one again.
    restart = decide(None, a_breach(), now=NOW, sustained_evaluations=2, suppression=SUPPRESSION)
    assert restart.consecutive_breaches == 1
    assert restart.should_fire is False


# --- dedup / suppression --------------------------------------------------


def test_the_same_condition_twice_produces_one_alert() -> None:
    fired = AlertState("feature_drift", 2, NOW, fired_at=NOW)

    again = decide(
        fired,
        a_breach(),
        now=NOW + timedelta(minutes=30),
        sustained_evaluations=2,
        suppression=SUPPRESSION,
    )

    assert again.action == ACTION_SUPPRESS
    assert again.should_fire is False


def test_a_still_broken_condition_refires_after_the_suppression_window() -> None:
    # A long-running incident must not silently fall off the radar.
    fired = AlertState("feature_drift", 2, NOW, fired_at=NOW)

    later = decide(
        fired,
        a_breach(),
        now=NOW + SUPPRESSION + timedelta(minutes=1),
        sustained_evaluations=2,
        suppression=SUPPRESSION,
    )

    assert later.action == ACTION_FIRE
    assert later.should_fire is True


def test_suppression_boundary_is_exclusive_of_the_window_itself() -> None:
    fired = AlertState("feature_drift", 2, NOW, fired_at=NOW)
    at_boundary = decide(
        fired,
        a_breach(),
        now=NOW + SUPPRESSION,
        sustained_evaluations=2,
        suppression=SUPPRESSION,
    )
    assert at_boundary.should_fire is True


def test_no_breach_and_nothing_open_is_a_no_op() -> None:
    decision = decide(None, None, now=NOW, sustained_evaluations=2, suppression=SUPPRESSION)
    assert decision.action == ACTION_NONE
    assert decision.should_fire is False


# --- rule evaluation ------------------------------------------------------


def _drift(names: list[str], severity: str = SEVERITY_SIGNIFICANT) -> list[DriftResult]:
    return [DriftResult(n, METRIC_PSI, 0.9, None, severity, 500) for n in names]


def test_drift_rule_requires_correlated_features() -> None:
    assert evaluate_drift(_drift(["a"]), MONITORING) is None
    assert evaluate_drift(_drift(["a", "b"]), MONITORING) is None

    breach = evaluate_drift(_drift(["a", "b", "c"]), MONITORING)
    assert breach is not None
    assert breach.rule is RULE_FEATURE_DRIFT
    assert breach.details["feature_count"] == 3
    assert breach.details["drifted_features"] == ["a", "b", "c"]


def test_accuracy_rule_ignores_slices_with_too_few_resolved_outcomes() -> None:
    # Early accuracy on a handful of samples swings for reasons that have
    # nothing to do with the model, so it is stored but never fired on.
    thin = build_slice("1", [LABEL_UP] * 5, [LABEL_DOWN] * 5, window_start=NOW, window_end=NOW)
    assert thin.accuracy == 0.0
    assert thin.resolved_count < MONITORING.performance_min_resolved
    assert evaluate_accuracy([thin], MONITORING) is None


def test_accuracy_rule_fires_on_a_well_sampled_degraded_slice() -> None:
    n = MONITORING.performance_min_resolved + 10
    degraded = build_slice("4", [LABEL_UP] * n, [LABEL_DOWN] * n, window_start=NOW, window_end=NOW)

    breach = evaluate_accuracy([degraded], MONITORING)

    assert breach is not None
    assert breach.rule is RULE_ACCURACY_DEGRADED
    # Keyed by version: a new promotion is a genuinely new condition.
    assert breach.dedup_key == "model_accuracy_degraded:4"
    assert breach.details["accuracy"] == pytest.approx(0.0)


def test_accuracy_rule_stays_quiet_on_a_healthy_slice() -> None:
    n = MONITORING.performance_min_resolved + 10
    healthy = build_slice("4", [LABEL_UP] * n, [LABEL_UP] * n, window_start=NOW, window_end=NOW)
    assert evaluate_accuracy([healthy], MONITORING) is None


def test_prediction_distribution_rule_fires_on_a_collapse() -> None:
    n = 100
    collapsed = build_slice(
        "5",
        [LABEL_UP] * n,
        [LABEL_DOWN] * n,  # the model predicts one class for everything
        window_start=NOW,
        window_end=NOW,
    )
    prior = {LABEL_DOWN: 0.3, LABEL_STABLE: 0.4, LABEL_UP: 0.3}

    breach = evaluate_prediction_distribution([collapsed], prior, MONITORING)

    assert breach is not None
    assert breach.rule is RULE_PREDICTION_DISTRIBUTION
    assert breach.details["shift"] > MONITORING.prediction_distribution_max_shift


def test_prediction_distribution_rule_is_quiet_when_the_mix_matches_the_prior() -> None:
    predicted = ([LABEL_DOWN] * 30) + ([LABEL_STABLE] * 40) + ([LABEL_UP] * 30)
    matching = build_slice("5", predicted, predicted, window_start=NOW, window_end=NOW)
    prior = {LABEL_DOWN: 0.3, LABEL_STABLE: 0.4, LABEL_UP: 0.3}

    assert evaluate_prediction_distribution([matching], prior, MONITORING) is None


def test_prediction_distribution_rule_needs_a_prior_to_compare_against() -> None:
    # No prior means no comparison, not a breach against an assumed uniform.
    slice_ = build_slice("5", [LABEL_UP] * 100, [LABEL_UP] * 100, window_start=NOW, window_end=NOW)
    assert evaluate_prediction_distribution([slice_], {}, MONITORING) is None
