"""Drift statistics (Phase 6).

Every assertion here is against an independently-derived number — a
hand-computed PSI, ``scipy`` called directly for KS — rather than against
whatever the implementation happens to return. A drift metric that only
agrees with itself is not a metric.
"""

import math

import numpy as np
import pytest
from scipy import stats

from marketpulse.config import MonitoringSettings
from marketpulse.monitoring.drift import (
    METRIC_KS,
    METRIC_PSI,
    SEVERITY_MODERATE,
    SEVERITY_SIGNIFICANT,
    SEVERITY_STABLE,
    DriftResult,
    chi_square_statistic,
    classify_psi,
    compare_feature,
    correlated_breach,
    ks_statistic,
    population_stability_index,
    reference_samples_from_stats,
    worst_severity,
)

pytestmark = pytest.mark.unit

MONITORING = MonitoringSettings()


# --- PSI ------------------------------------------------------------------


def test_psi_of_identical_distributions_is_approximately_zero() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5_000).tolist()

    psi, size = population_stability_index(sample, sample, bins=10)

    assert psi == pytest.approx(0.0, abs=1e-9)
    assert size == 5_000


def test_psi_matches_a_hand_computed_value() -> None:
    """A binning we can do on paper.

    Reference is uniform on the 10 deciles of ``range(1000)``, so each bin
    holds exactly 10%. The live sample puts 20% in the first bin and
    distributes 80% evenly across the remaining nine (8.888...% each).

        PSI = (0.20 - 0.10) * ln(0.20/0.10)
            + 9 * (0.08889 - 0.10) * ln(0.08889/0.10)
    """
    reference = list(range(1000))

    live: list[float] = []
    live.extend(float(v) for v in range(0, 100))  # bin 0: doubled weight
    for start in range(100, 1000, 100):
        live.extend(float(v) for v in range(start, start + 89))

    psi, _ = population_stability_index(reference, live, bins=10)

    n = len(live)
    first = 100 / n
    rest = 89 / n
    expected = (first - 0.1) * math.log(first / 0.1) + 9 * (rest - 0.1) * math.log(rest / 0.1)

    assert psi == pytest.approx(expected, rel=0.02)


def test_psi_grows_with_a_real_shift() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, 5_000).tolist()
    mild = rng.normal(0.3, 1, 5_000).tolist()
    severe = rng.normal(3.0, 1, 5_000).tolist()

    psi_mild, _ = population_stability_index(reference, mild)
    psi_severe, _ = population_stability_index(reference, severe)

    assert 0 < psi_mild < psi_severe
    assert psi_severe > MONITORING.psi_significant_threshold


def test_psi_bins_on_the_reference_not_the_pooled_sample() -> None:
    # Binning on pooled data would let the live sample move the edges and
    # absorb exactly the shift PSI exists to detect. A large shift must stay
    # large no matter how big the live sample is.
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2_000).tolist()

    small_shift, _ = population_stability_index(reference, rng.normal(4, 1, 200).tolist())
    large_shift, _ = population_stability_index(reference, rng.normal(4, 1, 20_000).tolist())

    assert small_shift > 1.0
    assert large_shift > 1.0


def test_psi_handles_empty_and_constant_inputs_without_exploding() -> None:
    assert population_stability_index([], [1.0, 2.0]) == (0.0, 2)
    assert population_stability_index([1.0, 2.0], []) == (0.0, 0)
    # A constant reference has no spread to bin -- PSI is undefined, not
    # infinite. KS carries the signal for this case.
    psi, _ = population_stability_index([5.0] * 100, [9.0] * 100)
    assert psi == 0.0


def test_psi_ignores_nulls_rather_than_zero_filling_them() -> None:
    # Zero-filling would shift the distribution toward zero and manufacture
    # drift that is really just a cold start (CLAUDE.md rule #7).
    reference = [1.0, 1.1, 0.9, 1.05, 0.95] * 40
    with_nulls: list[float | None] = [*reference, None, None, None]

    clean_psi, clean_n = population_stability_index(reference, reference)
    null_psi, null_n = population_stability_index(reference, with_nulls)

    assert null_n == len(reference)  # the Nones did not become samples
    assert null_psi == pytest.approx(clean_psi, abs=1e-9)


def test_live_values_outside_the_reference_range_land_in_the_end_bins() -> None:
    # Not silently dropped: a live sample entirely beyond the reference range
    # is maximal drift, not "no data".
    reference = list(range(100))
    live = [500.0] * 100

    psi, size = population_stability_index(reference, live)

    assert size == 100
    assert psi > MONITORING.psi_significant_threshold


# --- severity -------------------------------------------------------------


@pytest.mark.parametrize(
    ("psi", "expected"),
    [
        (0.0, SEVERITY_STABLE),
        (0.05, SEVERITY_STABLE),
        (0.10, SEVERITY_STABLE),  # boundary belongs to the lower band
        (0.11, SEVERITY_MODERATE),
        (0.25, SEVERITY_MODERATE),  # boundary belongs to the lower band
        (0.26, SEVERITY_SIGNIFICANT),
        (3.0, SEVERITY_SIGNIFICANT),
    ],
)
def test_severity_bands(psi: float, expected: str) -> None:
    assert classify_psi(psi, moderate=0.10, significant=0.25) == expected


def test_worst_severity_picks_the_highest_band() -> None:
    results = [
        DriftResult("a", METRIC_PSI, 0.01, None, SEVERITY_STABLE, 10),
        DriftResult("b", METRIC_PSI, 0.30, None, SEVERITY_SIGNIFICANT, 10),
        DriftResult("c", METRIC_PSI, 0.15, None, SEVERITY_MODERATE, 10),
    ]
    assert worst_severity(results) == SEVERITY_SIGNIFICANT
    assert worst_severity([]) == SEVERITY_STABLE


# --- KS -------------------------------------------------------------------


def test_ks_statistic_matches_scipy_on_a_reference_sample() -> None:
    rng = np.random.default_rng(11)
    reference = rng.normal(0, 1, 500).tolist()
    live = rng.normal(0.5, 1.2, 400).tolist()

    statistic, p_value, size = ks_statistic(reference, live)
    expected = stats.ks_2samp(np.array(reference), np.array(live))

    assert statistic == pytest.approx(float(expected.statistic))
    assert p_value == pytest.approx(float(expected.pvalue))
    assert size == 400


def test_ks_on_identical_samples_is_zero_with_p_one() -> None:
    sample = [float(v) for v in range(200)]
    statistic, p_value, _ = ks_statistic(sample, sample)
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_ks_on_empty_input_is_the_no_evidence_answer() -> None:
    assert ks_statistic([], [1.0]) == (0.0, 1.0, 1)


# --- chi-square -----------------------------------------------------------


def test_chi_square_on_matching_categorical_shapes_is_near_zero() -> None:
    reference = {"DOWN": 300, "STABLE": 400, "UP": 300}
    live = {"DOWN": 30, "STABLE": 40, "UP": 30}

    statistic, p_value, size = chi_square_statistic(reference, live)

    assert statistic == pytest.approx(0.0, abs=1e-6)
    assert p_value > 0.99
    assert size == 100


def test_chi_square_detects_a_collapsed_class_distribution() -> None:
    reference = {"DOWN": 300, "STABLE": 400, "UP": 300}
    collapsed = {"DOWN": 0, "STABLE": 100, "UP": 0}

    statistic, p_value, _ = chi_square_statistic(reference, collapsed)

    assert statistic > 100
    assert p_value < 0.01


def test_chi_square_unions_categories_present_in_only_one_side() -> None:
    statistic, _, size = chi_square_statistic({"A": 10}, {"A": 5, "B": 5})
    assert size == 10
    assert statistic > 0


def test_chi_square_with_no_live_samples_is_the_no_evidence_answer() -> None:
    assert chi_square_statistic({"A": 10}, {}) == (0.0, 1.0, 0)


# --- correlated breach ----------------------------------------------------


def test_single_feature_drift_does_not_count_as_a_breach() -> None:
    # Single-feature drift is usually noise; alerting on it produces the
    # fatigue that gets real signals ignored.
    results = [
        DriftResult("a", METRIC_PSI, 0.9, None, SEVERITY_SIGNIFICANT, 100),
        DriftResult("b", METRIC_PSI, 0.01, None, SEVERITY_STABLE, 100),
        DriftResult("c", METRIC_PSI, 0.01, None, SEVERITY_STABLE, 100),
    ]
    breached, features = correlated_breach(results, min_features=3)
    assert breached is False
    assert features == ["a"]


def test_correlated_multi_feature_drift_is_a_breach() -> None:
    results = [
        DriftResult(name, METRIC_PSI, 0.9, None, SEVERITY_SIGNIFICANT, 100)
        for name in ("a", "b", "c")
    ]
    breached, features = correlated_breach(results, min_features=3)
    assert breached is True
    assert features == ["a", "b", "c"]


def test_both_metrics_breaching_for_one_feature_counts_as_one_feature() -> None:
    results = [
        DriftResult("a", METRIC_PSI, 0.9, None, SEVERITY_SIGNIFICANT, 100),
        DriftResult("a", METRIC_KS, 0.7, 0.001, SEVERITY_SIGNIFICANT, 100),
        DriftResult("b", METRIC_PSI, 0.9, None, SEVERITY_SIGNIFICANT, 100),
        DriftResult("b", METRIC_KS, 0.7, 0.001, SEVERITY_SIGNIFICANT, 100),
    ]
    breached, features = correlated_breach(results, min_features=3)
    assert breached is False
    assert features == ["a", "b"]


# --- compare_feature ------------------------------------------------------


def test_compare_feature_emits_psi_and_ks_with_a_shared_severity() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 1, 1_000).tolist()
    live = rng.normal(4, 1, 1_000).tolist()

    results = compare_feature("ma_5m", reference, live, monitoring=MONITORING)

    assert [r.metric_name for r in results] == [METRIC_PSI, METRIC_KS]
    # One feature never reads "stable by one metric, significant by another"
    # in the same window -- that looks like a bug to anyone on the dashboard.
    assert len({r.severity for r in results}) == 1
    assert results[0].severity == SEVERITY_SIGNIFICANT
    assert results[0].p_value is None  # PSI has no null hypothesis
    assert results[1].p_value is not None


def test_synthetic_drift_injection_crosses_the_threshold() -> None:
    """The Phase 6 exit criterion, at the statistic level."""
    rng = np.random.default_rng(42)
    reference = rng.normal(100.0, 5.0, 2_000).tolist()

    quiet = compare_feature(
        "ma_5m", reference, rng.normal(100.0, 5.0, 2_000).tolist(), monitoring=MONITORING
    )
    assert worst_severity(quiet) == SEVERITY_STABLE

    # Inject the drift: a 4-sigma mean shift.
    drifted = compare_feature(
        "ma_5m", reference, rng.normal(120.0, 5.0, 2_000).tolist(), monitoring=MONITORING
    )
    assert worst_severity(drifted) == SEVERITY_SIGNIFICANT
    assert drifted[0].metric_value > MONITORING.psi_significant_threshold


# --- reference reconstruction --------------------------------------------


def test_reference_samples_respect_the_stored_summary_and_are_reproducible() -> None:
    stats_map = {"ma_5m": {"mean": 10.0, "std": 2.0, "min": 4.0, "max": 16.0, "p50": 10.0}}

    first = reference_samples_from_stats(
        stats_map, "ma_5m", size=1_000, rng=np.random.default_rng(1)
    )
    second = reference_samples_from_stats(
        stats_map, "ma_5m", size=1_000, rng=np.random.default_rng(1)
    )

    assert first == second  # an unreproducible monitoring metric is not a metric
    assert min(first) >= 4.0 and max(first) <= 16.0
    assert float(np.mean(first)) == pytest.approx(10.0, abs=0.3)


def test_reference_samples_for_an_unknown_feature_are_empty_not_zeros() -> None:
    assert reference_samples_from_stats({}, "ma_5m", size=10, rng=np.random.default_rng(0)) == []


def test_zero_std_reference_reconstructs_as_a_constant() -> None:
    stats_map = {"x": {"mean": 7.0, "std": 0.0, "min": 7.0, "max": 7.0}}
    assert (
        reference_samples_from_stats(stats_map, "x", size=5, rng=np.random.default_rng(0))
        == [7.0] * 5
    )
