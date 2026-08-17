"""Distribution drift: PSI, two-sample KS, and chi-square.

The statistics themselves are pure functions of two arrays — no database, no
model, no clock — so every threshold and edge case below is testable against
a hand-computed number. :func:`run_drift_monitoring` is the only thing here
that touches Postgres.

**What "drift" means here.** The live window is compared against the
reference snapshot taken from the *training split of the Production model*
(``model_versions.reference_feature_stats``, written on promotion by
``ml.pipeline``). Comparing live-vs-yesterday would measure change; this
measures divergence from what the deployed model actually learned, which is
the only version of the question that implies an action.

**Severity thresholds** (<0.1 stable, 0.1-0.25 moderate, >0.25 significant)
are the conventional PSI cuts. They are convention, not law — they live in
``MonitoringSettings`` so a genuinely noisier feature set can be retuned
without editing this module.

**A caveat worth keeping in view** (phase-6 plan): drift usually means a
broken pipeline, not a real regime shift. Before believing the model has
degraded, check whether the producer stalled or the consumer is behind.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import MonitoringSettings
from marketpulse.features.registry import FEATURE_SET_VERSION
from marketpulse.logging import get_logger
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.drift import record_drift_metric
from marketpulse.storage.repositories.features import list_feature_rows_in_range
from marketpulse.storage.repositories.ml_registry import get_current_production_version
from marketpulse.storage.repositories.symbols import list_symbols

logger = get_logger(__name__)

SEVERITY_STABLE = "stable"
SEVERITY_MODERATE = "moderate"
SEVERITY_SIGNIFICANT = "significant"

METRIC_PSI = "psi"
METRIC_KS = "ks"
METRIC_CHI2 = "chi2"

#: Severity order, used to reduce a set of per-feature severities to the
#: worst one. Defined once rather than compared ad hoc at each call site.
SEVERITY_RANK = {SEVERITY_STABLE: 0, SEVERITY_MODERATE: 1, SEVERITY_SIGNIFICANT: 2}

#: Replaces a zero bin proportion before the log in PSI. Without it a single
#: empty bin makes the whole statistic infinite, which is not "infinitely
#: drifted" -- it is "this bin had no samples", a very different claim. The
#: value is the conventional one; it caps a single empty bin's contribution
#: rather than letting it dominate.
_PSI_EPSILON = 1e-6


@dataclass(frozen=True)
class DriftResult:
    feature_name: str
    metric_name: str
    metric_value: float
    p_value: float | None
    severity: str
    sample_size: int

    @property
    def breached(self) -> bool:
        return self.severity != SEVERITY_STABLE


def classify_psi(psi: float, *, moderate: float, significant: float) -> str:
    """Map a PSI value onto a severity band.

    Boundaries are inclusive of the *lower* band: exactly 0.1 is still
    stable, exactly 0.25 is still moderate. Arbitrary but fixed — the point
    of writing it down is that the same number never lands in two bands on
    two different code paths.
    """
    if psi > significant:
        return SEVERITY_SIGNIFICANT
    if psi > moderate:
        return SEVERITY_MODERATE
    return SEVERITY_STABLE


def _clean(values: Sequence[float | None]) -> np.ndarray:
    """Drop nulls and non-finite values.

    Nulls are dropped, never zero-filled (CLAUDE.md rule #7): a feature that
    was null because of insufficient history has no value, and substituting
    0.0 would shift the distribution toward zero and manufacture drift that
    is really just a cold start.
    """
    array = np.asarray([v for v in values if v is not None], dtype=float)
    return array[np.isfinite(array)]


def population_stability_index(
    reference: Sequence[float | None],
    live: Sequence[float | None],
    *,
    bins: int = 10,
) -> tuple[float, int]:
    """PSI of ``live`` against ``reference``. Returns ``(psi, live_sample_size)``.

    Bin edges come from the *reference* distribution's quantiles, not from
    the combined sample. This is the part that is easy to get wrong: binning
    on the pooled data would let the live sample move the edges, which
    absorbs exactly the shift PSI is supposed to detect and reports a
    reassuring number while the world moves.

    Identical distributions give 0.0 (up to the epsilon floor), which is the
    property worth asserting in a test.
    """
    ref = _clean(reference)
    obs = _clean(live)
    if ref.size == 0 or obs.size == 0:
        return 0.0, int(obs.size)

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant reference feature has no spread to bin. Any live spread
        # is a real change, but PSI is undefined here -- report 0.0 and let
        # the KS test (which handles this fine) carry the signal.
        return 0.0, int(obs.size)

    # Open the outer edges so live values beyond the reference range land in
    # the end bins instead of being silently dropped by np.histogram.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    obs_counts, _ = np.histogram(obs, bins=edges)

    ref_frac = np.maximum(ref_counts / ref.size, _PSI_EPSILON)
    obs_frac = np.maximum(obs_counts / obs.size, _PSI_EPSILON)

    psi = float(np.sum((obs_frac - ref_frac) * np.log(obs_frac / ref_frac)))
    return psi, int(obs.size)


def ks_statistic(
    reference: Sequence[float | None], live: Sequence[float | None]
) -> tuple[float, float, int]:
    """Two-sample Kolmogorov-Smirnov. Returns ``(statistic, p_value, n_live)``.

    Complements PSI rather than duplicating it: PSI is a binned divergence
    with conventional cutoffs, KS is a distribution-free test with an actual
    p-value. On a large live window KS will flag differences too small to
    matter, which is precisely why alerting keys off PSI severity and KS is
    stored as supporting evidence.
    """
    ref = _clean(reference)
    obs = _clean(live)
    if ref.size == 0 or obs.size == 0:
        return 0.0, 1.0, int(obs.size)
    result = stats.ks_2samp(ref, obs)
    return float(result.statistic), float(result.pvalue), int(obs.size)


def chi_square_statistic(
    reference_counts: Mapping[str, int], live_counts: Mapping[str, int]
) -> tuple[float, float, int]:
    """Chi-square goodness-of-fit for a categorical distribution.

    Used for label/class mixes (the predicted-class distribution check), not
    for the continuous features. Categories present in either mapping are
    unioned so a category that appears only in the live data is compared
    against an expected count of ~0 rather than being dropped.
    """
    categories = sorted(set(reference_counts) | set(live_counts))
    observed = np.array([live_counts.get(c, 0) for c in categories], dtype=float)
    expected = np.array([reference_counts.get(c, 0) for c in categories], dtype=float)

    n_live = int(observed.sum())
    if n_live == 0 or expected.sum() == 0 or categories == []:
        return 0.0, 1.0, n_live

    # Floor *before* rescaling, not after: a category absent from the
    # reference would otherwise get an expected count of 0 (an infinite
    # contribution), and flooring afterwards would push the expected total
    # off the observed total, which scipy rejects outright.
    expected = np.maximum(expected, _PSI_EPSILON)
    # Rescale expected to the live sample size: chi-square compares shapes,
    # and two windows of different length are not otherwise comparable.
    expected = expected / expected.sum() * n_live

    result = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(result.statistic), float(result.pvalue), n_live


def compare_feature(
    feature_name: str,
    reference: Sequence[float | None],
    live: Sequence[float | None],
    *,
    monitoring: MonitoringSettings,
) -> list[DriftResult]:
    """PSI + KS for one feature, severity attached."""
    psi, n_live = population_stability_index(reference, live, bins=monitoring.drift_bins)
    severity = classify_psi(
        psi,
        moderate=monitoring.psi_moderate_threshold,
        significant=monitoring.psi_significant_threshold,
    )
    ks, p_value, _ = ks_statistic(reference, live)
    return [
        DriftResult(feature_name, METRIC_PSI, psi, None, severity, n_live),
        # KS carries no severity of its own -- it inherits PSI's so a single
        # feature never appears "stable by one metric, significant by
        # another" in the same window, which reads as a bug to anyone
        # looking at the dashboard.
        DriftResult(feature_name, METRIC_KS, ks, p_value, severity, n_live),
    ]


def worst_severity(results: Sequence[DriftResult]) -> str:
    if not results:
        return SEVERITY_STABLE
    return max((r.severity for r in results), key=lambda s: SEVERITY_RANK[s])


def correlated_breach(
    results: Sequence[DriftResult], *, min_features: int
) -> tuple[bool, list[str]]:
    """Is this a *correlated multi-feature* breach worth alerting on?

    Single-feature drift is usually noise, and alerting on it produces the
    fatigue that gets real signals ignored (phase-6 plan). So the alert
    condition is ``min_features`` distinct features breaching in the same
    window — not any one feature crossing a line.

    Counts distinct feature names, not rows: PSI and KS both breaching for
    one feature is one feature, not two.
    """
    breached = sorted({r.feature_name for r in results if r.breached})
    return len(breached) >= min_features, breached


def reference_samples_from_stats(
    reference_stats: Mapping[str, Mapping[str, float]],
    feature_name: str,
    *,
    size: int,
    rng: np.random.Generator,
) -> list[float]:
    """Reconstruct an approximate reference sample from stored summary stats.

    ``model_versions.reference_feature_stats`` holds mean/std/min/max/p50 per
    feature, not the raw training column — storing the full training
    distribution per promotion would grow without bound. A normal sample
    matching the stored mean and std, clipped to the stored range, is a
    deliberate approximation: it is enough to make PSI's binning meaningful,
    and it is honest about being an approximation rather than pretending the
    original data is still around.

    Consequence worth knowing: this understates drift for features whose
    training distribution was strongly non-normal. Storing raw reference
    quantiles would fix that and is the natural next change if drift ever
    reads as suspiciously quiet.
    """
    summary = reference_stats.get(feature_name)
    if summary is None:
        return []
    mean = float(summary.get("mean", 0.0))
    std = float(summary.get("std", 0.0))
    low = float(summary.get("min", mean))
    high = float(summary.get("max", mean))
    if std <= 0:
        return [mean] * size
    sample: list[float] = np.clip(rng.normal(mean, std, size), low, high).tolist()
    return sample


def run_drift_monitoring(
    session_factory: sessionmaker[Session],
    monitoring: MonitoringSettings,
    mlflow_model_name: str,
    *,
    now: datetime | None = None,
    seed: int = 20260817,
) -> list[DriftResult]:
    """Compare the live feature window against the Production model's
    reference snapshot and persist one row per (feature, metric).

    Returns an empty list — and writes nothing — when there is no Production
    model or no stored reference snapshot. That is not a silent pass:
    ``drift_metrics`` staying empty is what makes the API report
    ``computed_at=None`` rather than "no breaches", so "never evaluated"
    never renders as a clean bill of health.

    The RNG is seeded so a re-run over the same window reproduces the same
    numbers; the reference reconstruction above is sampled, and an
    unreproducible monitoring metric is not a metric.
    """
    now = now or datetime.now(UTC)
    window = timedelta(hours=monitoring.drift_window_hours)
    window_start, window_end = now - window, now
    rng = np.random.default_rng(seed)

    results: list[DriftResult] = []
    with session_scope(session_factory) as session:
        production = get_current_production_version(session, mlflow_model_name)
        if production is None or not production.reference_feature_stats:
            logger.warning(
                "skipping drift monitoring: no Production model reference snapshot",
                extra={"extra_fields": {"model_name": mlflow_model_name}},
            )
            return []

        reference_stats = _coerce_reference_stats(production.reference_feature_stats)
        live_values = _collect_live_values(session, window_start, window_end)

        for feature_name, live in sorted(live_values.items()):
            reference = reference_samples_from_stats(
                reference_stats, feature_name, size=max(len(live), 1), rng=rng
            )
            if not reference or not live:
                continue
            results.extend(compare_feature(feature_name, reference, live, monitoring=monitoring))

        for result in results:
            record_drift_metric(
                session,
                feature_name=result.feature_name,
                metric_name=result.metric_name,
                metric_value=result.metric_value,
                p_value=result.p_value,
                severity=result.severity,
                reference_model_version=production.mlflow_model_version,
                sample_size=result.sample_size,
                window_start=window_start,
                window_end=window_end,
                computed_at=now,
            )

    logger.info(
        "drift monitoring complete",
        extra={
            "extra_fields": {
                "features_evaluated": len({r.feature_name for r in results}),
                "worst_severity": worst_severity(results),
            }
        },
    )
    return results


def _coerce_reference_stats(raw: Mapping[str, object]) -> dict[str, dict[str, float]]:
    """JSONB comes back as plain dicts of unknown shape; narrow it, skipping
    anything that isn't a per-feature stats mapping rather than crashing the
    whole DAG run on one malformed key.
    """
    coerced: dict[str, dict[str, float]] = {}
    for name, summary in raw.items():
        if isinstance(summary, dict):
            coerced[name] = {
                key: float(value)
                for key, value in summary.items()
                if isinstance(value, int | float)
            }
    return coerced


def _collect_live_values(
    session: Session, window_start: datetime, window_end: datetime
) -> dict[str, list[float | None]]:
    """Pool every symbol's feature values in the window, per feature name.

    Pooled rather than per-symbol on purpose: the reference snapshot is
    itself pooled across the symbols the model trained on, so a per-symbol
    comparison would be measuring against the wrong baseline.
    """
    live: dict[str, list[float | None]] = {}
    for symbol in list_symbols(session):
        rows = list_feature_rows_in_range(
            session, symbol.id, FEATURE_SET_VERSION, window_start, window_end
        )
        for row in rows:
            if row.insufficient_history:
                continue
            for name, value in row.feature_values.items():
                live.setdefault(name, []).append(value)
    return live
