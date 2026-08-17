"""Alert evaluation: thresholds, dedup, suppression, sustained breach.

Two properties make the difference between an alerting system people act on
and one they mute:

1. **A single spike does not fire.** A breach has to persist for
   ``alert_sustained_evaluations`` consecutive evaluations. The first breach
   opens a record and starts counting; nothing pages until the count is met.
2. **The same condition fires once.** Dedup is by ``dedup_key``: while an
   alert for that key is open, a repeat breach updates it in place. A
   condition that stays broken for six hours is one alert, not seventy-two.

And one rule with no exceptions: **every alert names its runbook.** An alert
with no attached action is just anxiety (phase-6 plan), so :class:`AlertRule`
requires one, and the ``alerts.runbook`` column is ``NOT NULL`` behind it.

:func:`decide` is a pure function of (existing state, current breach, clock).
That is what lets the suppression and sustained-breach rules — the two
easiest things here to get subtly wrong — be tested exhaustively with no
database.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import MonitoringSettings
from marketpulse.logging import get_logger
from marketpulse.monitoring.drift import DriftResult, correlated_breach, worst_severity
from marketpulse.monitoring.performance import PerformanceSlice, distribution_shift
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.alerts import (
    get_open_alert,
    open_alert,
    resolve_alert,
    update_alert,
)

logger = get_logger(__name__)

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

#: Runbook paths, relative to the repo root. Referenced by name from every
#: rule below so a renamed runbook is a single edit and a missing one is
#: caught by ``tests/unit/test_alerts.py``, not by whoever is on call.
RUNBOOK_CONSUMER_LAG = "docs/runbooks/consumer_lag.md"
RUNBOOK_MODEL_ROLLBACK = "docs/runbooks/model_rollback.md"
RUNBOOK_DLQ_TRIAGE = "docs/runbooks/dlq_triage.md"

ACTION_NONE = "none"
ACTION_ACCUMULATE = "accumulate"
ACTION_FIRE = "fire"
ACTION_SUPPRESS = "suppress"
ACTION_RESOLVE = "resolve"


@dataclass(frozen=True)
class AlertRule:
    """A named condition and what to do about it.

    ``runbook`` has no default. Adding a rule without one is a type error,
    which is the cheapest possible enforcement of "every alert names its
    runbook".
    """

    name: str
    severity: str
    runbook: str


#: Drift in several features at once against the Production model's training
#: reference. Rolling back the model is the containment action; the runbook
#: also covers checking the pipeline first, since drift more often means a
#: broken producer than a moved market.
RULE_FEATURE_DRIFT = AlertRule("feature_drift", SEVERITY_WARNING, RUNBOOK_MODEL_ROLLBACK)

#: Realised accuracy under the configured floor, sliced by model version.
RULE_ACCURACY_DEGRADED = AlertRule(
    "model_accuracy_degraded", SEVERITY_CRITICAL, RUNBOOK_MODEL_ROLLBACK
)

#: The predicted-class mix has moved away from the training prior. Fires long
#: before accuracy can and usually means broken features, so it points at the
#: pipeline runbook rather than at model rollback.
RULE_PREDICTION_DISTRIBUTION = AlertRule(
    "prediction_distribution_shift", SEVERITY_WARNING, RUNBOOK_CONSUMER_LAG
)

#: A Phase 4 data-quality check failed. Surfaced here so quality failures
#: live in the same alert stream as everything else instead of only in a DAG
#: log nobody reads on a good day.
RULE_QUALITY_FAILED = AlertRule("data_quality_failed", SEVERITY_WARNING, RUNBOOK_DLQ_TRIAGE)

ALL_RULES = (
    RULE_FEATURE_DRIFT,
    RULE_ACCURACY_DEGRADED,
    RULE_PREDICTION_DISTRIBUTION,
    RULE_QUALITY_FAILED,
)


@dataclass(frozen=True)
class Breach:
    """One evaluation's verdict for one rule."""

    rule: AlertRule
    dedup_key: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertState:
    """The currently-open alert for a dedup key, as stored."""

    dedup_key: str
    consecutive_breaches: int
    first_breached_at: datetime
    fired_at: datetime | None


@dataclass(frozen=True)
class AlertDecision:
    action: str
    consecutive_breaches: int
    should_fire: bool
    reason: str


def decide(
    existing: AlertState | None,
    breach: Breach | None,
    *,
    now: datetime,
    sustained_evaluations: int,
    suppression: timedelta,
) -> AlertDecision:
    """Decide what this evaluation does to the alert for one dedup key.

    The four outcomes:

    * **resolve** — nothing is breaching and an alert was open. The condition
      cleared; close it.
    * **none** — nothing is breaching and nothing was open.
    * **accumulate** — breaching, but not yet for enough consecutive
      evaluations. Record the breach, page no one. This is the single-spike
      case.
    * **fire** — breaching, and the sustained threshold is now met, and this
      key is not already firing inside its suppression window.
    * **suppress** — breaching and already fired recently. Update the open
      alert, stay quiet.
    """
    if breach is None:
        if existing is not None:
            return AlertDecision(ACTION_RESOLVE, 0, False, "condition cleared")
        return AlertDecision(ACTION_NONE, 0, False, "no breach")

    consecutive = (existing.consecutive_breaches + 1) if existing else 1

    if consecutive < sustained_evaluations:
        return AlertDecision(
            ACTION_ACCUMULATE,
            consecutive,
            False,
            f"breach {consecutive}/{sustained_evaluations}; not yet sustained",
        )

    if existing is not None and existing.fired_at is not None:
        if now - existing.fired_at < suppression:
            return AlertDecision(
                ACTION_SUPPRESS,
                consecutive,
                False,
                f"already fired at {existing.fired_at.isoformat()}; within suppression window",
            )
        # Past the suppression window and still broken -- re-fire so a
        # long-running incident does not silently fall off the radar.
        return AlertDecision(ACTION_FIRE, consecutive, True, "suppression window elapsed")

    return AlertDecision(ACTION_FIRE, consecutive, True, f"sustained for {consecutive} evaluations")


def evaluate_drift(results: Sequence[DriftResult], monitoring: MonitoringSettings) -> Breach | None:
    """Breach only on *correlated* multi-feature drift, never on one feature."""
    is_breach, features = correlated_breach(
        results, min_features=monitoring.drift_min_correlated_features
    )
    if not is_breach:
        return None
    return Breach(
        rule=RULE_FEATURE_DRIFT,
        dedup_key=RULE_FEATURE_DRIFT.name,
        details={
            "drifted_features": features,
            "feature_count": len(features),
            "min_correlated_features": monitoring.drift_min_correlated_features,
            "worst_severity": worst_severity(results),
        },
    )


def evaluate_accuracy(
    slices: Sequence[PerformanceSlice], monitoring: MonitoringSettings
) -> Breach | None:
    """Breach when a model version's realised accuracy falls below the floor.

    Slices with fewer than ``performance_min_resolved`` outcomes are skipped:
    the metric is still computed and stored (it is the record), it is just
    not trusted enough to wake anyone, because early accuracy on a handful of
    samples swings wildly for reasons that have nothing to do with the model.
    """
    degraded = [
        s
        for s in slices
        if s.resolved_count >= monitoring.performance_min_resolved
        and s.accuracy < monitoring.performance_min_accuracy
    ]
    if not degraded:
        return None
    worst = min(degraded, key=lambda s: s.accuracy)
    return Breach(
        rule=RULE_ACCURACY_DEGRADED,
        # Keyed by model version: a new promotion is a genuinely new
        # condition and deserves its own alert rather than being deduped
        # against the version it replaced.
        dedup_key=f"{RULE_ACCURACY_DEGRADED.name}:{worst.model_version}",
        details={
            "model_version": worst.model_version,
            "accuracy": worst.accuracy,
            "macro_f1": worst.macro_f1,
            "resolved_count": worst.resolved_count,
            "min_accuracy": monitoring.performance_min_accuracy,
        },
    )


def evaluate_prediction_distribution(
    slices: Sequence[PerformanceSlice],
    training_prior: Mapping[str, float],
    monitoring: MonitoringSettings,
) -> Breach | None:
    """Breach when the live predicted-class mix drifts from the training prior."""
    if not training_prior:
        return None
    shifted = [
        (s, distribution_shift(s.predicted_distribution, dict(training_prior)))
        for s in slices
        if s.resolved_count > 0
    ]
    breaching = [
        (s, shift) for s, shift in shifted if shift > monitoring.prediction_distribution_max_shift
    ]
    if not breaching:
        return None
    worst_slice, worst_shift = max(breaching, key=lambda pair: pair[1])
    return Breach(
        rule=RULE_PREDICTION_DISTRIBUTION,
        dedup_key=f"{RULE_PREDICTION_DISTRIBUTION.name}:{worst_slice.model_version}",
        details={
            "model_version": worst_slice.model_version,
            "shift": worst_shift,
            "max_shift": monitoring.prediction_distribution_max_shift,
            "live_distribution": worst_slice.predicted_distribution,
            "training_prior": dict(training_prior),
        },
    )


def apply_decision(
    session: Session,
    breach: Breach | None,
    dedup_key: str,
    *,
    now: datetime,
    monitoring: MonitoringSettings,
) -> AlertDecision:
    """Load the current state for ``dedup_key``, decide, and write the result."""
    row = get_open_alert(session, dedup_key)
    existing = (
        AlertState(
            dedup_key=row.dedup_key,
            consecutive_breaches=row.consecutive_breaches,
            first_breached_at=row.first_breached_at,
            fired_at=row.fired_at,
        )
        if row is not None
        else None
    )

    decision = decide(
        existing,
        breach,
        now=now,
        sustained_evaluations=monitoring.alert_sustained_evaluations,
        suppression=timedelta(minutes=monitoring.alert_suppression_minutes),
    )

    if decision.action == ACTION_RESOLVE and row is not None:
        resolve_alert(session, row, resolved_at=now)
        logger.info("alert resolved", extra={"extra_fields": {"dedup_key": dedup_key}})
        return decision

    if breach is None:
        return decision

    fired_at = now if decision.should_fire else None
    if row is None:
        open_alert(
            session,
            alert_name=breach.rule.name,
            dedup_key=dedup_key,
            severity=breach.rule.severity,
            runbook=breach.rule.runbook,
            details=breach.details,
            consecutive_breaches=decision.consecutive_breaches,
            first_breached_at=now,
            fired_at=fired_at,
            updated_at=now,
        )
    else:
        update_alert(
            session,
            row,
            details=breach.details,
            consecutive_breaches=decision.consecutive_breaches,
            fired_at=fired_at,
            updated_at=now,
        )

    if decision.should_fire:
        logger.error(
            "ALERT FIRED",
            extra={
                "extra_fields": {
                    "alert_name": breach.rule.name,
                    "dedup_key": dedup_key,
                    "severity": breach.rule.severity,
                    "runbook": breach.rule.runbook,
                    "consecutive_breaches": decision.consecutive_breaches,
                    **breach.details,
                }
            },
        )
    return decision


def evaluate_and_record(
    session_factory: sessionmaker[Session],
    monitoring: MonitoringSettings,
    *,
    drift_results: Sequence[DriftResult] = (),
    performance_slices: Sequence[PerformanceSlice] = (),
    training_prior: Mapping[str, float] | None = None,
    now: datetime | None = None,
) -> list[AlertDecision]:
    """Evaluate every rule this run has evidence for, and persist the outcome.

    Rules with no evidence are skipped rather than evaluated as "not
    breaching": a drift run that produced no results must not silently
    resolve an open accuracy alert it knows nothing about. Absence of a
    signal is never treated as a passing signal — the same discipline
    ``storage.repositories.quality.latest_checks_passed`` applies to the
    retraining gate.
    """
    now = now or datetime.now(UTC)
    decisions: list[AlertDecision] = []

    with session_scope(session_factory) as session:
        if drift_results:
            breach = evaluate_drift(drift_results, monitoring)
            decisions.append(
                apply_decision(
                    session, breach, RULE_FEATURE_DRIFT.name, now=now, monitoring=monitoring
                )
            )

        if performance_slices:
            accuracy_breach = evaluate_accuracy(performance_slices, monitoring)
            for candidate in performance_slices:
                key = f"{RULE_ACCURACY_DEGRADED.name}:{candidate.model_version}"
                decisions.append(
                    apply_decision(
                        session,
                        accuracy_breach if _matches(accuracy_breach, key) else None,
                        key,
                        now=now,
                        monitoring=monitoring,
                    )
                )

            if training_prior:
                dist_breach = evaluate_prediction_distribution(
                    performance_slices, training_prior, monitoring
                )
                for candidate in performance_slices:
                    key = f"{RULE_PREDICTION_DISTRIBUTION.name}:{candidate.model_version}"
                    decisions.append(
                        apply_decision(
                            session,
                            dist_breach if _matches(dist_breach, key) else None,
                            key,
                            now=now,
                            monitoring=monitoring,
                        )
                    )

    return decisions


def _matches(breach: Breach | None, dedup_key: str) -> bool:
    return breach is not None and breach.dedup_key == dedup_key
