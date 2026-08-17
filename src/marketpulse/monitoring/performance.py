"""Performance attribution: what the model actually got right in production.

This is the part most portfolio projects skip, and the reason is the lag. A
prediction made at ``t`` is a claim about the price at ``t + H``, so nothing
can be scored until ``H`` has genuinely elapsed. Every debug cycle therefore
costs at least ``H``, which is why the off-by-one in the resolution window is
both the likeliest bug in this phase and the most annoying one to find.

The rule, stated once and enforced in exactly one place
(``storage.repositories.predictions.resolvable_predictions``):

    a prediction with ``feature_ts = t`` is resolvable iff ``t + H <= now``

Anything younger is *pending*, not wrong, and is reported separately so that
"we don't know yet" never gets averaged into an accuracy number as if it were
a miss.

Metrics are sliced by ``model_version`` because an unsliced rolling accuracy
straddling a promotion is a weighted average of two different models — which
smears out exactly the step change a promotion is supposed to produce.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import MonitoringSettings
from marketpulse.logging import get_logger
from marketpulse.ml.config import load_training_config
from marketpulse.ml.labeling import LABELS, forward_return, label_from_return
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import TrainingRun
from marketpulse.storage.repositories.predictions import (
    count_pending,
    list_resolved,
    price_at,
    price_at_or_after,
    record_outcome,
    resolvable_predictions,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerformanceSlice:
    """Rolling metrics for one model version over one window."""

    model_version: str
    resolved_count: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    predicted_distribution: dict[str, float]
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True)
class ResolutionSummary:
    resolved: int
    skipped_no_future_price: int
    skipped_no_base_price: int
    pending: int


def confusion_matrix(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, dict[str, int]]:
    """``matrix[actual][predicted]`` over the full label set.

    Built from ``labeling.LABELS`` rather than from the labels observed, so a
    class the model never predicted shows as an explicit row of zeros instead
    of vanishing — "never predicted UP" is a finding, not an absence.
    """
    matrix = {a: dict.fromkeys(LABELS, 0) for a in LABELS}
    for a, p in zip(actual, predicted, strict=True):
        matrix[a][p] += 1
    return matrix


def per_class_f1(matrix: dict[str, dict[str, int]]) -> dict[str, float]:
    """F1 per class from a confusion matrix.

    A class with no true instances *and* no predictions scores 0.0 rather
    than being omitted — omitting it would quietly raise the macro average
    by shrinking its denominator.
    """
    scores: dict[str, float] = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        predicted_positive = sum(matrix[a][label] for a in LABELS)
        actual_positive = sum(matrix[label].values())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def macro_f1(matrix: dict[str, dict[str, int]]) -> float:
    scores = per_class_f1(matrix)
    return sum(scores.values()) / len(LABELS)


def accuracy_of(matrix: dict[str, dict[str, int]]) -> float:
    total = sum(sum(row.values()) for row in matrix.values())
    if total == 0:
        return 0.0
    return sum(matrix[label][label] for label in LABELS) / total


def class_distribution(labels: Sequence[str]) -> dict[str, float]:
    """Predicted-class mix as proportions over the full label set."""
    if not labels:
        return dict.fromkeys(LABELS, 0.0)
    counts = {label: 0 for label in LABELS}
    for label in labels:
        counts[label] += 1
    return {label: counts[label] / len(labels) for label in LABELS}


def distribution_shift(live: dict[str, float], reference: dict[str, float]) -> float:
    """Total-variation distance between two class distributions (0 to 1).

    Watched because it moves *long* before accuracy can: accuracy needs
    ``H`` to elapse and enough resolved samples to be stable, whereas the
    predicted-class mix is observable the instant predictions are served. A
    sudden collapse to one class almost always means broken features rather
    than a regime change (phase-6 plan) — check the feature pipeline before
    the market.
    """
    return 0.5 * sum(abs(live.get(label, 0.0) - reference.get(label, 0.0)) for label in LABELS)


def build_slice(
    model_version: str,
    actual: Sequence[str],
    predicted: Sequence[str],
    *,
    window_start: datetime,
    window_end: datetime,
) -> PerformanceSlice:
    matrix = confusion_matrix(actual, predicted)
    return PerformanceSlice(
        model_version=model_version,
        resolved_count=len(actual),
        accuracy=accuracy_of(matrix),
        macro_f1=macro_f1(matrix),
        per_class_f1=per_class_f1(matrix),
        confusion_matrix=matrix,
        predicted_distribution=class_distribution(predicted),
        window_start=window_start,
        window_end=window_end,
    )


def resolve_outcomes(
    session_factory: sessionmaker[Session],
    *,
    horizon: timedelta,
    theta: float,
    price_tolerance: timedelta,
    now: datetime | None = None,
) -> ResolutionSummary:
    """Join predictions past their horizon to the realised price and score them.

    A prediction whose ``t + H`` price cannot be found within
    ``price_tolerance`` is left *unresolved*, not scored against a distant
    substitute price. Widening the search to "whatever came next" would score
    a prediction against a horizon it never claimed anything about, and the
    resulting accuracy would be quietly meaningless around every data gap.

    ``theta`` is the same deadband the model was trained with, read from the
    versioned training config — deriving a fresh one from live returns would
    score the model against a different labeling problem than the one it
    learned.
    """
    now = now or datetime.now(UTC)
    resolved = skipped_future = skipped_base = 0

    with session_scope(session_factory) as session:
        candidates = resolvable_predictions(session, horizon=horizon, now=now)

        for prediction_id, symbol, feature_ts, predicted_label, _version in candidates:
            base_price = price_at(session, symbol=symbol, observed_at=feature_ts)
            if base_price is None or base_price <= 0:
                skipped_base += 1
                continue

            future = price_at_or_after(
                session,
                symbol=symbol,
                target=feature_ts + horizon,
                tolerance=price_tolerance,
            )
            if future is None:
                skipped_future += 1
                continue

            future_ts, future_price = future
            realised = forward_return(float(base_price), float(future_price))
            actual_label = label_from_return(realised, theta)

            record_outcome(
                session,
                prediction_id=prediction_id,
                horizon_minutes=horizon.total_seconds() / 60.0,
                theta=theta,
                base_price=base_price,
                future_price=future_price,
                future_ts=future_ts,
                realised_return=realised,
                actual_label=actual_label,
                is_correct=actual_label == predicted_label,
            )
            resolved += 1

        pending = count_pending(session, horizon=horizon, now=now)

    logger.info(
        "prediction outcomes resolved",
        extra={
            "extra_fields": {
                "resolved": resolved,
                "skipped_no_future_price": skipped_future,
                "skipped_no_base_price": skipped_base,
                "pending": pending,
            }
        },
    )
    return ResolutionSummary(resolved, skipped_future, skipped_base, pending)


def compute_performance_slices(
    session_factory: sessionmaker[Session],
    monitoring: MonitoringSettings,
    *,
    now: datetime | None = None,
) -> list[PerformanceSlice]:
    """Rolling metrics over the configured window, one slice per model version."""
    now = now or datetime.now(UTC)
    window_start = now - timedelta(hours=monitoring.performance_window_hours)

    by_version: dict[str, tuple[list[str], list[str]]] = {}
    with session_scope(session_factory) as session:
        for model_version, _symbol, predicted, actual, _predicted_at in list_resolved(
            session, start=window_start, end=now
        ):
            actuals, predictions = by_version.setdefault(model_version, ([], []))
            actuals.append(actual)
            predictions.append(predicted)

    return [
        build_slice(
            version,
            actual,
            predicted,
            window_start=window_start,
            window_end=now,
        )
        for version, (actual, predicted) in sorted(by_version.items())
    ]


def run_performance_attribution(
    session_factory: sessionmaker[Session],
    monitoring: MonitoringSettings,
    *,
    price_tolerance: timedelta,
    now: datetime | None = None,
) -> tuple[ResolutionSummary, list[PerformanceSlice]]:
    """Resolve everything past its horizon, then recompute rolling metrics.

    ``H`` and ``theta`` come from the versioned training config, so this
    scores production against exactly the labeling problem the model was
    trained on.
    """
    config = load_training_config()
    horizon = config.labeling.horizon
    theta = config.labeling.theta
    if theta is None:
        # theta was derived from the training sample rather than pinned in
        # config. Fall back to the value recorded on the most recent training
        # run so scoring still uses a real, logged deadband instead of an
        # invented one.
        theta = _theta_from_last_training_run(session_factory)

    summary = resolve_outcomes(
        session_factory,
        horizon=horizon,
        theta=theta,
        price_tolerance=price_tolerance,
        now=now,
    )
    return summary, compute_performance_slices(session_factory, monitoring, now=now)


def _theta_from_last_training_run(session_factory: sessionmaker[Session]) -> float:
    with session_scope(session_factory) as session:
        stmt = select(TrainingRun.theta).order_by(TrainingRun.finished_at.desc()).limit(1)
        theta = session.execute(stmt).scalar_one_or_none()
    if theta is None:
        raise ValueError(
            "cannot score predictions: config.labeling.theta is unset and no "
            "training run has recorded one"
        )
    return float(theta)
