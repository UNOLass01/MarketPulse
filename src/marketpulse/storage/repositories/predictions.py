"""Prediction logging (Phase 5) and outcome resolution reads (Phase 6).

Two tables, one module, because every query here is about the same join:
what the model said, and — ``H`` later — what actually happened. Keeping the
outcome reads next to the prediction writes is what makes the horizon-lag
rule visible in one place instead of re-derived at each call site.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import Row, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.storage.models import Prediction, PredictionOutcome, RawTick, Symbol
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id


def upsert_prediction(
    session: Session,
    *,
    symbol: str,
    model_version: str,
    feature_set_version: int,
    feature_ts: datetime,
    predicted_at: datetime,
    label: str,
    probabilities: Mapping[str, float],
    latency_ms: float,
    correlation_id: UUID | None = None,
) -> int | None:
    """Record one served prediction. Returns the new row's id, or ``None``
    when this ``(symbol, feature_ts, model_version)`` was already logged.

    ``DO NOTHING`` rather than ``DO UPDATE``: re-serving the same features
    through the same model is by definition the same prediction, so the
    first row — the request that actually paid the inference cost — is the
    one worth keeping.
    """
    symbol_id = get_or_create_symbol_id(session, symbol)
    stmt = (
        pg_insert(Prediction)
        .values(
            symbol_id=symbol_id,
            model_version=model_version,
            feature_set_version=feature_set_version,
            feature_ts=feature_ts,
            predicted_at=predicted_at,
            label=label,
            probabilities=dict(probabilities),
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        .on_conflict_do_nothing(constraint="uq_predictions_symbol_ts_model")
        .returning(Prediction.id)
    )
    return session.execute(stmt).scalar_one_or_none()


def list_predictions(
    session: Session,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    model_version: str | None = None,
    limit: int,
    offset: int = 0,
) -> tuple[list[Prediction], bool]:
    """A page of a symbol's prediction history, newest first.

    Returns ``(rows, has_more)``. ``has_more`` comes from fetching one extra
    row and discarding it, not from a second ``COUNT(*)`` — the table only
    grows, and an exact total nobody displays is not worth a full scan.
    """
    stmt = (
        select(Prediction)
        .join(Symbol, Symbol.id == Prediction.symbol_id)
        .where(
            Symbol.code == symbol,
            Prediction.predicted_at >= start,
            Prediction.predicted_at < end,
        )
        .order_by(Prediction.predicted_at.desc())
        .offset(offset)
        .limit(limit + 1)
    )
    if model_version is not None:
        stmt = stmt.where(Prediction.model_version == model_version)

    rows = list(session.execute(stmt).scalars())
    return rows[:limit], len(rows) > limit


def resolvable_predictions(
    session: Session,
    *,
    horizon: timedelta,
    now: datetime,
    limit: int = 5_000,
) -> list[Row[tuple[int, str, datetime, str, str]]]:
    """Predictions old enough to have a realised outcome, and not yet resolved.

    The horizon rule, stated once: a prediction made off features at ``t``
    claims something about the price at ``t + H``, so it becomes resolvable
    exactly when ``t + H <= now`` — i.e. ``feature_ts <= now - H``. Anything
    younger is *pending*, not wrong, and must stay out of every accuracy
    number until its horizon has actually elapsed.

    The ``LEFT JOIN ... IS NULL`` (rather than ``NOT IN``) makes re-running
    the attribution DAG idempotent: already-resolved rows simply don't come
    back.
    """
    cutoff = now - horizon
    stmt = (
        select(
            Prediction.id,
            Symbol.code,
            Prediction.feature_ts,
            Prediction.label,
            Prediction.model_version,
        )
        .join(Symbol, Symbol.id == Prediction.symbol_id)
        .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(Prediction.feature_ts <= cutoff, PredictionOutcome.id.is_(None))
        .order_by(Prediction.feature_ts.asc())
        .limit(limit)
    )
    return list(session.execute(stmt))


def price_at_or_after(
    session: Session, *, symbol: str, target: datetime, tolerance: timedelta
) -> tuple[datetime, Decimal] | None:
    """Earliest tick at or after ``target``, within ``tolerance``.

    Ticks are not evenly spaced, so "the price H later" has to be located by
    time rather than by row offset — the same as-of lookup
    ``ml.dataset._attach_forward_price`` does offline, kept bounded for the
    same reason: without a tolerance a prediction sitting next to a data gap
    would silently borrow a price from hours later and be scored against it.
    """
    stmt = (
        select(RawTick.observed_at, RawTick.price)
        .join(Symbol, Symbol.id == RawTick.symbol_id)
        .where(
            Symbol.code == symbol,
            RawTick.observed_at >= target,
            RawTick.observed_at <= target + tolerance,
        )
        .order_by(RawTick.observed_at.asc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    return (row[0], row[1]) if row is not None else None


def price_at(session: Session, *, symbol: str, observed_at: datetime) -> Decimal | None:
    """The tick price at exactly ``observed_at`` — features and raw ticks are
    1:1 on ``(symbol, observed_at)``, so a prediction's ``feature_ts`` always
    has a matching base price unless that partition has been archived away.
    """
    stmt = (
        select(RawTick.price)
        .join(Symbol, Symbol.id == RawTick.symbol_id)
        .where(Symbol.code == symbol, RawTick.observed_at == observed_at)
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def record_outcome(
    session: Session,
    *,
    prediction_id: int,
    horizon_minutes: float,
    theta: float,
    base_price: Decimal,
    future_price: Decimal,
    future_ts: datetime,
    realised_return: float,
    actual_label: str,
    is_correct: bool,
) -> int | None:
    """Record what actually happened. Idempotent per prediction."""
    stmt = (
        pg_insert(PredictionOutcome)
        .values(
            prediction_id=prediction_id,
            horizon_minutes=horizon_minutes,
            theta=theta,
            base_price=base_price,
            future_price=future_price,
            future_ts=future_ts,
            realised_return=realised_return,
            actual_label=actual_label,
            is_correct=is_correct,
        )
        .on_conflict_do_nothing(constraint="uq_prediction_outcomes_prediction")
        .returning(PredictionOutcome.id)
    )
    return session.execute(stmt).scalar_one_or_none()


def list_resolved(
    session: Session, *, start: datetime, end: datetime
) -> list[Row[tuple[str, str, str, str, datetime]]]:
    """Resolved (predicted, actual) pairs in a window, tagged by model version.

    Windowed on ``predicted_at``, not ``resolved_at``: the question rolling
    accuracy answers is "how did the model do over this period", and the
    period a prediction belongs to is when it was made, not when a batch job
    happened to get around to scoring it.
    """
    stmt = (
        select(
            Prediction.model_version,
            Symbol.code,
            Prediction.label,
            PredictionOutcome.actual_label,
            Prediction.predicted_at,
        )
        .join(Symbol, Symbol.id == Prediction.symbol_id)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(Prediction.predicted_at >= start, Prediction.predicted_at < end)
        .order_by(Prediction.predicted_at.asc())
    )
    return list(session.execute(stmt))


def count_pending(session: Session, *, horizon: timedelta, now: datetime) -> int:
    """How many predictions are still inside their horizon. Reported
    alongside accuracy so "not known yet" never reads as "got it wrong".
    """
    cutoff = now - horizon
    stmt = (
        select(Prediction.id)
        .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(Prediction.feature_ts > cutoff, PredictionOutcome.id.is_(None))
    )
    return len(list(session.execute(stmt)))


def latest_prediction_at(session: Session) -> datetime | None:
    stmt = select(Prediction.predicted_at).order_by(Prediction.predicted_at.desc()).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def distinct_model_versions(session: Session, *, start: datetime, end: datetime) -> Sequence[str]:
    stmt = (
        select(Prediction.model_version)
        .where(Prediction.predicted_at >= start, Prediction.predicted_at < end)
        .distinct()
    )
    return list(session.execute(stmt).scalars())
