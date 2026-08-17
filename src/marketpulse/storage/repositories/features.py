"""Idempotent feature vector persistence + latest-per-symbol lookup."""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.contracts.features import FeatureVector
from marketpulse.storage.models import Feature, Symbol
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id


def upsert_feature_vector(session: Session, vector: FeatureVector) -> None:
    """Insert a feature row; recomputing the same (symbol, ts, version) is a
    no-op. Feature computation is a pure function of the same raw data, so a
    conflicting row is always identical in practice — ``DO NOTHING`` mirrors
    ``repositories.ticks.upsert_tick``'s idempotency pattern.
    """
    symbol_id = get_or_create_symbol_id(session, vector.symbol)
    stmt = (
        pg_insert(Feature)
        .values(
            feature_ts=vector.feature_ts,
            symbol_id=symbol_id,
            feature_set_version=vector.feature_set_version,
            feature_values=vector.feature_values,
            insufficient_history=vector.insufficient_history,
            has_gap=vector.has_gap,
        )
        .on_conflict_do_nothing(constraint="uq_features_symbol_ts_version")
    )
    session.execute(stmt)


def latest_feature_vector(
    session: Session, symbol_id: int, feature_set_version: int
) -> Feature | None:
    """Most recent feature row for ``symbol_id`` — the API's hot-path read."""
    stmt = (
        select(Feature)
        .where(
            Feature.symbol_id == symbol_id,
            Feature.feature_set_version == feature_set_version,
        )
        .order_by(Feature.feature_ts.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def latest_feature_vector_per_symbol(
    session: Session, feature_set_version: int, *, symbols: Sequence[str] | None = None
) -> list[tuple[str, Feature]]:
    """Newest feature row for every symbol, in one query.

    ``DISTINCT ON`` rather than a per-symbol loop: the batch predictions
    endpoint must not issue N round-trips to answer one request, and the
    partial index ``ix_features_symbol_ts_desc`` already orders exactly this
    way.
    """
    stmt = (
        select(Symbol.code, Feature)
        .join(Symbol, Symbol.id == Feature.symbol_id)
        .where(Feature.feature_set_version == feature_set_version)
        .distinct(Feature.symbol_id)
        .order_by(Feature.symbol_id, Feature.feature_ts.desc())
    )
    if symbols:
        stmt = stmt.where(Symbol.code.in_(symbols))
    return [(code, feature) for code, feature in session.execute(stmt)]


def latest_feature_ts_per_symbol(session: Session) -> dict[str, datetime]:
    """Newest ``feature_ts`` per symbol code — the ``/api/v1/symbols`` read
    and the dashboard's "last seen" column.
    """
    stmt = (
        select(Symbol.code, func.max(Feature.feature_ts))
        .join(Symbol, Symbol.id == Feature.symbol_id)
        .group_by(Symbol.code)
    )
    return {code: ts for code, ts in session.execute(stmt) if ts is not None}


def list_feature_rows_in_range(
    session: Session,
    symbol_id: int,
    feature_set_version: int,
    start: datetime,
    end: datetime,
) -> list[Feature]:
    """Ascending feature rows for ``symbol_id`` in ``[start, end)`` —
    ``monitoring.quality``'s validity + distribution-sanity checks.
    """
    stmt = (
        select(Feature)
        .where(
            Feature.symbol_id == symbol_id,
            Feature.feature_set_version == feature_set_version,
            Feature.feature_ts >= start,
            Feature.feature_ts < end,
        )
        .order_by(Feature.feature_ts.asc())
    )
    return list(session.execute(stmt).scalars())
