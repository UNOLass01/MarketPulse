"""Idempotent feature vector persistence + latest-per-symbol lookup."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.contracts.features import FeatureVector
from marketpulse.storage.models import Feature
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
