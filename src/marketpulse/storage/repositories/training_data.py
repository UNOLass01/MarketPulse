"""Fetch persisted feature rows joined to the price they were computed from,
for ``ml.dataset.assemble_dataset`` to turn into a training set.

The only query in this module filters on ``Feature.feature_ts`` (which is
1:1 with ``RawTick.observed_at`` -- CLAUDE.md rule #6: never ``ingested_at``
for anything model-relevant. This module returns plain rows; the actual
join/label/split logic stays in ``ml.dataset``, which is DB-free and unit
testable on its own.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketpulse.ml.dataset import FeatureRow
from marketpulse.storage.models import Feature, RawTick, Symbol


def fetch_feature_rows(
    session: Session,
    *,
    feature_set_version: int,
    symbols: Sequence[str] | None = None,
) -> list[FeatureRow]:
    """Ascending :class:`~marketpulse.ml.dataset.FeatureRow` rows for every
    persisted feature vector of ``feature_set_version``, joined to the price
    of the same tick it was computed from.
    """
    stmt = (
        select(
            Symbol.code,
            Feature.feature_ts,
            RawTick.price,
            Feature.feature_values,
            Feature.insufficient_history,
            Feature.has_gap,
        )
        .join(
            RawTick,
            (RawTick.symbol_id == Feature.symbol_id) & (RawTick.observed_at == Feature.feature_ts),
        )
        .join(Symbol, Symbol.id == Feature.symbol_id)
        .where(Feature.feature_set_version == feature_set_version)
        .order_by(Feature.feature_ts.asc())
    )
    if symbols:
        stmt = stmt.where(Symbol.code.in_(symbols))

    rows = session.execute(stmt).all()
    return [
        FeatureRow(
            symbol=code,
            feature_ts=feature_ts,
            price=float(price),
            feature_values=dict(feature_values),
            insufficient_history=insufficient_history,
            has_gap=has_gap,
        )
        for code, feature_ts, price, feature_values, insufficient_history, has_gap in rows
    ]
