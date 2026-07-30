"""ORM models.

``raw_ticks`` and ``features`` are both range-partitioned monthly (by
``observed_at`` / ``feature_ts`` respectively — native Postgres partitioning,
see ``storage.partitions``). Every unique index on a partitioned table must
include the partition key, which is why the ``message_id`` uniqueness
constraint also carries ``observed_at`` even though ``message_id`` alone is
already globally unique in practice, and likewise for ``features``.
"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Symbol(Base):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RawTick(Base):
    __tablename__ = "raw_ticks"
    __table_args__ = (
        UniqueConstraint("symbol_id", "observed_at", name="uq_raw_ticks_symbol_observed"),
        UniqueConstraint("message_id", "observed_at", name="uq_raw_ticks_message_observed"),
        Index("ix_raw_ticks_observed_at_brin", "observed_at", postgresql_using="brin"),
        CheckConstraint("source IN ('stream', 'seed')", name="ck_raw_ticks_source"),
        {"postgresql_partition_by": "RANGE (observed_at)"},
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    message_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    correlation_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    emitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    # Distinguishes rows backfilled by scripts/seed_historical.py from ones
    # written by the live streaming consumer — see phase-2 plan, "Wiring + seeding".
    source: Mapped[str] = mapped_column(String(16), nullable=False, server_default="stream")


class Feature(Base):
    """A symbol's computed feature vector as of ``feature_ts``.

    ``feature_values`` is a JSONB name -> value map, not fixed columns, so
    the feature set can evolve without a destructive migration — old and new
    ``feature_set_version``s simply coexist. Column order for training/serving
    always comes from ``features.registry``, never from this JSON blob's key
    order.
    """

    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "feature_ts",
            "feature_set_version",
            name="uq_features_symbol_ts_version",
        ),
        {"postgresql_partition_by": "RANGE (feature_ts)"},
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    feature_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    feature_set_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    feature_values: Mapped[dict[str, float | None]] = mapped_column(JSONB, nullable=False)
    insufficient_history: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_gap: Mapped[bool] = mapped_column(Boolean, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# Descending on feature_ts for the API's "latest features for symbol" hot
# path (Phase 5); defined after the class so it can reference the mapped
# column directly instead of a bare string.
Index("ix_features_symbol_ts_desc", Feature.symbol_id, Feature.feature_ts.desc())
