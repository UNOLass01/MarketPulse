"""ORM models.

``raw_ticks`` is range-partitioned monthly by ``observed_at`` (native
Postgres partitioning — see ``storage.partitions``). Every unique index on a
partitioned table must include the partition key, which is why the
``message_id`` uniqueness constraint also carries ``observed_at`` even though
``message_id`` alone is already globally unique in practice.
"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
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
