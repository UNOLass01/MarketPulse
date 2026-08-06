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
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
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


class TrainingRun(Base):
    """One ``ml.pipeline.run_training_pipeline`` invocation -- logged even
    when the outcome is "did not promote" (phase-3 plan: "record rejections
    too"). ``mlflow_run_id`` is the join key back to the tracking server;
    everything here is a locally-queryable mirror of what that run logged,
    not a replacement for it.
    """

    __tablename__ = "training_runs"
    __table_args__ = (UniqueConstraint("mlflow_run_id", name="uq_training_runs_mlflow_run_id"),)

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    mlflow_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_set_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    config_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    horizon_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    theta: Mapped[float] = mapped_column(Float, nullable=False)
    train_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    train_class_distribution: Mapped[dict[str, int]] = mapped_column(JSONB, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    candidate_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    baseline_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    incumbent_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ModelVersion(Base):
    """A registered MLflow model version, mirrored locally so promotion
    history and reference feature distributions (Phase 6 drift depends on
    the latter) are queryable without round-tripping to the tracking server.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint(
            "mlflow_model_name", "mlflow_model_version", name="uq_model_versions_name_version"
        ),
        CheckConstraint(
            "stage IN ('Staging', 'Production', 'Archived')", name="ck_model_versions_stage"
        ),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False)
    mlflow_model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    mlflow_model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(16), nullable=False)
    # Populated only when this version is transitioned to Production -- the
    # snapshot Phase 6's drift monitor compares live feature windows against.
    reference_feature_stats: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


Index("ix_model_versions_stage", ModelVersion.stage)


class QualityCheck(Base):
    """One data-quality check result (``monitoring.quality``, Phase 4).

    ``symbol`` is null for checks that aren't per-symbol (none currently
    are, but the column stays nullable rather than forcing a sentinel like
    ``"ALL"`` — CLAUDE.md rule #7's "never fake a value" spirit applies here
    too). ``dag_model_retraining``'s gate reads the most recent row per
    ``check_name`` and requires all of them ``passed``.
    """

    __tablename__ = "quality_checks"
    __table_args__ = (
        CheckConstraint(
            "check_name IN ('freshness', 'completeness', 'validity', 'distribution')",
            name="ck_quality_checks_check_name",
        ),
        Index("ix_quality_checks_name_checked_at", "check_name", "checked_at"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    check_name: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(20))
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    details: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ArchivedPartition(Base):
    """Record of one ``dag_data_archival`` export -- the audit trail proving
    a dropped partition's data really did make it to object storage first.
    Written only after the verify step (row count + checksum) passes; a
    partition drop with no corresponding row here is the signal something
    went wrong (phase-4 plan: "the verify step is non-negotiable").
    """

    __tablename__ = "archived_partitions"
    __table_args__ = (
        UniqueConstraint(
            "table_name", "partition_year", "partition_month", name="uq_archived_partitions"
        ),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    table_name: Mapped[str] = mapped_column(String(64), nullable=False)
    partition_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    partition_month: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
