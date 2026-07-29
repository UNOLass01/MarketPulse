"""streaming backbone: symbols and partitioned raw_ticks

Revision ID: 009fb718fc66
Revises:
Create Date: 2026-07-29 20:36:32.976478

"""

from collections.abc import Sequence
from datetime import UTC, date, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from marketpulse.storage.partitions import ensure_partitions_covering

# revision identifiers, used by Alembic.
revision: str = "009fb718fc66"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def _previous_month(today: date) -> date:
    return date(today.year - 1, 12, 1) if today.month == 1 else date(today.year, today.month - 1, 1)


def upgrade() -> None:
    op.create_table(
        "symbols",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("code", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("code", name="uq_symbols_code"),
    )

    op.create_table(
        "raw_ticks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("symbol_id", sa.BigInteger(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("schema_version", sa.SmallInteger(), nullable=False),
        sa.Column("emitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("price", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(24, 8), nullable=False),
        sa.UniqueConstraint("symbol_id", "observed_at", name="uq_raw_ticks_symbol_observed"),
        sa.UniqueConstraint("message_id", "observed_at", name="uq_raw_ticks_message_observed"),
        postgresql_partition_by="RANGE (observed_at)",
    )
    op.create_index(
        "ix_raw_ticks_observed_at_brin",
        "raw_ticks",
        ["observed_at"],
        postgresql_using="brin",
    )

    # Partitions must exist before any row can be inserted. Cover last month
    # through three months ahead; recurring partition creation for further
    # horizons is an operational task for Phase 4 (Airflow).
    connection = op.get_bind()
    start = _previous_month(datetime.now(UTC).date())
    ensure_partitions_covering(connection, start, months_ahead=4)


def downgrade() -> None:
    op.drop_index("ix_raw_ticks_observed_at_brin", table_name="raw_ticks")
    op.drop_table("raw_ticks")
    op.drop_table("symbols")
