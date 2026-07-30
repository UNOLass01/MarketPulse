"""feature layer: features table, raw_ticks provenance

Revision ID: 0e3d4d3fc361
Revises: 009fb718fc66
Create Date: 2026-07-30 17:39:56.318525

"""

from collections.abc import Sequence
from datetime import UTC, date, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from marketpulse.storage.partitions import ensure_partitions_covering

# revision identifiers, used by Alembic.
revision: str = "0e3d4d3fc361"
down_revision: str | None = "009fb718fc66"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def _previous_month(today: date) -> date:
    return date(today.year - 1, 12, 1) if today.month == 1 else date(today.year, today.month - 1, 1)


def upgrade() -> None:
    op.add_column(
        "raw_ticks",
        sa.Column("source", sa.String(length=16), nullable=False, server_default="stream"),
    )
    op.create_check_constraint(
        "ck_raw_ticks_source", "raw_ticks", "source IN ('stream', 'seed')"
    )

    op.create_table(
        "features",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("feature_ts", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("symbol_id", sa.BigInteger(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("feature_set_version", sa.SmallInteger(), nullable=False),
        sa.Column("feature_values", postgresql.JSONB(), nullable=False),
        sa.Column("insufficient_history", sa.Boolean(), nullable=False),
        sa.Column("has_gap", sa.Boolean(), nullable=False),
        sa.Column(
            "computed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "symbol_id",
            "feature_ts",
            "feature_set_version",
            name="uq_features_symbol_ts_version",
        ),
        postgresql_partition_by="RANGE (feature_ts)",
    )
    op.create_index(
        "ix_features_symbol_ts_desc",
        "features",
        ["symbol_id", sa.text("feature_ts DESC")],
    )

    # Partitions must exist before any row can be inserted. Cover last month
    # through three months ahead, mirroring raw_ticks' partition horizon.
    connection = op.get_bind()
    start = _previous_month(datetime.now(UTC).date())
    ensure_partitions_covering(connection, "features", start, months_ahead=4)


def downgrade() -> None:
    op.drop_index("ix_features_symbol_ts_desc", table_name="features")
    op.drop_table("features")
    op.drop_constraint("ck_raw_ticks_source", "raw_ticks", type_="check")
    op.drop_column("raw_ticks", "source")
