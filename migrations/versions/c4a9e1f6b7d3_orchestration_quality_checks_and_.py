"""orchestration: quality_checks and archived_partitions

Revision ID: c4a9e1f6b7d3
Revises: a1f7c9d3e8b2
Create Date: 2026-08-06 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c4a9e1f6b7d3"
down_revision: str | None = "a1f7c9d3e8b2"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "quality_checks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("check_name", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "checked_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "check_name IN ('freshness', 'completeness', 'validity', 'distribution')",
            name="ck_quality_checks_check_name",
        ),
    )
    op.create_index(
        "ix_quality_checks_name_checked_at", "quality_checks", ["check_name", "checked_at"]
    )

    op.create_table(
        "archived_partitions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("table_name", sa.String(length=64), nullable=False),
        sa.Column("partition_year", sa.SmallInteger(), nullable=False),
        sa.Column("partition_month", sa.SmallInteger(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column(
            "archived_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "table_name", "partition_year", "partition_month", name="uq_archived_partitions"
        ),
    )


def downgrade() -> None:
    op.drop_table("archived_partitions")
    op.drop_index("ix_quality_checks_name_checked_at", table_name="quality_checks")
    op.drop_table("quality_checks")
