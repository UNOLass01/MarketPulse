"""ml pipeline: training_runs and model_versions

Revision ID: a1f7c9d3e8b2
Revises: 0e3d4d3fc361
Create Date: 2026-08-05 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1f7c9d3e8b2"
down_revision: str | None = "0e3d4d3fc361"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=False),
        sa.Column("feature_set_version", sa.SmallInteger(), nullable=False),
        sa.Column("config_version", sa.SmallInteger(), nullable=False),
        sa.Column("horizon_minutes", sa.Float(), nullable=False),
        sa.Column("theta", sa.Float(), nullable=False),
        sa.Column("train_row_count", sa.Integer(), nullable=False),
        sa.Column("validation_row_count", sa.Integer(), nullable=False),
        sa.Column("test_row_count", sa.Integer(), nullable=False),
        sa.Column("train_class_distribution", postgresql.JSONB(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_metrics", postgresql.JSONB(), nullable=False),
        sa.Column("baseline_metrics", postgresql.JSONB(), nullable=False),
        sa.Column("incumbent_metrics", postgresql.JSONB(), nullable=True),
        sa.Column("promoted", sa.Boolean(), nullable=False),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("git_sha", sa.String(length=40), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("mlflow_run_id", name="uq_training_runs_mlflow_run_id"),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "training_run_id",
            sa.BigInteger(),
            sa.ForeignKey("training_runs.id"),
            nullable=False,
        ),
        sa.Column("mlflow_model_name", sa.String(length=128), nullable=False),
        sa.Column("mlflow_model_version", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("reference_feature_stats", postgresql.JSONB(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "mlflow_model_name", "mlflow_model_version", name="uq_model_versions_name_version"
        ),
        sa.CheckConstraint(
            "stage IN ('Staging', 'Production', 'Archived')", name="ck_model_versions_stage"
        ),
    )
    op.create_index("ix_model_versions_stage", "model_versions", ["stage"])


def downgrade() -> None:
    op.drop_index("ix_model_versions_stage", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_table("training_runs")
