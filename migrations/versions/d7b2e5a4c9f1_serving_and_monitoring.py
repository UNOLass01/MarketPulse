"""serving and monitoring: predictions, outcomes, drift_metrics, alerts

Phases 5 and 6. Four tables, added together because they are one causal
chain: the API logs a prediction, the attribution DAG resolves it into an
outcome, drift and outcome metrics feed the alert evaluator.

Revision ID: d7b2e5a4c9f1
Revises: c4a9e1f6b7d3
Create Date: 2026-08-17 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d7b2e5a4c9f1"
down_revision: str | None = "c4a9e1f6b7d3"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    # --- predictions ------------------------------------------------------
    # Not partitioned, unlike raw_ticks/features: the unique constraint means
    # one row per (symbol, feature_ts, model) rather than one per request, so
    # growth is bounded by tick cadence rather than by API traffic.
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("feature_set_version", sa.SmallInteger(), nullable=False),
        sa.Column("feature_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.String(length=8), nullable=False),
        sa.Column("probabilities", postgresql.JSONB(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "symbol_id", "feature_ts", "model_version", name="uq_predictions_symbol_ts_model"
        ),
        sa.CheckConstraint("label IN ('DOWN', 'STABLE', 'UP')", name="ck_predictions_label"),
    )
    op.create_index("ix_predictions_predicted_at", "predictions", ["predicted_at"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])
    op.create_index(
        "ix_predictions_symbol_predicted_at_desc",
        "predictions",
        ["symbol_id", sa.text("predicted_at DESC")],
    )

    # --- prediction_outcomes ---------------------------------------------
    op.create_table(
        "prediction_outcomes",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "prediction_id", sa.BigInteger(), sa.ForeignKey("predictions.id"), nullable=False
        ),
        sa.Column("horizon_minutes", sa.Float(), nullable=False),
        sa.Column("theta", sa.Float(), nullable=False),
        sa.Column("base_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("future_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("future_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("realised_return", sa.Float(), nullable=False),
        sa.Column("actual_label", sa.String(length=8), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column(
            "resolved_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("prediction_id", name="uq_prediction_outcomes_prediction"),
        sa.CheckConstraint(
            "actual_label IN ('DOWN', 'STABLE', 'UP')", name="ck_prediction_outcomes_label"
        ),
    )
    op.create_index(
        "ix_prediction_outcomes_resolved_at", "prediction_outcomes", ["resolved_at"]
    )

    # --- drift_metrics ----------------------------------------------------
    # Long, not wide: one row per (feature, metric, window). Adding a feature
    # to features.registry needs no migration here.
    op.create_table(
        "drift_metrics",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("metric_name", sa.String(length=16), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("reference_model_version", sa.String(length=32), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "computed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "feature_name",
            "metric_name",
            "window_start",
            "window_end",
            "reference_model_version",
            name="uq_drift_metrics_feature_metric_window",
        ),
        sa.CheckConstraint(
            "severity IN ('stable', 'moderate', 'significant')", name="ck_drift_metrics_severity"
        ),
    )
    op.create_index("ix_drift_metrics_computed_at", "drift_metrics", ["computed_at"])
    op.create_index(
        "ix_drift_metrics_feature_window", "drift_metrics", ["feature_name", "window_end"]
    )

    # --- alerts -----------------------------------------------------------
    # runbook is NOT NULL: an alert with no attached action is just anxiety,
    # so the schema refuses to store one.
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("alert_name", sa.String(length=64), nullable=False),
        sa.Column("dedup_key", sa.String(length=128), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("runbook", sa.String(length=128), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("consecutive_breaches", sa.Integer(), nullable=False),
        sa.Column("first_breached_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("status IN ('open', 'resolved')", name="ck_alerts_status"),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'critical')", name="ck_alerts_severity"
        ),
    )
    op.create_index("ix_alerts_dedup_status", "alerts", ["dedup_key", "status"])
    op.create_index("ix_alerts_first_breached_at", "alerts", ["first_breached_at"])


def downgrade() -> None:
    # Reverse dependency order: prediction_outcomes references predictions.
    op.drop_table("alerts")
    op.drop_table("drift_metrics")
    op.drop_table("prediction_outcomes")
    op.drop_table("predictions")
