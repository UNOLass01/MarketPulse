"""Persistence for training runs and registered model versions.

Every ``ml.pipeline`` invocation writes exactly one :class:`TrainingRun` row
-- including a rejected one (CLAUDE.md / phase-3 plan: "record rejections
too"; a rejected candidate is a working pipeline, not a failure). Re-running
the pipeline creates a *new* row rather than mutating a prior one, matching
Phase 4's idempotency requirement for the retraining DAG.
"""

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketpulse.storage.models import ModelVersion, TrainingRun


def record_training_run(
    session: Session,
    *,
    mlflow_run_id: str,
    feature_set_version: int,
    config_version: int,
    horizon_minutes: float,
    theta: float,
    train_row_count: int,
    validation_row_count: int,
    test_row_count: int,
    train_class_distribution: Mapping[str, int],
    window_start: datetime,
    window_end: datetime,
    candidate_metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
    incumbent_metrics: Mapping[str, object] | None,
    promoted: bool,
    rejection_reason: str | None,
    git_sha: str | None,
    started_at: datetime,
    finished_at: datetime,
) -> int:
    """Insert one training run record and return its id."""
    run = TrainingRun(
        mlflow_run_id=mlflow_run_id,
        feature_set_version=feature_set_version,
        config_version=config_version,
        horizon_minutes=horizon_minutes,
        theta=theta,
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
        test_row_count=test_row_count,
        train_class_distribution=dict(train_class_distribution),
        window_start=window_start,
        window_end=window_end,
        candidate_metrics=dict(candidate_metrics),
        baseline_metrics=dict(baseline_metrics),
        incumbent_metrics=dict(incumbent_metrics) if incumbent_metrics is not None else None,
        promoted=promoted,
        rejection_reason=rejection_reason,
        git_sha=git_sha,
        started_at=started_at,
        finished_at=finished_at,
    )
    session.add(run)
    session.flush()
    return run.id


def record_model_version(
    session: Session,
    *,
    training_run_id: int,
    mlflow_model_name: str,
    mlflow_model_version: str,
    stage: str,
    reference_feature_stats: Mapping[str, object] | None = None,
    promoted_at: datetime | None = None,
) -> int:
    """Insert one model version record and return its id."""
    version = ModelVersion(
        training_run_id=training_run_id,
        mlflow_model_name=mlflow_model_name,
        mlflow_model_version=mlflow_model_version,
        stage=stage,
        reference_feature_stats=(
            dict(reference_feature_stats) if reference_feature_stats is not None else None
        ),
        promoted_at=promoted_at,
    )
    session.add(version)
    session.flush()
    return version.id


def archive_model_version(
    session: Session, model_version_id: int, *, archived_at: datetime
) -> None:
    """Mark a version Archived -- called on the previous Production version
    when a new one is promoted (phase-4 plan: "archive the previous
    Production version on promotion").
    """
    version = session.get(ModelVersion, model_version_id)
    if version is None:
        raise ValueError(f"no model_version with id={model_version_id}")
    version.stage = "Archived"
    version.archived_at = archived_at


def get_current_production_version(session: Session, mlflow_model_name: str) -> ModelVersion | None:
    """The single row currently marked Production for ``mlflow_model_name``,
    or ``None`` if nothing has ever been promoted.
    """
    stmt = (
        select(ModelVersion)
        .where(
            ModelVersion.mlflow_model_name == mlflow_model_name,
            ModelVersion.stage == "Production",
        )
        .order_by(ModelVersion.promoted_at.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def get_model_version(session: Session, model_version_id: int) -> ModelVersion | None:
    return session.get(ModelVersion, model_version_id)


def get_training_run(session: Session, training_run_id: int) -> TrainingRun | None:
    return session.get(TrainingRun, training_run_id)
