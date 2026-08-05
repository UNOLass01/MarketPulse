"""``training_runs`` / ``model_versions`` persistence -- including a
rejected run, which the phase-3 plan requires to be recorded, not discarded.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.ml_registry import (
    archive_model_version,
    get_current_production_version,
    get_model_version,
    get_training_run,
    record_model_version,
    record_training_run,
)

pytestmark = pytest.mark.integration


def _run_kwargs(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "mlflow_run_id": "run-abc123",
        "feature_set_version": 1,
        "config_version": 1,
        "horizon_minutes": 15.0,
        "theta": 0.02,
        "train_row_count": 1000,
        "validation_row_count": 200,
        "test_row_count": 200,
        "train_class_distribution": {"DOWN": 300, "STABLE": 400, "UP": 300},
        "window_start": now - timedelta(days=30),
        "window_end": now,
        "candidate_metrics": {"accuracy": 0.5, "macro_f1": 0.48},
        "baseline_metrics": {"majority": {"macro_f1": 0.3}},
        "incumbent_metrics": None,
        "promoted": False,
        "rejection_reason": None,
        "git_sha": "a" * 40,
        "started_at": now - timedelta(minutes=5),
        "finished_at": now,
    }
    base.update(overrides)
    return base


def test_record_and_get_training_run_round_trips(session_factory: sessionmaker[Session]) -> None:
    with session_scope(session_factory) as session:
        run_id = record_training_run(session, **_run_kwargs())  # type: ignore[arg-type]

    with session_factory() as session:
        run = get_training_run(session, run_id)

    assert run is not None
    assert run.mlflow_run_id == "run-abc123"
    assert run.train_class_distribution == {"DOWN": 300, "STABLE": 400, "UP": 300}
    assert run.promoted is False


def test_rejected_run_is_recorded_with_its_reason(session_factory: sessionmaker[Session]) -> None:
    with session_scope(session_factory) as session:
        run_id = record_training_run(
            session,
            **_run_kwargs(
                mlflow_run_id="run-rejected",
                promoted=False,
                rejection_reason="does not beat baseline 'majority'",
            ),  # type: ignore[arg-type]
        )

    with session_factory() as session:
        run = get_training_run(session, run_id)

    assert run is not None
    assert run.promoted is False
    assert run.rejection_reason == "does not beat baseline 'majority'"


def test_rejected_run_still_registers_a_staging_model_version(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        run_id = record_training_run(
            session, **_run_kwargs(mlflow_run_id="run-staging")
        )  # type: ignore[arg-type]
        version_id = record_model_version(
            session,
            training_run_id=run_id,
            mlflow_model_name="marketpulse",
            mlflow_model_version="7",
            stage="Staging",
        )

    with session_factory() as session:
        version = get_model_version(session, version_id)

    assert version is not None
    assert version.stage == "Staging"
    assert version.promoted_at is None
    assert version.reference_feature_stats is None


def test_promoted_run_registers_a_production_model_version_with_reference_stats(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        run_id = record_training_run(
            session, **_run_kwargs(mlflow_run_id="run-promoted", promoted=True)
        )  # type: ignore[arg-type]
        version_id = record_model_version(
            session,
            training_run_id=run_id,
            mlflow_model_name="marketpulse",
            mlflow_model_version="8",
            stage="Production",
            reference_feature_stats={"ma_5m": {"mean": 100.0, "std": 5.0}},
            promoted_at=now,
        )

    with session_factory() as session:
        version = get_model_version(session, version_id)

    assert version is not None
    assert version.stage == "Production"
    assert version.reference_feature_stats == {"ma_5m": {"mean": 100.0, "std": 5.0}}
    assert version.promoted_at is not None


def test_get_current_production_version_is_none_when_nothing_promoted(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        version = get_current_production_version(session, "marketpulse")
    assert version is None


def test_get_current_production_version_ignores_staging_versions(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        run_id = record_training_run(session, **_run_kwargs())  # type: ignore[arg-type]
        record_model_version(
            session,
            training_run_id=run_id,
            mlflow_model_name="marketpulse",
            mlflow_model_version="1",
            stage="Staging",
        )

    with session_factory() as session:
        version = get_current_production_version(session, "marketpulse")
    assert version is None


def test_get_current_production_version_returns_the_latest_promoted(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        run_id = record_training_run(session, **_run_kwargs())  # type: ignore[arg-type]
        record_model_version(
            session,
            training_run_id=run_id,
            mlflow_model_name="marketpulse",
            mlflow_model_version="1",
            stage="Production",
            promoted_at=now - timedelta(days=1),
        )
        record_model_version(
            session,
            training_run_id=run_id,
            mlflow_model_name="marketpulse",
            mlflow_model_version="2",
            stage="Production",
            promoted_at=now,
        )

    with session_factory() as session:
        version = get_current_production_version(session, "marketpulse")

    assert version is not None
    assert version.mlflow_model_version == "2"


def test_archive_model_version_sets_stage_and_timestamp(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_scope(session_factory) as session:
        run_id = record_training_run(session, **_run_kwargs())  # type: ignore[arg-type]
        version_id = record_model_version(
            session,
            training_run_id=run_id,
            mlflow_model_name="marketpulse",
            mlflow_model_version="1",
            stage="Production",
            promoted_at=now - timedelta(days=1),
        )

    with session_scope(session_factory) as session:
        archive_model_version(session, version_id, archived_at=now)

    with session_factory() as session:
        version = get_model_version(session, version_id)

    assert version is not None
    assert version.stage == "Archived"
    assert version.archived_at is not None


def test_archive_model_version_raises_for_unknown_id(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session, pytest.raises(ValueError):
        archive_model_version(session, 999_999, archived_at=datetime.now(UTC))
