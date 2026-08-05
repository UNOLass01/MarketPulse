"""MLflow registry integration: register -> transition -> resolve, plus the
post-training artifact-persisted assertion the phase-3 plan calls out by
name. Runs against a local sqlite-backed tracking store (no docker needed)
-- ``monkeypatch.chdir(tmp_path)`` keeps any default-artifact-root writes
inside the test's own tmp dir instead of the repo checkout.
"""

from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest

from marketpulse.config import MLflowSettings
from marketpulse.ml.registry import (
    STAGE_PRODUCTION,
    STAGE_STAGING,
    ArtifactNotPersistedError,
    configure,
    get_stage_version,
    load_production_model,
    log_model,
    start_run,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mlflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A fresh sqlite-backed tracking store per test, with the process cwd
    pointed at tmp_path so any default (relative) artifact root lands there.
    """
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _tiny_booster(seed: int = 0) -> tuple[lgb.Booster, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({f"f{i}": rng.random(30) for i in range(3)})
    y = rng.integers(0, 3, size=30)
    dataset = lgb.Dataset(x, label=y)
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbosity": -1}, dataset, num_boost_round=5
    )
    return booster, x


def _settings(uri: str, *, model_name: str, experiment: str) -> MLflowSettings:
    return MLflowSettings(
        tracking_uri=uri, registry_model_name=model_name, experiment_name=experiment
    )


# --- configure ---------------------------------------------------------------


def test_configure_sets_tracking_uri_and_creates_the_experiment(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="m1", experiment="exp-configure")
    configure(settings)
    assert mlflow.get_tracking_uri() == mlflow_env
    experiment = mlflow.get_experiment_by_name("exp-configure")
    assert experiment is not None


# --- log_model + artifact assertion -------------------------------------------


def test_log_model_requires_an_active_run(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="m2", experiment="exp-norun")
    configure(settings)
    booster, x = _tiny_booster()
    with pytest.raises(RuntimeError):
        log_model(
            booster,
            input_example=x,
            predictions_example=booster.predict(x),
            registered_model_name=settings.registry_model_name,
        )


def test_log_model_registers_version_one_on_first_call(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="m3", experiment="exp-log")
    configure(settings)
    booster, x = _tiny_booster()

    with start_run():
        logged = log_model(
            booster,
            input_example=x,
            predictions_example=booster.predict(x),
            registered_model_name=settings.registry_model_name,
        )

    assert logged.registered_name == "m3"
    assert logged.registered_version == "1"
    assert logged.model_uri.startswith("models:/")
    assert logged.run_id


def test_artifact_not_persisted_error_when_the_store_has_nothing(
    mlflow_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(mlflow_env, model_name="m4", experiment="exp-missing-artifact")
    configure(settings)
    booster, x = _tiny_booster()

    monkeypatch.setattr(mlflow.artifacts, "list_artifacts", lambda artifact_uri: [])

    with start_run(), pytest.raises(ArtifactNotPersistedError):
        log_model(
            booster,
            input_example=x,
            predictions_example=booster.predict(x),
            registered_model_name=settings.registry_model_name,
        )


# --- full round trip: register -> transition -> resolve ----------------------


def test_registry_round_trip_register_transition_resolve(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-roundtrip")
    client = configure(settings)
    booster, x = _tiny_booster()

    with start_run():
        logged = log_model(
            booster,
            input_example=x,
            predictions_example=booster.predict(x),
            registered_model_name=settings.registry_model_name,
        )

    # Nothing promoted yet.
    assert get_stage_version(client, settings.registry_model_name, STAGE_PRODUCTION) is None

    client.transition_model_version_stage(
        name=settings.registry_model_name,
        version=logged.registered_version,
        stage=STAGE_PRODUCTION,
        archive_existing_versions=True,
    )

    resolved_version = get_stage_version(client, settings.registry_model_name, STAGE_PRODUCTION)
    assert resolved_version == logged.registered_version

    loaded = load_production_model(settings.registry_model_name)
    predictions = loaded.predict(x)
    assert len(predictions) == len(x)


def test_promoting_a_new_version_archives_the_previous_production_version(
    mlflow_env: str,
) -> None:
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-archive")
    client = configure(settings)

    booster_v1, x = _tiny_booster(seed=1)
    with start_run():
        v1 = log_model(
            booster_v1,
            input_example=x,
            predictions_example=booster_v1.predict(x),
            registered_model_name=settings.registry_model_name,
        )
    client.transition_model_version_stage(
        name=settings.registry_model_name,
        version=v1.registered_version,
        stage=STAGE_PRODUCTION,
    )

    booster_v2, x2 = _tiny_booster(seed=2)
    with start_run():
        v2 = log_model(
            booster_v2,
            input_example=x2,
            predictions_example=booster_v2.predict(x2),
            registered_model_name=settings.registry_model_name,
        )
    client.transition_model_version_stage(
        name=settings.registry_model_name,
        version=v2.registered_version,
        stage=STAGE_PRODUCTION,
        archive_existing_versions=True,
    )

    assert get_stage_version(client, settings.registry_model_name, STAGE_PRODUCTION) == "2"
    archived = client.get_model_version(settings.registry_model_name, v1.registered_version)
    assert archived.current_stage == "Archived"


def test_get_stage_version_is_none_for_a_never_registered_model(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="does-not-exist", experiment="exp-none")
    client = configure(settings)
    assert get_stage_version(client, "totally-unregistered-model", STAGE_PRODUCTION) is None


def test_rejected_candidate_can_be_kept_in_staging_without_promotion(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-staging")
    client = configure(settings)
    booster, x = _tiny_booster()

    with start_run():
        logged = log_model(
            booster,
            input_example=x,
            predictions_example=booster.predict(x),
            registered_model_name=settings.registry_model_name,
        )
    client.transition_model_version_stage(
        name=settings.registry_model_name, version=logged.registered_version, stage=STAGE_STAGING
    )

    assert get_stage_version(client, settings.registry_model_name, STAGE_STAGING) == "1"
    assert get_stage_version(client, settings.registry_model_name, STAGE_PRODUCTION) is None
