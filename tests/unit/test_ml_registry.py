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
from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from marketpulse.ml.predict import ModelSignatureMismatchError
from marketpulse.ml.registry import (
    STAGE_PRODUCTION,
    STAGE_STAGING,
    ArtifactNotPersistedError,
    configure,
    get_stage_version,
    load_production_bundle,
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


# --- serving: load_production_bundle (Phase 5) -------------------------------
#
# This is the seam the phase-5 exit criterion runs through -- "promotion
# changes model_version with no redeploy" is exactly this function returning a
# different LoadedModel on the next poll. It is also the only place the
# serving-time schema guard gets its numbers, so both the happy path and each
# refusal are pinned here rather than inferred from the API tests.


def _registry_booster(seed: int = 0) -> tuple[lgb.Booster, pd.DataFrame]:
    """A booster whose columns really are ``features.registry.FEATURE_NAMES``.

    ``_tiny_booster``'s f0/f1/f2 columns deliberately do *not* match the
    registry -- that is what makes it useful for the mismatch test below.
    """
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({name: rng.random(60) for name in FEATURE_NAMES})
    y = rng.integers(0, 3, size=60)
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbosity": -1},
        lgb.Dataset(x, label=y),
        num_boost_round=5,
    )
    return booster, x


def _promote(client, settings, *, feature_set_version: int | None, booster_and_x) -> str:
    booster, x = booster_and_x
    with start_run():
        if feature_set_version is not None:
            mlflow.log_param("feature_set_version", feature_set_version)
        logged = log_model(
            booster,
            input_example=x,
            predictions_example=booster.predict(x),
            registered_model_name=settings.registry_model_name,
        )
    client.transition_model_version_stage(
        name=settings.registry_model_name,
        version=logged.registered_version,
        stage=STAGE_PRODUCTION,
        archive_existing_versions=True,
    )
    return logged.registered_version


def test_load_production_bundle_is_none_when_nothing_is_promoted(mlflow_env: str) -> None:
    # None, not an exception: the API must start against an empty registry
    # and report 503 on /ready rather than crash-looping.
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-bundle-empty")
    client = configure(settings)
    assert load_production_bundle(client, settings.registry_model_name) is None


def test_load_production_bundle_returns_version_schema_and_signature(mlflow_env: str) -> None:
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-bundle")
    client = configure(settings)
    version = _promote(client, settings, feature_set_version=7, booster_and_x=_registry_booster())

    bundle = load_production_bundle(client, settings.registry_model_name)

    assert bundle is not None
    assert bundle.version == version
    # Read off the run param ml.pipeline logs, not assumed from the registry.
    assert bundle.feature_set_version == 7
    assert bundle.feature_names == FEATURE_NAMES
    assert bundle.run_id


def test_load_production_bundle_falls_back_to_the_current_feature_set_version(
    mlflow_env: str,
) -> None:
    # A model registered before the param existed. The fallback is the
    # optimistic branch, so it only applies when the param is genuinely
    # absent -- a present-but-different value must always win.
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-bundle-nofsv")
    client = configure(settings)
    _promote(client, settings, feature_set_version=None, booster_and_x=_registry_booster())

    bundle = load_production_bundle(client, settings.registry_model_name)

    assert bundle is not None
    assert bundle.feature_set_version == FEATURE_SET_VERSION


def test_a_model_whose_signature_is_not_the_registry_is_refused_at_load_time(
    mlflow_env: str,
) -> None:
    # Never per request: a model with the wrong columns would feed every
    # value into the wrong slot and still return a confident probability
    # vector, so it must not reach the serving path at all.
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-bundle-badsig")
    client = configure(settings)
    _promote(client, settings, feature_set_version=1, booster_and_x=_tiny_booster())

    with pytest.raises(ModelSignatureMismatchError):
        load_production_bundle(client, settings.registry_model_name)


def test_promoting_a_new_version_changes_what_the_bundle_resolves_to(
    mlflow_env: str,
) -> None:
    """The exit criterion at the registry seam: no process restart involved."""
    settings = _settings(mlflow_env, model_name="marketpulse", experiment="exp-bundle-swap")
    client = configure(settings)

    first = _promote(client, settings, feature_set_version=1, booster_and_x=_registry_booster(0))
    assert load_production_bundle(client, settings.registry_model_name).version == first

    second = _promote(client, settings, feature_set_version=1, booster_and_x=_registry_booster(1))

    assert second != first
    assert load_production_bundle(client, settings.registry_model_name).version == second
