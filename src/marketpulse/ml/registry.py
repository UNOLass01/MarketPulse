"""MLflow tracking + model registry integration.

The only module in ``marketpulse.ml`` that imports ``mlflow`` directly --
every other module here computes on plain arrays/DataFrames/dataclasses and
stays testable without a tracking server, the same separation
``storage/repositories`` gives the rest of the codebase for SQL.

CLAUDE.md: "never train without MLflow reachable" -- functions here fail
loudly (``mlflow`` itself raises on a connection failure) rather than
letting a run proceed against nothing. ``_assert_artifact_persisted`` exists
specifically for the "metadata logging silently succeeding while the
artifact write fails" trap the phase-3 plan calls out by name.

Model registry **stages** (Staging/Production/Archived) are used throughout,
per this project's exit criterion ("a model sits in MLflow Production
stage"), even though MLflow has deprecated stages in favor of aliases as of
2.9 -- the stage-based API remains fully functional and is what the phase
plan asks for; revisit via an ADR if a future MLflow major release removes it.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from marketpulse.config import MLflowSettings
from marketpulse.exceptions import TransientError
from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from marketpulse.ml.predict import LoadedModel, validate_feature_names

STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"


class ArtifactNotPersistedError(TransientError):
    """No artifact was found in the tracking server's store after logging.

    Retryable: it usually means the artifact-store side (S3/MinIO) had a
    transient failure while the tracking server's metadata write succeeded.
    """


@dataclass(frozen=True)
class LoggedModel:
    run_id: str
    model_uri: str
    registered_name: str
    registered_version: str


def configure(settings: MLflowSettings) -> MlflowClient:
    """Point the fluent API + a fresh client at the tracking server and
    ensure the experiment exists. Call once per process before any other
    function in this module.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    return MlflowClient(tracking_uri=settings.tracking_uri)


@contextmanager
def start_run(*, run_name: str | None = None) -> Iterator[Run]:
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    mlflow.log_params(dict(params))


def log_metrics(metrics: Mapping[str, float]) -> None:
    mlflow.log_metrics(dict(metrics))


def log_dict_artifact(data: Mapping[str, Any], artifact_file: str) -> None:
    """JSON-serialize ``data`` (confusion matrices, baseline metrics, feature
    importances, ...) as a run artifact."""
    mlflow.log_dict(dict(data), artifact_file)


def _assert_artifact_persisted(model_uri: str) -> None:
    listing = mlflow.artifacts.list_artifacts(artifact_uri=model_uri)
    if not listing:
        raise ArtifactNotPersistedError(
            f"no artifact files found at {model_uri} after logging -- the run's "
            "metadata was recorded but the model artifact did not persist"
        )


def log_model(
    booster: lgb.Booster,
    *,
    input_example: pd.DataFrame,
    predictions_example: Any,
    registered_model_name: str,
) -> LoggedModel:
    """Log ``booster`` with a signature inferred from real train-split data,
    register it, and assert the artifact is actually retrievable from the
    store before returning -- never trust the metadata write alone.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("registry.log_model must be called inside an active MLflow run")

    signature = infer_signature(input_example, predictions_example)
    model_info = mlflow.lightgbm.log_model(
        booster,
        name="model",
        signature=signature,
        registered_model_name=registered_model_name,
    )
    _assert_artifact_persisted(model_info.model_uri)

    if model_info.registered_model_version is None:
        raise ArtifactNotPersistedError(
            f"model logged at {model_info.model_uri} but was not registered under "
            f"'{registered_model_name}'"
        )
    return LoggedModel(
        run_id=run.info.run_id,
        model_uri=model_info.model_uri,
        registered_name=registered_model_name,
        registered_version=str(model_info.registered_model_version),
    )


def transition_stage(
    client: MlflowClient,
    *,
    name: str,
    version: str,
    stage: str,
    archive_existing: bool = False,
) -> None:
    client.transition_model_version_stage(
        name=name, version=version, stage=stage, archive_existing_versions=archive_existing
    )


def get_stage_version(client: MlflowClient, model_name: str, stage: str) -> str | None:
    """The version currently in ``stage`` for ``model_name``, or ``None`` if
    the model has never been registered or nothing is in that stage.
    """
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
    except MlflowException:
        return None
    # MlflowClient returns ``version`` as an int despite ModelVersion.version
    # being typed/used as a string everywhere else in this module (and in
    # storage.models.ModelVersion.mlflow_model_version) -- normalize here.
    return str(versions[0].version) if versions else None


def load_production_model(model_name: str) -> Any:
    """Load the current ``Production``-stage model via the ``models:/`` URI
    -- this is what serving (Phase 5) resolves too, so evaluation against
    "the incumbent" uses exactly what's live.
    """
    return mlflow.pyfunc.load_model(f"models:/{model_name}/{STAGE_PRODUCTION}")


def _signature_feature_names(model: Any) -> tuple[str, ...]:
    """Input column names from the model's logged signature.

    Falls back to the registry's own order when a model was logged without a
    signature. That fallback is safe only because
    ``predict.validate_feature_names`` still runs on the result -- it just
    means an unsigned model is checked against the registry rather than
    against its own (absent) declaration.
    """
    try:
        schema = model.metadata.get_input_schema()
    except (AttributeError, MlflowException):
        return FEATURE_NAMES
    if schema is None:
        return FEATURE_NAMES
    names = schema.input_names()
    return tuple(names) if names else FEATURE_NAMES


def _feature_set_version_of(client: MlflowClient, name: str, version: str) -> int:
    """Read ``feature_set_version`` off the run that produced this model
    version -- ``ml.pipeline`` logs it as a run param on every run.

    Falls back to the *current* ``FEATURE_SET_VERSION`` only when the param
    is absent (a model registered before the param existed). That fallback
    is the optimistic branch, so it is deliberately narrow: a present-but-
    different value must always win, since detecting exactly that difference
    is what the serving schema guard exists for.
    """
    model_version = client.get_model_version(name, version)
    run_id = getattr(model_version, "run_id", None)
    if not run_id:
        return FEATURE_SET_VERSION
    raw = client.get_run(run_id).data.params.get("feature_set_version")
    return int(raw) if raw is not None else FEATURE_SET_VERSION


def load_production_bundle(client: MlflowClient, model_name: str) -> LoadedModel | None:
    """Resolve the Production model plus the metadata serving's guards need.

    Returns ``None`` -- not an exception -- when nothing is in Production, so
    the API can start against an empty registry and report 503 on ``/ready``
    instead of crash-looping (phase-5 plan). Genuine failures (tracking
    server unreachable, artifact missing) still raise, because those are
    exactly what ``ModelCache.refresh`` must distinguish from "not promoted
    yet" in order to keep serving the model it already holds.
    """
    version = get_stage_version(client, model_name, STAGE_PRODUCTION)
    if version is None:
        return None

    model = load_production_model(model_name)
    feature_names = _signature_feature_names(model)
    validate_feature_names(feature_names)

    model_version = client.get_model_version(model_name, version)
    return LoadedModel(
        model=model,
        version=version,
        feature_set_version=_feature_set_version_of(client, model_name, version),
        feature_names=feature_names,
        run_id=getattr(model_version, "run_id", None),
    )
