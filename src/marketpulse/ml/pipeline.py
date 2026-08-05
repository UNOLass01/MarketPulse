"""Training pipeline orchestration: the one place that wires dataset
assembly, training, evaluation, MLflow, and Postgres persistence together.

Airflow's ``dag_model_retraining`` (Phase 4) is meant to call
``run_training_pipeline`` and nothing else -- CLAUDE.md rule #9 ("no business
logic in airflow/dags/") depends on every real decision living here, not in
a DAG callable.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import Settings
from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from marketpulse.ml import evaluate, registry
from marketpulse.ml.config import TrainingConfig, load_training_config
from marketpulse.ml.dataset import Dataset, SplitConfig, assemble_dataset
from marketpulse.ml.evaluate import Metrics, PromotionResult
from marketpulse.ml.labeling import INDEX_TO_LABEL
from marketpulse.ml.train import TrainedModel, predict_labels, predict_proba, train_model
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.ml_registry import (
    archive_model_version,
    get_current_production_version,
    record_model_version,
    record_training_run,
)
from marketpulse.storage.repositories.training_data import fetch_feature_rows


@dataclass(frozen=True)
class PipelineResult:
    training_run_id: int
    mlflow_run_id: str
    model_version_id: int
    mlflow_model_version: str
    promoted: bool
    rejection_reason: str | None
    candidate_metrics: Metrics


def _git_sha() -> str | None:
    """Best-effort current commit SHA, logged for reproducibility. ``None``
    (never a crash) if git isn't available -- e.g. inside a stripped image.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def _flatten_config(config: TrainingConfig) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flatten("", config.model_dump(), flat)
    return {key.lstrip("."): value for key, value in flat.items()}


def _predict_with_incumbent(incumbent: Any, test_frame: Any) -> tuple[np.ndarray, np.ndarray]:
    proba = np.asarray(incumbent.predict(test_frame[list(FEATURE_NAMES)]))
    labels = np.array([INDEX_TO_LABEL[i] for i in proba.argmax(axis=1)])
    return labels, proba


def run_training_pipeline(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    training_config: TrainingConfig | None = None,
    symbols: Sequence[str] | None = None,
) -> PipelineResult:
    """Extract -> assemble -> train -> evaluate -> promotion gate ->
    (promote + archive incumbent + snapshot reference | hold in Staging with
    a recorded reason) -> persist. Runs to completion and records a result
    either way -- a rejected candidate is a working pipeline (CLAUDE.md).
    """
    started_at = datetime.now(UTC)
    config = training_config or load_training_config()
    git_sha = _git_sha()

    with session_factory() as session:
        rows = fetch_feature_rows(session, feature_set_version=FEATURE_SET_VERSION, symbols=symbols)

    dataset = assemble_dataset(
        rows,
        horizon=config.labeling.horizon,
        feature_set_version=FEATURE_SET_VERSION,
        theta=config.labeling.theta,
        theta_quantile=config.labeling.theta_quantile,
        split=SplitConfig(
            train_frac=config.split.train_frac,
            val_frac=config.split.val_frac,
            min_rows_per_split=config.split.min_rows_per_split,
            min_class_frac=config.split.min_class_frac,
        ),
        forward_price_tolerance=timedelta(seconds=settings.features.gap_threshold_seconds),
    )

    model = train_model(dataset, config.lightgbm)

    candidate_metrics = _evaluate_split(model, dataset.test, config.evaluation.time_buckets)
    baseline_metrics = _evaluate_baselines(dataset, config)

    client = registry.configure(settings.mlflow)
    incumbent_metrics = _evaluate_incumbent(
        client, settings.mlflow.registry_model_name, dataset, config.evaluation.time_buckets
    )

    decision = evaluate.decide_promotion(candidate_metrics, baseline_metrics, incumbent_metrics)

    mlflow_run = _log_to_mlflow(
        model,
        dataset,
        config,
        candidate_metrics,
        baseline_metrics,
        decision,
        git_sha,
        model_name=settings.mlflow.registry_model_name,
    )

    stage = registry.STAGE_PRODUCTION if decision.promote else registry.STAGE_STAGING
    registry.transition_stage(
        client,
        name=settings.mlflow.registry_model_name,
        version=mlflow_run.registered_version,
        stage=stage,
        archive_existing=decision.promote,
    )

    finished_at = datetime.now(UTC)
    with session_scope(session_factory) as session:
        previous_production = (
            get_current_production_version(session, settings.mlflow.registry_model_name)
            if decision.promote
            else None
        )

        training_run_id = record_training_run(
            session,
            mlflow_run_id=mlflow_run.run_id,
            feature_set_version=FEATURE_SET_VERSION,
            config_version=config.config_version,
            horizon_minutes=config.labeling.horizon_minutes,
            theta=dataset.theta,
            train_row_count=len(dataset.train),
            validation_row_count=len(dataset.validation),
            test_row_count=len(dataset.test),
            train_class_distribution=dataset.class_distribution("train"),
            window_start=dataset.train["feature_ts"].min().to_pydatetime(),
            window_end=dataset.test["feature_ts"].max().to_pydatetime(),
            candidate_metrics=asdict(candidate_metrics),
            baseline_metrics={name: asdict(m) for name, m in baseline_metrics.items()},
            incumbent_metrics=asdict(incumbent_metrics) if incumbent_metrics else None,
            promoted=decision.promote,
            rejection_reason=decision.rejection_reason,
            git_sha=git_sha,
            started_at=started_at,
            finished_at=finished_at,
        )

        model_version_id = record_model_version(
            session,
            training_run_id=training_run_id,
            mlflow_model_name=settings.mlflow.registry_model_name,
            mlflow_model_version=mlflow_run.registered_version,
            stage=stage,
            reference_feature_stats=dataset.reference_feature_stats() if decision.promote else None,
            promoted_at=finished_at if decision.promote else None,
        )

        if decision.promote and previous_production is not None:
            archive_model_version(session, previous_production.id, archived_at=finished_at)

    return PipelineResult(
        training_run_id=training_run_id,
        mlflow_run_id=mlflow_run.run_id,
        model_version_id=model_version_id,
        mlflow_model_version=mlflow_run.registered_version,
        promoted=decision.promote,
        rejection_reason=decision.rejection_reason,
        candidate_metrics=candidate_metrics,
    )


def _evaluate_split(model: TrainedModel, frame: Any, time_buckets: int) -> Metrics:
    predicted = predict_labels(model, frame)
    proba = predict_proba(model, frame)
    return evaluate.compute_metrics(
        frame["label"].tolist(),
        predicted.tolist(),
        proba,
        timestamps=frame["feature_ts"],
        time_buckets=time_buckets,
    )


def _evaluate_baselines(dataset: Dataset, config: TrainingConfig) -> dict[str, Metrics]:
    train_labels = dataset.train["label"].tolist()
    test_labels = dataset.test["label"].tolist()
    test_size = len(dataset.test)
    buckets = config.evaluation.time_buckets

    majority = evaluate.majority_baseline(train_labels, test_size)
    random_stratified = evaluate.random_stratified_baseline(
        train_labels, test_size, seed=config.evaluation.random_baseline_seed
    )
    history = dataset.train["label"].tolist() + dataset.validation["label"].tolist()
    history_ts = dataset.train["feature_ts"].tolist() + dataset.validation["feature_ts"].tolist()
    persistence = evaluate.persistence_baseline(
        history_ts,
        history,
        dataset.test["feature_ts"].tolist(),
        horizon=dataset.horizon,
        fallback_label=evaluate.majority_class(train_labels),
    )

    return {
        "majority": evaluate.compute_metrics(
            test_labels, majority.labels.tolist(), majority.proba, time_buckets=buckets
        ),
        "persistence": evaluate.compute_metrics(
            test_labels, persistence.labels.tolist(), persistence.proba, time_buckets=buckets
        ),
        "random_stratified": evaluate.compute_metrics(
            test_labels,
            random_stratified.labels.tolist(),
            random_stratified.proba,
            time_buckets=buckets,
        ),
    }


def _evaluate_incumbent(
    client: Any, model_name: str, dataset: Dataset, time_buckets: int
) -> Metrics | None:
    if registry.get_stage_version(client, model_name, registry.STAGE_PRODUCTION) is None:
        return None
    incumbent = registry.load_production_model(model_name)
    labels, proba = _predict_with_incumbent(incumbent, dataset.test)
    return evaluate.compute_metrics(
        dataset.test["label"].tolist(),
        labels.tolist(),
        proba,
        timestamps=dataset.test["feature_ts"],
        time_buckets=time_buckets,
    )


def _log_to_mlflow(
    model: TrainedModel,
    dataset: Dataset,
    config: TrainingConfig,
    candidate_metrics: Metrics,
    baseline_metrics: dict[str, Metrics],
    decision: PromotionResult,
    git_sha: str | None,
    *,
    model_name: str,
) -> registry.LoggedModel:
    with registry.start_run():
        params = _flatten_config(config)
        params["feature_set_version"] = FEATURE_SET_VERSION
        params["theta"] = dataset.theta
        params["symbols"] = ",".join(dataset.symbols)
        if git_sha:
            params["git_sha"] = git_sha
        registry.log_params(params)

        metrics_to_log = {
            f"candidate_{k}": v for k, v in _scalar_metrics(candidate_metrics).items()
        }
        for name, metrics in baseline_metrics.items():
            metrics_to_log.update(
                {f"baseline_{name}_{k}": v for k, v in _scalar_metrics(metrics).items()}
            )
        registry.log_metrics(metrics_to_log)

        registry.log_dict_artifact(candidate_metrics.confusion_matrix, "confusion_matrix.json")
        gains = model.booster.feature_importance(importance_type="gain").tolist()
        importances = dict(zip(model.feature_names, gains, strict=True))
        registry.log_dict_artifact(importances, "feature_importances.json")
        registry.log_dict_artifact(
            {"promoted": decision.promote, "rejection_reason": decision.rejection_reason},
            "promotion_decision.json",
        )

        input_example = dataset.train[list(FEATURE_NAMES)].head(5)
        predictions_example = predict_proba(model, dataset.train.head(5))
        return registry.log_model(
            model.booster,
            input_example=input_example,
            predictions_example=predictions_example,
            registered_model_name=model_name,
        )


def _scalar_metrics(metrics: Metrics) -> dict[str, float]:
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "logloss": metrics.logloss,
        "brier": metrics.brier,
    }
