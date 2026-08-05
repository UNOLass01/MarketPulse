"""``ml.pipeline.run_training_pipeline`` end-to-end: real Postgres (raw ticks
+ features written through the same online feature pipeline the streaming
consumer uses) and a real (sqlite-backed, local) MLflow tracking server.

Promotion outcome depends on the trained model's actual quality, so most
assertions here are conditioned on ``result.promoted`` rather than assuming
a specific outcome -- a rejected run is just as valid a pipeline result as a
promoted one (CLAUDE.md). The archive-on-reprovision test forces a
deterministic outcome by patching ``evaluate.decide_promotion`` -- the one
place this file mocks something -- because that mechanic must be verified
regardless of what a real model happens to score.
"""

import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import (
    DatabaseSettings,
    FeaturesSettings,
    MLflowSettings,
    RabbitMQSettings,
    Settings,
)
from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.features.pipeline import compute_feature_vector
from marketpulse.features.windows import Observation, WindowStore
from marketpulse.ml import evaluate
from marketpulse.ml.config import (
    EvaluationConfig,
    LabelingConfig,
    LightGBMConfig,
    TrainingConfig,
)
from marketpulse.ml.config import SplitConfig as MLSplitConfig
from marketpulse.ml.evaluate import PromotionResult
from marketpulse.ml.pipeline import run_training_pipeline
from marketpulse.ml.registry import STAGE_PRODUCTION, STAGE_STAGING
from marketpulse.storage.engine import session_scope
from marketpulse.storage.partitions import ensure_partitions_covering
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.ml_registry import get_model_version, get_training_run
from marketpulse.storage.repositories.ticks import upsert_tick

pytestmark = pytest.mark.integration

STEP = timedelta(minutes=5)
HORIZON_MINUTES = 15.0
# Must exceed the actual tick spacing (STEP=5min=300s) or every consecutive
# pair would trip has_gap; also doubles as assemble_dataset's forward-price
# join tolerance (both read from the same FeaturesSettings.gap_threshold_seconds).
GAP_THRESHOLD_SECONDS = 600.0
WARMUP_TICKS = 300  # > 24h of 5-minute ticks, so realised_vol_24h can be sufficient
USABLE_TICKS = 500
# Short enough that every split (including the ~15% validation/test slices)
# spans several full cycles -- long enough relative to the 3-tick horizon to
# stay genuinely learnable. A period close to the split size risks a split
# landing entirely inside one monotonic half-cycle, starving it of one class.
SINE_PERIOD_TICKS = 60


def _price(i: int) -> float:
    return 100.0 + 10.0 * math.sin(2 * math.pi * i / SINE_PERIOD_TICKS)


def _seed_trending_symbol(session: Session, symbol: str, n: int, *, start: datetime) -> None:
    window = WindowStore()
    gap_threshold = timedelta(seconds=GAP_THRESHOLD_SECONDS)

    for i in range(n):
        observed_at = start + i * STEP
        price = _price(i)
        envelope = TickEnvelope(
            emitted_at=observed_at,
            symbol=symbol,
            payload=TickPayload(price=price, volume=10.0, provider_observed_at=observed_at),
        )
        upsert_tick(session, envelope)
        window.push(symbol, Observation(observed_at=observed_at, price=price, volume=10.0))
        vector = compute_feature_vector(
            symbol, window.snapshot(symbol), as_of=observed_at, gap_threshold=gap_threshold
        )
        upsert_feature_vector(session, vector)


def _fast_training_config(**split_overrides: object) -> TrainingConfig:
    split_kwargs: dict[str, object] = {
        "train_frac": 0.7,
        "val_frac": 0.15,
        "min_rows_per_split": 40,
        "min_class_frac": 0.02,
    }
    split_kwargs.update(split_overrides)
    return TrainingConfig(
        config_version=1,
        labeling=LabelingConfig(horizon_minutes=HORIZON_MINUTES, theta_quantile=0.4, theta=None),
        split=MLSplitConfig(**split_kwargs),  # type: ignore[arg-type]
        lightgbm=LightGBMConfig(
            objective="multiclass",
            learning_rate=0.2,
            num_leaves=15,
            max_depth=-1,
            min_data_in_leaf=5,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            bagging_freq=0,
            lambda_l1=0.0,
            lambda_l2=0.0,
            num_boost_round=100,
            early_stopping_rounds=20,
            seed=42,
            verbosity=-1,
        ),
        evaluation=EvaluationConfig(random_baseline_seed=42, time_buckets=3),
    )


@pytest.fixture
def pipeline_settings(
    db_settings: DatabaseSettings, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    monkeypatch.chdir(tmp_path)  # keep any default mlflow artifact root inside tmp_path
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"  # type: ignore[operator]
    return Settings(
        env="test",
        db=db_settings,
        rabbitmq=RabbitMQSettings(user="x", password="x"),
        features=FeaturesSettings(gap_threshold_seconds=GAP_THRESHOLD_SECONDS),
        mlflow=MLflowSettings(
            tracking_uri=tracking_uri,
            registry_model_name="marketpulse-pipeline-test",
            experiment_name="pipeline-test",
        ),
    )


def _seed(
    session_factory: sessionmaker[Session], engine: Engine, *, symbol: str = "BTC-USD"
) -> None:
    # Anchored to "now" so this doesn't rely on a hardcoded date matching
    # whatever month the suite happens to run in. The ~2.8-day span can still
    # cross a month boundary near the 1st, so partitions covering `start`'s
    # month (+1 ahead) are ensured explicitly rather than assumed from the
    # `engine` fixture's own (today-anchored) coverage.
    start = datetime.now(UTC) - timedelta(days=5)
    with engine.begin() as connection:
        ensure_partitions_covering(connection, "raw_ticks", start.date(), months_ahead=1)
        ensure_partitions_covering(connection, "features", start.date(), months_ahead=1)
    with session_scope(session_factory) as session:
        _seed_trending_symbol(session, symbol, WARMUP_TICKS + USABLE_TICKS, start=start)


def test_pipeline_runs_end_to_end_and_persists_consistent_state(
    session_factory: sessionmaker[Session], pipeline_settings: Settings, engine: Engine
) -> None:
    _seed(session_factory, engine)
    config = _fast_training_config()

    result = run_training_pipeline(session_factory, pipeline_settings, training_config=config)

    assert result.training_run_id > 0
    assert result.model_version_id > 0
    assert result.mlflow_run_id
    assert result.mlflow_model_version == "1"
    assert 0.0 <= result.candidate_metrics.macro_f1 <= 1.0

    with session_factory() as session:
        run = get_training_run(session, result.training_run_id)
        version = get_model_version(session, result.model_version_id)

    assert run is not None
    assert run.promoted == result.promoted
    assert run.mlflow_run_id == result.mlflow_run_id
    assert run.feature_set_version == 1
    assert run.train_row_count > 0

    assert version is not None
    if result.promoted:
        assert version.stage == STAGE_PRODUCTION
        assert version.promoted_at is not None
        assert version.reference_feature_stats is not None
        assert run.rejection_reason is None
    else:
        assert version.stage == STAGE_STAGING
        assert version.promoted_at is None
        assert version.reference_feature_stats is None
        assert run.rejection_reason is not None


def test_rejected_run_is_recorded_and_kept_in_staging(
    session_factory: sessionmaker[Session],
    pipeline_settings: Settings,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(session_factory, engine)
    forced_rejection = PromotionResult(promote=False, rejection_reasons=("forced for test",))
    monkeypatch.setattr(evaluate, "decide_promotion", lambda *a, **k: forced_rejection)

    result = run_training_pipeline(
        session_factory, pipeline_settings, training_config=_fast_training_config()
    )

    assert result.promoted is False
    assert result.rejection_reason == "forced for test"

    with session_factory() as session:
        version = get_model_version(session, result.model_version_id)
    assert version is not None
    assert version.stage == STAGE_STAGING


def test_second_promoted_run_archives_the_previous_production_version(
    session_factory: sessionmaker[Session],
    pipeline_settings: Settings,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(session_factory, engine)
    monkeypatch.setattr(evaluate, "decide_promotion", lambda *a, **k: PromotionResult(promote=True))
    config = _fast_training_config()

    first = run_training_pipeline(session_factory, pipeline_settings, training_config=config)
    assert first.promoted is True
    assert first.mlflow_model_version == "1"

    second = run_training_pipeline(session_factory, pipeline_settings, training_config=config)
    assert second.promoted is True
    assert second.mlflow_model_version == "2"
    assert second.training_run_id != first.training_run_id  # a new row, never mutated in place

    with session_factory() as session:
        first_version = get_model_version(session, first.model_version_id)
        second_version = get_model_version(session, second.model_version_id)
        second_run = get_training_run(session, second.training_run_id)

    assert first_version is not None
    assert first_version.stage == "Archived"
    assert first_version.archived_at is not None

    assert second_version is not None
    assert second_version.stage == STAGE_PRODUCTION

    # The second run should have evaluated against the first as its incumbent.
    assert second_run is not None
    assert second_run.incumbent_metrics is not None
