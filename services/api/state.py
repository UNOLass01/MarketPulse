"""Application state and the FastAPI dependencies that expose it.

Everything the app needs at runtime is built once in :func:`build_state` and
hung off ``app.state``, rather than reached for through module-level globals.
That is what lets a test construct an app with a stub model loader and an
in-memory session factory without monkeypatching anything.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import Depends, Request
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import Settings
from marketpulse.logging import get_correlation_id, get_logger
from marketpulse.ml.predict import ModelCache, Prediction
from marketpulse.monitoring.metrics import MetricsRegistry
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.predictions import upsert_prediction

logger = get_logger(__name__)


@dataclass
class AppState:
    settings: Settings
    session_factory: sessionmaker[Session]
    model_cache: ModelCache
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    #: Held only so shutdown can dispose the pool. Nothing reads through it
    #: directly -- queries go via ``session_factory``.
    engine: Engine | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def max_feature_age(self) -> timedelta:
        return timedelta(seconds=self.settings.serving.max_feature_age_seconds)


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


def get_session(state: AppState = Depends(get_state)) -> Iterator[Session]:
    """Request-scoped session, committed on success and rolled back on error.

    Read routes don't need the commit, but write paths (prediction logging)
    do, and having one session dependency means no route can accidentally
    leak a connection by opening its own.
    """
    with session_scope(state.session_factory) as session:
        yield session


def log_prediction(state: AppState, prediction: Prediction) -> None:
    """Persist one served prediction, swallowing any failure.

    Observability never sits on the critical path (phase-5 plan). This runs
    as a background task *after* the response has been produced, and a
    database problem here is logged and dropped — the caller already has
    their prediction, and failing the request to protect a log row would
    trade the product for the telemetry.
    """
    correlation_id = get_correlation_id()
    try:
        with session_scope(state.session_factory) as session:
            upsert_prediction(
                session,
                symbol=prediction.symbol,
                model_version=prediction.model_version,
                feature_set_version=prediction.feature_set_version,
                feature_ts=prediction.feature_ts,
                predicted_at=prediction.predicted_at,
                label=prediction.label,
                probabilities=prediction.probabilities,
                latency_ms=prediction.latency_ms,
                correlation_id=_as_uuid(correlation_id),
            )
    except Exception as exc:  # noqa: BLE001 - logging must never fail a request
        state.metrics.increment(
            "marketpulse_api_prediction_log_failures_total",
            help_text="Predictions served but not persisted.",
        )
        logger.error(
            "failed to persist prediction; the response was already served",
            extra={
                "extra_fields": {
                    "symbol": prediction.symbol,
                    "model_version": prediction.model_version,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            },
        )


def _as_uuid(value: str | None) -> UUID | None:
    """Correlation ids are free-form strings (an upstream caller may send
    anything), while the column is a UUID. A non-UUID id is stored as null
    rather than rejected -- the id still appears in the logs, which is where
    it is actually useful.
    """
    if value is None:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None
