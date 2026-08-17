"""API entrypoint: wiring only.

Every route delegates to ``marketpulse.*``. Deleting this directory would
leave the guards, the inference, the model cache, and the metric rendering
fully intact and fully tested — which is the check CLAUDE.md's layout rule
is really asking for.

Two startup behaviours are worth calling out because they are the difference
between an API that survives a bad morning and one that does not:

* **The model loads once, at startup, into a cache** — never per request.
  Loading per request would make every prediction's latency include an
  MLflow round-trip and, worse, couple this service's availability to the
  tracking server's.
* **A missing Production model is not fatal.** The app starts, ``/health``
  returns 200, and ``/ready`` returns 503 with the reason. Crash-looping on
  an empty registry would take the whole service down over a state that
  resolves itself the moment something is promoted.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import APIRouter, FastAPI, Response

from marketpulse.config import Settings, get_settings
from marketpulse.logging import get_logger
from marketpulse.ml import registry
from marketpulse.ml.predict import LoadedModel, ModelCache
from marketpulse.storage.engine import make_engine, make_session_factory
from services.api.errors import ERROR_RESPONSES, register_exception_handlers
from services.api.middleware import CorrelationIdMiddleware, RequestMetricsMiddleware
from services.api.routers import data, health, model, monitoring, predictions
from services.api.state import AppState, get_state

logger = get_logger(__name__)

API_PREFIX = "/api/v1"


def _mlflow_loader(settings: Settings) -> Callable[[], LoadedModel | None]:
    """Build the callable ``ModelCache`` polls.

    The MLflow client is created once and closed over, not rebuilt per
    refresh: reconnecting on every poll would turn a routine 60-second tick
    into a recurring source of connection churn against the tracking server.
    """
    client = registry.configure(settings.mlflow)

    def load() -> LoadedModel | None:
        return registry.load_production_bundle(client, settings.mlflow.registry_model_name)

    return load


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: AppState = app.state.app_state

    # A failure here is survivable by design: the app must come up so
    # /ready can *report* the problem rather than the container restarting
    # in a loop nobody can inspect.
    state.model_cache.refresh()
    state.model_cache.start_background_refresh(
        timedelta(seconds=state.settings.serving.model_refresh_seconds)
    )
    logger.info(
        "api started",
        extra={
            "extra_fields": {
                "model_version": (
                    state.model_cache.current.version if state.model_cache.current else None
                ),
                "refresh_seconds": state.settings.serving.model_refresh_seconds,
            }
        },
    )

    yield

    state.model_cache.stop()
    if state.engine is not None:
        state.engine.dispose()


def build_state(settings: Settings | None = None) -> AppState:
    resolved = settings or get_settings()
    engine = make_engine(resolved.db)
    return AppState(
        settings=resolved,
        session_factory=make_session_factory(engine),
        model_cache=ModelCache(_mlflow_loader(resolved)),
        engine=engine,
    )


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the application.

    ``state`` is injectable so tests construct an app with a stub loader and
    a test database instead of monkeypatching module globals — the same
    reason every other seam in this codebase takes its dependencies as
    arguments.
    """
    app_state = state or build_state()

    app = FastAPI(
        title="MarketPulse API",
        version="1.0.0",
        description="Real-time crypto direction predictions from the promoted model.",
        lifespan=lifespan,
    )
    app.state.app_state = app_state

    # Outermost first: the correlation id must be set before the metrics
    # middleware or any handler runs, so every log line in the request --
    # including one emitted while recording metrics -- carries it.
    app.add_middleware(RequestMetricsMiddleware, registry=app_state.metrics)
    app.add_middleware(CorrelationIdMiddleware)

    register_exception_handlers(app)

    # Unversioned: orchestrators and scrapers address these by convention,
    # and moving them under /api/v1 would break every probe config for no
    # benefit. The *business* API is versioned from day one.
    app.include_router(health.router)

    versioned = APIRouter(prefix=API_PREFIX, responses=ERROR_RESPONSES)
    versioned.include_router(predictions.router)
    versioned.include_router(model.router)
    versioned.include_router(monitoring.router)
    versioned.include_router(data.router)
    app.include_router(versioned)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        state_ = app.state.app_state
        loaded = state_.model_cache.current
        state_.metrics.set_gauge(
            "marketpulse_api_model_loaded",
            1.0 if loaded is not None else 0.0,
            help_text="1 when a Production model is loaded, 0 otherwise.",
        )
        return Response(
            content=state_.metrics.render(),
            # The version parameter is part of the format contract; some
            # scrapers fall back to a slower parser without it.
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


# Module-level app for ``uvicorn services.api.main:app``. Built lazily by the
# ASGI server at import time; tests call ``create_app(...)`` directly instead.
def app_factory() -> FastAPI:
    return create_app()


__all__ = ["API_PREFIX", "app_factory", "build_state", "create_app", "get_state"]
