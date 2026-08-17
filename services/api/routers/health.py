"""Liveness, readiness, and per-dependency detail.

These are three different questions and the plan is explicit that conflating
them is the trap:

* ``/health`` — is this process alive? Answered from nothing. A healthy
  process whose database blinked must not be restarted by an orchestrator.
* ``/ready`` — can it serve a prediction right now? Answered by actually
  checking. Returning 503 sheds traffic without killing anything.
* ``/health/dependencies`` — which dependency is the problem? For a human,
  during an incident.
"""

import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from marketpulse.contracts.api import (
    DependenciesResponse,
    DependencyStatus,
    HealthResponse,
    ReadinessResponse,
)
from marketpulse.ml.predict import feature_age_seconds
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.features import latest_feature_ts_per_symbol
from services.api.state import AppState, get_state

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness. Touches no dependency, on purpose — see module docstring."""
    return HealthResponse()


def _check_database(state: AppState) -> DependencyStatus:
    started = time.perf_counter()
    try:
        with session_scope(state.session_factory) as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
        return DependencyStatus(
            name="postgres", healthy=False, detail=f"{type(exc).__name__}: {exc}"
        )
    return DependencyStatus(
        name="postgres", healthy=True, latency_ms=(time.perf_counter() - started) * 1000.0
    )


def _check_model(state: AppState) -> DependencyStatus:
    loaded = state.model_cache.current
    if loaded is None:
        return DependencyStatus(
            name="model",
            healthy=False,
            detail=state.model_cache.last_refresh_error or "no Production model loaded",
        )
    return DependencyStatus(name="model", healthy=True, detail=f"version={loaded.version}")


def _check_feature_freshness(state: AppState) -> DependencyStatus:
    """Freshest feature row across all symbols, against the staleness guard.

    Uses the *newest* symbol rather than the oldest: readiness asks whether
    this process can serve anything at all. A single symbol that stopped
    updating is a data-quality alert (Phase 6's job), not a reason to pull
    the whole API out of the load balancer.
    """
    max_age = timedelta(seconds=state.settings.serving.max_feature_age_seconds)
    try:
        with session_scope(state.session_factory) as session:
            latest = latest_feature_ts_per_symbol(session)
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="features", healthy=False, detail=f"{type(exc).__name__}: {exc}"
        )

    if not latest:
        return DependencyStatus(name="features", healthy=False, detail="no feature rows exist")

    now = datetime.now(UTC)
    freshest = min(feature_age_seconds(ts, now=now) for ts in latest.values())
    healthy = freshest <= max_age.total_seconds()
    return DependencyStatus(
        name="features",
        healthy=healthy,
        detail=(
            f"freshest feature row is {freshest:.1f}s old "
            f"(limit {max_age.total_seconds():.0f}s)"
        ),
    )


def _dependencies(state: AppState) -> list[DependencyStatus]:
    return [_check_database(state), _check_model(state), _check_feature_freshness(state)]


@router.get("/ready", response_model=ReadinessResponse)
def ready(response: Response, state: AppState = Depends(get_state)) -> ReadinessResponse:
    """503 with a stated reason when anything required is missing.

    Notably this returns *a body*, not a bare status: "not ready" without
    saying which dependency is down turns every deploy question into a log
    dive.
    """
    dependencies = _dependencies(state)
    unhealthy = [d for d in dependencies if not d.healthy]
    if unhealthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        ready=not unhealthy,
        reason="; ".join(f"{d.name}: {d.detail}" for d in unhealthy) or None,
        dependencies=dependencies,
    )


@router.get("/health/dependencies", response_model=DependenciesResponse)
def dependencies(state: AppState = Depends(get_state)) -> DependenciesResponse:
    """Always 200 — this route reports on dependencies rather than gating on
    them, so a monitoring scrape of it never fails just because Postgres is
    down. That fact is in the body.
    """
    return DependenciesResponse(dependencies=_dependencies(state), checked_at=datetime.now(UTC))
