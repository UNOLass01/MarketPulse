"""Model introspection and manual refresh.

``/model/versions`` reads the local ``model_versions`` mirror rather than
querying MLflow. That is deliberate: promotion history is already mirrored
into Postgres (``storage.models.ModelVersion``) precisely so this read does
not couple API availability to the tracking server's, and the local table is
the only place ``promoted_at`` / ``archived_at`` exist at all.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from marketpulse.contracts.api import (
    ModelInfoResponse,
    ModelRefreshResponse,
    ModelVersionItem,
    ModelVersionsResponse,
)
from marketpulse.storage.models import ModelVersion
from services.api.state import AppState, get_session, get_state

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/current", response_model=ModelInfoResponse)
def current_model(state: AppState = Depends(get_state)) -> ModelInfoResponse:
    """What is loaded right now.

    Returns 200 with null fields when nothing is loaded rather than 503:
    "which model is serving?" is answerable — the answer is "none" — and a
    monitoring dashboard needs to read that answer without special-casing an
    error status.
    """
    loaded = state.model_cache.current
    if loaded is None:
        return ModelInfoResponse(
            model_version=None,
            feature_set_version=None,
            feature_names=[],
            loaded_at=None,
            last_refresh_error=state.model_cache.last_refresh_error,
        )
    return ModelInfoResponse(
        model_version=loaded.version,
        feature_set_version=loaded.feature_set_version,
        feature_names=list(loaded.feature_names),
        loaded_at=loaded.loaded_at,
        last_refresh_error=state.model_cache.last_refresh_error,
    )


@router.get("/versions", response_model=ModelVersionsResponse)
def model_versions(
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
) -> ModelVersionsResponse:
    model_name = state.settings.mlflow.registry_model_name
    stmt = (
        select(ModelVersion)
        .where(ModelVersion.mlflow_model_name == model_name)
        .order_by(ModelVersion.created_at.desc())
    )
    return ModelVersionsResponse(
        model_name=model_name,
        versions=[
            ModelVersionItem(
                mlflow_model_version=row.mlflow_model_version,
                stage=row.stage,
                promoted_at=row.promoted_at,
                archived_at=row.archived_at,
                created_at=row.created_at,
            )
            for row in session.execute(stmt).scalars()
        ],
    )


@router.post("/refresh", response_model=ModelRefreshResponse)
def refresh_model(state: AppState = Depends(get_state)) -> ModelRefreshResponse:
    """Force an immediate registry re-check.

    The background task already does this on an interval; this endpoint just
    collapses the wait after a deliberate promotion. It returns 200 even when
    the refresh failed — the failure is reported in ``error`` and the
    previously-loaded model is still serving, which is a successful outcome
    for the *system*, however unwelcome.
    """
    result = state.model_cache.refresh()
    return ModelRefreshResponse(
        changed=result.changed,
        previous_version=result.previous_version,
        current_version=result.current_version,
        error=result.error,
    )
