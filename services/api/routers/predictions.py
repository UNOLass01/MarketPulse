"""Prediction routes.

Every guard, every probability, and the whole feature-ordering discipline
live in ``marketpulse.ml.predict``. What is left here is: read stored rows,
hand them over, shape the result. If a moving average ever appears in this
file, something has gone badly wrong (CLAUDE.md rule #5).
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from marketpulse.contracts.api import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionListResponse,
    PredictionResponse,
)
from marketpulse.ml.predict import FeatureSnapshot, Prediction, predict_batch
from marketpulse.storage.models import Feature
from marketpulse.storage.repositories.features import (
    latest_feature_vector,
    latest_feature_vector_per_symbol,
)
from marketpulse.storage.repositories.predictions import list_predictions
from marketpulse.storage.repositories.symbols import list_symbols
from services.api.errors import HTTP_422_UNPROCESSABLE_CONTENT
from services.api.state import AppState, get_session, get_state, log_prediction

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _to_snapshot(symbol: str, row: Feature) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol=symbol,
        feature_ts=row.feature_ts,
        feature_set_version=row.feature_set_version,
        feature_values=dict(row.feature_values),
        insufficient_history=row.insufficient_history,
        has_gap=row.has_gap,
    )


def _to_response(prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(
        symbol=prediction.symbol,
        label=prediction.label,  # type: ignore[arg-type]
        probabilities=prediction.probabilities,
        model_version=prediction.model_version,
        feature_set_version=prediction.feature_set_version,
        feature_ts=prediction.feature_ts,
        feature_age_seconds=prediction.feature_age_seconds,
        predicted_at=prediction.predicted_at,
    )


@router.get("", response_model=PredictionListResponse)
def predict_all(
    background: BackgroundTasks,
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
) -> PredictionListResponse:
    """Every active symbol, batched into a single model call.

    One query for the latest row per symbol and one ``predict`` over the
    whole frame â€” not a loop of single-symbol predictions, which would make
    response time scale with the symbol count.
    """
    loaded = state.model_cache.require()
    rows = latest_feature_vector_per_symbol(session, loaded.feature_set_version)
    snapshots = [_to_snapshot(code, row) for code, row in rows]

    predictions = predict_batch(
        loaded,
        snapshots,
        now=datetime.now(UTC),
        max_feature_age=state.max_feature_age,
    )
    for prediction in predictions:
        background.add_task(log_prediction, state, prediction)

    return PredictionListResponse(
        predictions=[_to_response(p) for p in predictions],
        model_version=loaded.version,
        generated_at=datetime.now(UTC),
    )


@router.get("/{symbol}", response_model=PredictionResponse)
def predict_symbol(
    symbol: str,
    background: BackgroundTasks,
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
) -> PredictionResponse:
    loaded = state.model_cache.require()

    symbol_row = next((s for s in list_symbols(session) if s.code == symbol), None)
    if symbol_row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown symbol '{symbol}'")

    row = latest_feature_vector(session, symbol_row.id, loaded.feature_set_version)
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no feature rows for '{symbol}' at feature_set_version="
            f"{loaded.feature_set_version}",
        )

    prediction = predict_batch(
        loaded,
        [_to_snapshot(symbol, row)],
        now=datetime.now(UTC),
        max_feature_age=state.max_feature_age,
    )[0]
    background.add_task(log_prediction, state, prediction)
    return _to_response(prediction)


@router.get("/{symbol}/history", response_model=PredictionHistoryResponse)
def prediction_history(
    symbol: str,
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
    hours: float = Query(default=24.0, gt=0, description="Look-back window in hours."),
    model_version: str | None = Query(default=None),
    limit: int | None = Query(default=None, gt=0),
    offset: int = Query(default=0, ge=0),
) -> PredictionHistoryResponse:
    """Past predictions, read back from storage â€” never recomputed.

    Both bounds are enforced server-side: ``limit`` is clamped to
    ``serving.max_page_size`` and the window to ``serving.max_history_days``.
    A client cannot widen either, which is the difference between pagination
    and a full table scan with extra steps.
    """
    serving = state.settings.serving
    resolved_limit = limit or serving.default_page_size
    if resolved_limit > serving.max_page_size:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_CONTENT,
            f"limit={resolved_limit} exceeds the maximum page size of {serving.max_page_size}",
        )

    max_hours = serving.max_history_days * 24
    if hours > max_hours:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_CONTENT,
            f"hours={hours} exceeds the maximum look-back of {max_hours}",
        )

    end = datetime.now(UTC)
    rows, has_more = list_predictions(
        session,
        symbol=symbol,
        start=end - timedelta(hours=hours),
        end=end,
        model_version=model_version,
        limit=resolved_limit,
        offset=offset,
    )
    return PredictionHistoryResponse(
        symbol=symbol,
        items=[
            PredictionHistoryItem(
                symbol=symbol,
                label=row.label,
                probabilities=row.probabilities,
                model_version=row.model_version,
                feature_set_version=row.feature_set_version,
                feature_ts=row.feature_ts,
                predicted_at=row.predicted_at,
                latency_ms=row.latency_ms,
            )
            for row in rows
        ],
        limit=resolved_limit,
        offset=offset,
        has_more=has_more,
    )
