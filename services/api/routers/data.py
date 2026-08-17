"""Symbol, feature, and raw-tick reads.

The features route serves *stored* rows verbatim, nulls and all. The API
never computes a feature (CLAUDE.md rule #5); if a value is null here, that
is what the feature pipeline wrote, and the ``insufficient_history`` flag
next to it says why.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from marketpulse.contracts.api import (
    FeatureVectorResponse,
    SymbolItem,
    SymbolsResponse,
    TickItem,
    TicksResponse,
)
from marketpulse.features.registry import FEATURE_SET_VERSION
from marketpulse.ml.predict import feature_age_seconds
from marketpulse.storage.repositories.features import (
    latest_feature_ts_per_symbol,
    latest_feature_vector,
)
from marketpulse.storage.repositories.symbols import list_symbols
from marketpulse.storage.repositories.ticks import (
    latest_observed_at_per_symbol,
    list_ticks_page,
)
from services.api.errors import HTTP_422_UNPROCESSABLE_CONTENT
from services.api.state import AppState, get_session, get_state

router = APIRouter(tags=["data"])


@router.get("/symbols", response_model=SymbolsResponse)
def symbols(session: Session = Depends(get_session)) -> SymbolsResponse:
    feature_ts = latest_feature_ts_per_symbol(session)
    tick_at = latest_observed_at_per_symbol(session)
    return SymbolsResponse(
        symbols=[
            SymbolItem(
                symbol=row.code,
                latest_feature_ts=feature_ts.get(row.code),
                latest_tick_at=tick_at.get(row.code),
            )
            for row in list_symbols(session)
        ]
    )


@router.get("/features/{symbol}/latest", response_model=FeatureVectorResponse)
def latest_features(
    symbol: str,
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
) -> FeatureVectorResponse:
    """The newest stored feature row for a symbol.

    Deliberately *not* staleness-guarded: unlike a prediction, a feature row
    carries its own timestamp and age, so a caller inspecting the pipeline
    can see exactly how old the data is. Refusing to show it is what would
    make debugging a stalled producer harder.
    """
    symbol_row = next((s for s in list_symbols(session) if s.code == symbol), None)
    if symbol_row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown symbol '{symbol}'")

    # Prefer the loaded model's feature set so this route shows the caller
    # exactly what a prediction would have consumed; fall back to the
    # registry's current version when no model is loaded, so the route still
    # works on a cold registry.
    loaded = state.model_cache.current
    version = loaded.feature_set_version if loaded is not None else FEATURE_SET_VERSION
    row = latest_feature_vector(session, symbol_row.id, version)
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no feature rows for '{symbol}' at feature_set_version={version}",
        )

    return FeatureVectorResponse(
        symbol=symbol,
        feature_ts=row.feature_ts,
        feature_set_version=row.feature_set_version,
        feature_values=dict(row.feature_values),
        insufficient_history=row.insufficient_history,
        has_gap=row.has_gap,
        feature_age_seconds=feature_age_seconds(row.feature_ts, now=datetime.now(UTC)),
    )


@router.get("/ticks/{symbol}", response_model=TicksResponse)
def ticks(
    symbol: str,
    state: AppState = Depends(get_state),
    session: Session = Depends(get_session),
    hours: float = Query(default=1.0, gt=0),
    limit: int | None = Query(default=None, gt=0),
    offset: int = Query(default=0, ge=0),
) -> TicksResponse:
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
    rows, has_more = list_ticks_page(
        session,
        symbol=symbol,
        start=end - timedelta(hours=hours),
        end=end,
        limit=resolved_limit,
        offset=offset,
    )
    return TicksResponse(
        symbol=symbol,
        ticks=[
            TickItem(
                symbol=symbol,
                observed_at=row.observed_at,
                price=float(row.price),
                volume=float(row.volume),
            )
            for row in rows
        ],
        limit=resolved_limit,
        offset=offset,
        has_more=has_more,
    )
