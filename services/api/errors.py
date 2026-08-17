"""One error shape, every failure path.

The phase-5 plan asks for a consistent error envelope; the way to actually
get one is to give the app *no* other way to produce an error body. So every
handler here funnels through :func:`error_response`, and the last one is a
catch-all for ``Exception`` â€” without it, an unhandled bug renders
Starlette's default ``{"detail": ...}`` and the envelope guarantee quietly
becomes "usually".
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from marketpulse.contracts.api import (
    ERROR_FEATURE_SCHEMA_MISMATCH,
    ERROR_FEATURES_STALE,
    ERROR_INTERNAL,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NOT_FOUND,
    ERROR_VALIDATION,
    ErrorEnvelope,
)
from marketpulse.logging import get_correlation_id, get_logger
from marketpulse.ml.predict import (
    FeatureSchemaMismatchError,
    FeaturesStaleError,
    ModelUnavailableError,
)

logger = get_logger(__name__)

#: Starlette renamed ``HTTP_422_UNPROCESSABLE_ENTITY`` to ``..._CONTENT`` and
#: now emits a DeprecationWarning on the old name. The integer is the stable
#: thing: it does not depend on which of the two names a given Starlette
#: release happens to export. Routers import this rather than either name.
HTTP_422_UNPROCESSABLE_CONTENT = 422

#: The error envelope, declared once for OpenAPI. Attached to every router so
#: the generated document tells a client what a failure looks like -- an API
#: whose docs only describe the happy path is half-documented.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "default": {"model": ErrorEnvelope, "description": "Error envelope"}
}

#: HTTP status codes that already carry a machine-readable meaning, mapped to
#: the envelope's ``error_code``. Anything not listed becomes ``ERROR_INTERNAL``.
_STATUS_TO_CODE = {
    status.HTTP_404_NOT_FOUND: ERROR_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_CONTENT: ERROR_VALIDATION,
}


def error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error_code=error_code,
        message=message,
        # Always a string, never null: the middleware guarantees a
        # correlation id exists before any handler runs, so an error a user
        # reports can always be traced back to its log lines.
        correlation_id=get_correlation_id() or "unknown",
        timestamp=datetime.now(UTC),
        details=details or {},
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FeaturesStaleError)
    async def _stale(_: Request, exc: FeaturesStaleError) -> JSONResponse:
        # 503, not 200-with-a-warning: a stale prediction that still looks
        # like a prediction is worse than no prediction, because the caller
        # cannot tell the difference. The age goes in the body so they can.
        logger.warning(
            "refusing prediction on stale features",
            extra={"extra_fields": exc.details()},
        )
        return error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ERROR_FEATURES_STALE,
            message=str(exc),
            details=exc.details(),
        )

    @app.exception_handler(FeatureSchemaMismatchError)
    async def _schema(_: Request, exc: FeatureSchemaMismatchError) -> JSONResponse:
        # 409: the stored features and the loaded model disagree about what
        # the feature set *is*. Logged at error level because it means a
        # retrain shipped without a coordinated deploy, not that a caller
        # did anything wrong.
        logger.error(
            "feature schema mismatch; refusing to predict",
            extra={"extra_fields": exc.details()},
        )
        return error_response(
            status_code=status.HTTP_409_CONFLICT,
            error_code=ERROR_FEATURE_SCHEMA_MISMATCH,
            message=str(exc),
            details=exc.details(),
        )

    @app.exception_handler(ModelUnavailableError)
    async def _no_model(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        return error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ERROR_MODEL_UNAVAILABLE,
            message=str(exc),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            error_code=ERROR_VALIDATION,
            message="request validation failed",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(
            status_code=exc.status_code,
            error_code=_STATUS_TO_CODE.get(exc.status_code, ERROR_INTERNAL),
            message=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # The message is generic on purpose -- an exception string can carry
        # a DSN or a row of data. The correlation id is what connects this
        # response to the full traceback in the logs.
        logger.exception("unhandled exception", exc_info=exc)
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ERROR_INTERNAL,
            message="an internal error occurred",
        )
