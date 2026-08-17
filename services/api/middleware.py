"""Correlation-ID propagation and per-request metrics.

The correlation id is set into ``marketpulse.logging``'s contextvar before
any handler runs, so every log line emitted during the request — including
ones from deep inside ``marketpulse.*``, which knows nothing about HTTP —
carries it without being threaded through a single function signature.
"""

import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from marketpulse.logging import get_logger, set_correlation_id
from marketpulse.monitoring.metrics import MetricsRegistry

#: Inbound header honoured if present, so a correlation id created by an
#: upstream caller survives the hop instead of being replaced by a fresh one
#: that can't be joined back to their logs.
CORRELATION_HEADER = "X-Correlation-ID"

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid4())
        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        # Echoed on every response, including error responses -- a caller
        # reporting a failure can quote the id straight from the headers.
        response.headers[CORRELATION_HEADER] = correlation_id
        return response


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Counts requests and records latency into the in-process registry.

    Labels use the *route template* (``/api/v1/predictions/{symbol}``), never
    the concrete path — labelling by concrete path would mint a new time
    series per symbol and, on a free-form path, unbounded cardinality.
    """

    def __init__(self, app: object, registry: MetricsRegistry) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._registry = registry

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        labels = {
            "method": request.method,
            "path": path,
            "status": str(response.status_code),
        }
        self._registry.increment(
            "marketpulse_api_requests_total",
            help_text="Total HTTP requests handled by the API.",
            labels=labels,
        )
        self._registry.set_gauge(
            "marketpulse_api_request_latency_ms",
            elapsed_ms,
            help_text="Latency of the most recent request per route.",
            labels={"method": request.method, "path": path},
        )
        return response
