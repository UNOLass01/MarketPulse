"""HTTP response contracts for the Phase 5 API.

Every route declares one of these as its ``response_model`` so the OpenAPI
document is *generated* from the same types the handlers return, never
hand-written alongside them (phase-5 plan). Nothing here does I/O or
computation — these are shapes, and the mapping from storage rows into them
lives in ``services/api`` wiring or in the ``marketpulse`` modules that
produced the values.

The one rule worth stating explicitly: :class:`ErrorEnvelope` is the *only*
error body this API ever emits. A 404 from a missing symbol, a 503 from the
staleness guard, and a 500 from an unhandled exception all render through it,
so a client can parse failures with one branch instead of four.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Machine-readable ``error_code`` values. Clients branch on these, never on
#: the human-facing ``message``, which is free to change wording.
ERROR_NOT_FOUND = "not_found"
ERROR_VALIDATION = "validation_error"
ERROR_FEATURES_STALE = "features_stale"
ERROR_FEATURE_SCHEMA_MISMATCH = "feature_schema_mismatch"
ERROR_MODEL_UNAVAILABLE = "model_unavailable"
ERROR_NOT_READY = "not_ready"
ERROR_INTERNAL = "internal_error"


class ErrorEnvelope(BaseModel):
    """The single error body shape for every failing route."""

    model_config = ConfigDict(frozen=True)

    error_code: str
    message: str
    correlation_id: str
    timestamp: datetime
    #: Structured context the code alone can't carry — e.g. the actual
    #: feature age on a staleness refusal, which is the whole point of
    #: refusing with a body rather than a bare 503.
    details: dict[str, object] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """One symbol's prediction from the currently-loaded Production model."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    label: Literal["DOWN", "STABLE", "UP"]
    #: All three class probabilities, keyed by label — never just the
    #: argmax. A caller that wants to threshold on confidence needs the
    #: whole distribution, and a near-tie is information.
    probabilities: dict[str, float]
    model_version: str
    feature_set_version: int
    feature_ts: datetime
    #: How old the underlying feature row was when this was served. Present
    #: on every successful response, not only on the staleness refusal, so a
    #: caller can see the guard approaching instead of only its trip.
    feature_age_seconds: float
    predicted_at: datetime


class PredictionListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    predictions: list[PredictionResponse]
    model_version: str
    generated_at: datetime


class PredictionHistoryItem(BaseModel):
    """A previously-served prediction, read back from the ``predictions``
    table — not recomputed.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    label: str
    probabilities: dict[str, float]
    model_version: str
    feature_set_version: int
    feature_ts: datetime
    predicted_at: datetime
    latency_ms: float


class PredictionHistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    items: list[PredictionHistoryItem]
    limit: int
    offset: int
    #: True when more rows exist past this page. Derived by fetching
    #: ``limit + 1`` rows and discarding the extra, which avoids a second
    #: COUNT(*) over a table that only grows.
    has_more: bool


class HealthResponse(BaseModel):
    """Liveness only. Deliberately answers without touching any dependency:
    a healthy process whose database blinked must not be restarted.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    service: str = "marketpulse-api"


class DependencyStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    healthy: bool
    detail: str | None = None
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    """Readiness: can this process actually serve a prediction right now?

    Distinct from liveness on purpose — this one *does* check dependencies,
    and returning 503 here sheds traffic without killing the process.
    """

    model_config = ConfigDict(frozen=True)

    ready: bool
    reason: str | None = None
    dependencies: list[DependencyStatus]


class DependenciesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    dependencies: list[DependencyStatus]
    checked_at: datetime


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version: str | None
    feature_set_version: int | None
    feature_names: list[str]
    loaded_at: datetime | None
    #: Populated when the last background refresh failed. The old model is
    #: still being served — this is how a caller finds out.
    last_refresh_error: str | None = None


class ModelVersionItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    mlflow_model_version: str
    stage: str
    promoted_at: datetime | None
    archived_at: datetime | None
    created_at: datetime


class ModelVersionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    versions: list[ModelVersionItem]


class ModelRefreshResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    changed: bool
    previous_version: str | None
    current_version: str | None
    error: str | None = None


class SymbolItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    latest_feature_ts: datetime | None
    latest_tick_at: datetime | None


class SymbolsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbols: list[SymbolItem]


class FeatureVectorResponse(BaseModel):
    """A stored feature row, served verbatim. The API never computes a
    feature (CLAUDE.md rule #5) — ``feature_values`` is whatever the feature
    pipeline persisted, nulls and all.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    feature_ts: datetime
    feature_set_version: int
    feature_values: dict[str, float | None]
    insufficient_history: bool
    has_gap: bool
    feature_age_seconds: float


class TickItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    observed_at: datetime
    price: float
    volume: float


class TicksResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    ticks: list[TickItem]
    limit: int
    offset: int
    has_more: bool


class DriftMetricItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_name: str
    metric_name: str
    metric_value: float
    p_value: float | None
    severity: str
    reference_model_version: str
    window_start: datetime
    window_end: datetime
    computed_at: datetime


class DriftResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: list[DriftMetricItem]
    #: Highest severity present in ``metrics``; ``"stable"`` when the list
    #: is empty *because nothing breached*. An empty list from "drift has
    #: never run" is reported through ``computed_at is None`` instead —
    #: absence of a signal is never rendered as a passing signal.
    worst_severity: str
    computed_at: datetime | None


class PerformanceSliceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version: str
    resolved_count: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    predicted_distribution: dict[str, float]
    window_start: datetime
    window_end: datetime


class PerformanceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    slices: list[PerformanceSliceItem]
    #: Predictions younger than the horizon are deliberately unresolved, not
    #: missing. Surfacing the count keeps "we don't know yet" visually
    #: distinct from "the model got them wrong".
    pending_count: int
    horizon_minutes: float


class QualityCheckItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_name: str
    symbol: str | None
    passed: bool
    details: dict[str, object]
    window_start: datetime
    window_end: datetime
    checked_at: datetime


class QualityResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    checks: list[QualityCheckItem]
    all_passed: bool


class AlertItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert_name: str
    severity: str
    status: str
    #: Never null. An alert with no action attached is just anxiety
    #: (phase-6 plan), so the runbook is part of the alert's identity, not
    #: an optional annotation.
    runbook: str
    details: dict[str, object]
    consecutive_breaches: int
    first_breached_at: datetime
    fired_at: datetime | None
    resolved_at: datetime | None


class PipelineResponse(BaseModel):
    """Pipeline-level state: what has run, what is loaded, what is open."""

    model_config = ConfigDict(frozen=True)

    model_version: str | None
    last_training_run_at: datetime | None
    last_promotion_at: datetime | None
    last_quality_check_at: datetime | None
    last_drift_check_at: datetime | None
    open_alerts: list[AlertItem]
