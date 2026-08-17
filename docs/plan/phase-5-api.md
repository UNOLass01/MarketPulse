# Phase 5 â€” Serving

**Objective:** predictions available over HTTP from the promoted model.
**Complexity:** Medium Â· **Effort:** ~1.5 days Â· **Depends on:** Phase 3 (a Production model must exist)
**Exit criterion:** promoting a new model in MLflow changes `model_version` in API responses **with no redeploy**.

## Tasks

### App skeleton
- [x] `services/api/main.py` â€” app factory, lifespan handler, middleware stack
- [x] Correlation-ID middleware: accept inbound header or generate, attach to every log line and response
- [x] Consistent error envelope: `error_code` (machine-readable), `message`, `correlation_id`, `timestamp`
- [x] Versioned path prefix `/api/v1` from day one
- [x] Response models from `contracts/api.py` so OpenAPI docs are generated, never hand-written

### Model cache
- [x] Load Production model **once at startup**, cache model + version + signature in memory (never per request â€” that couples API availability to MLflow's)
- [x] Background refresh task polling the registry; on version change, load into a second slot and **swap the reference atomically**
- [x] On failed refresh: keep the old model, alert. Never fail open into an unmodelled state.
- [x] Missing Production model â†’ start anyway, `/ready` returns 503 with reason. Do not crash-loop.

### Endpoints
- [x] `GET /api/v1/predictions/{symbol}` â€” label, all 3 class probabilities, `model_version`, `feature_ts`, **`feature_age_seconds`**
- [x] `GET /api/v1/predictions` â€” all active symbols, batched in one pass
- [x] `GET /api/v1/predictions/{symbol}/history` â€” paginated, time-bounded, filterable by model version
- [x] `GET /health` â€” liveness only, no dependency checks
- [x] `GET /ready` â€” DB reachable + model loaded + features fresh
- [x] `GET /health/dependencies` â€” per-dependency detail
- [x] `GET /api/v1/model/current`, `/model/versions`, `POST /api/v1/model/refresh`
- [x] `GET /api/v1/monitoring/{drift,performance,quality,pipeline}` (stubs until Phase 6)
- [x] `GET /api/v1/symbols`, `/features/{symbol}/latest`, `/ticks/{symbol}`
- [x] `GET /metrics` â€” Prometheus format

### Guards (the important part)
- [x] **Staleness guard:** features older than threshold â†’ 503 with the age in the body. A prediction on 40-minute-old data is worse than no prediction, because the caller can't tell.
- [x] **Schema guard:** feature vector's `feature_set_version` must match the model's â†’ refuse + alert. Catches a retrain that changed features without a coordinated deploy.
- [x] Feature ordering read from `features/registry.py`, never incidental column order
- [x] Bounded pagination with an enforced max page size

### Prediction logging
- [x] `predictions` table: `model_version`, `feature_ts`, `predicted_at`, label, probabilities, `latency_ms`
- [x] Written asynchronously; **a logging failure must still serve the prediction**. Observability never sits on the critical path.

## Tests
- [x] Stale features â†’ 503, and the response body states the age
- [x] Schema version mismatch â†’ refusal, not a silent prediction
- [x] No Production model at startup â†’ app starts, `/ready` 503, `/health` 200
- [x] Liveness and readiness genuinely differ (kill DB: `/health` still 200, `/ready` 503)
- [x] Model refresh swaps version without dropping in-flight requests
- [x] Failed refresh retains the previous model
- [x] Prediction logging failure does not fail the request
- [x] Feature ordering matches the registry even if the DB returns columns reordered
- [x] Pagination cannot be forced past the max limit
- [x] Error envelope shape is identical across all error paths

## Watch out for
- Conflating `/health` and `/ready` means either restarting a healthy process because the DB blinked, or routing traffic to a process that can't serve.
