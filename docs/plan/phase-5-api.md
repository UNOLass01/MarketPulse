# Phase 5 — Serving

**Objective:** predictions available over HTTP from the promoted model.
**Complexity:** Medium · **Effort:** ~1.5 days · **Depends on:** Phase 3 (a Production model must exist)
**Exit criterion:** promoting a new model in MLflow changes `model_version` in API responses **with no redeploy**.

## Tasks

### App skeleton
- [ ] `services/api/main.py` — app factory, lifespan handler, middleware stack
- [ ] Correlation-ID middleware: accept inbound header or generate, attach to every log line and response
- [ ] Consistent error envelope: `error_code` (machine-readable), `message`, `correlation_id`, `timestamp`
- [ ] Versioned path prefix `/api/v1` from day one
- [ ] Response models from `contracts/api.py` so OpenAPI docs are generated, never hand-written

### Model cache
- [ ] Load Production model **once at startup**, cache model + version + signature in memory (never per request — that couples API availability to MLflow's)
- [ ] Background refresh task polling the registry; on version change, load into a second slot and **swap the reference atomically**
- [ ] On failed refresh: keep the old model, alert. Never fail open into an unmodelled state.
- [ ] Missing Production model → start anyway, `/ready` returns 503 with reason. Do not crash-loop.

### Endpoints
- [ ] `GET /api/v1/predictions/{symbol}` — label, all 3 class probabilities, `model_version`, `feature_ts`, **`feature_age_seconds`**
- [ ] `GET /api/v1/predictions` — all active symbols, batched in one pass
- [ ] `GET /api/v1/predictions/{symbol}/history` — paginated, time-bounded, filterable by model version
- [ ] `GET /health` — liveness only, no dependency checks
- [ ] `GET /ready` — DB reachable + model loaded + features fresh
- [ ] `GET /health/dependencies` — per-dependency detail
- [ ] `GET /api/v1/model/current`, `/model/versions`, `POST /api/v1/model/refresh`
- [ ] `GET /api/v1/monitoring/{drift,performance,quality,pipeline}` (stubs until Phase 6)
- [ ] `GET /api/v1/symbols`, `/features/{symbol}/latest`, `/ticks/{symbol}`
- [ ] `GET /metrics` — Prometheus format

### Guards (the important part)
- [ ] **Staleness guard:** features older than threshold → 503 with the age in the body. A prediction on 40-minute-old data is worse than no prediction, because the caller can't tell.
- [ ] **Schema guard:** feature vector's `feature_set_version` must match the model's → refuse + alert. Catches a retrain that changed features without a coordinated deploy.
- [ ] Feature ordering read from `features/registry.py`, never incidental column order
- [ ] Bounded pagination with an enforced max page size

### Prediction logging
- [ ] `predictions` table: `model_version`, `feature_ts`, `predicted_at`, label, probabilities, `latency_ms`
- [ ] Written asynchronously; **a logging failure must still serve the prediction**. Observability never sits on the critical path.

## Tests
- [ ] Stale features → 503, and the response body states the age
- [ ] Schema version mismatch → refusal, not a silent prediction
- [ ] No Production model at startup → app starts, `/ready` 503, `/health` 200
- [ ] Liveness and readiness genuinely differ (kill DB: `/health` still 200, `/ready` 503)
- [ ] Model refresh swaps version without dropping in-flight requests
- [ ] Failed refresh retains the previous model
- [ ] Prediction logging failure does not fail the request
- [ ] Feature ordering matches the registry even if the DB returns columns reordered
- [ ] Pagination cannot be forced past the max limit
- [ ] Error envelope shape is identical across all error paths

## Watch out for
- Conflating `/health` and `/ready` means either restarting a healthy process because the DB blinked, or routing traffic to a process that can't serve.
