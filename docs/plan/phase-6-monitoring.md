# Phase 6 — Monitoring

**Objective:** the system observes itself, including true production accuracy.
**Complexity:** Medium-High · **Effort:** ~2 days · **Depends on:** Phases 4 + 5, plus elapsed time for outcomes to resolve
**Exit criterion:** injecting synthetic drift into the feature stream produces a visible PSI breach and an alert.

> **Budget note:** this phase overruns. The `H`-lag makes every debug cycle slow. If time is short, cut scope *here* (fewer drift metrics, simpler dashboard) — never in Phase 7. Documentation is not the buffer.

## Tasks

### Drift
- [ ] `monitoring/drift.py` — PSI, two-sample KS, chi-square for categoricals
- [ ] Compare live window against the **reference snapshot tied to the Production model** (not against yesterday's live data — that measures change, not drift-from-training)
- [ ] Severity: PSI <0.1 stable, 0.1–0.25 moderate, >0.25 significant (conventional thresholds — document them as convention, not law)
- [ ] `drift_metrics` table: long not wide, one row per (feature, window), so adding a feature needs no schema change
- [ ] `dag_drift_monitoring` every 6h
- [ ] **Alert on correlated multi-feature drift, not single-feature.** Single-feature drift is usually noise; alerting on it produces fatigue that gets real signals ignored.

### Performance attribution — the part most portfolio projects skip
- [ ] `monitoring/performance.py` — join predictions older than `H` to realised prices
- [ ] `prediction_outcomes` table (separate from `predictions` because it's written later, by a different process)
- [ ] Rolling accuracy / macro-F1 **sliced by model version**
- [ ] `dag_performance_attribution` hourly, lagged by `H`
- [ ] Prediction class-distribution shift vs training prior — this fires long before accuracy can, and usually means broken features rather than a regime change

### Quality + alerts
- [ ] `monitoring/quality.py` — surface Phase 4 check results
- [ ] `monitoring/alerts.py` — threshold evaluation, dedup/suppression, sustained-breach requirement (N consecutive evaluations, not instantaneous spikes)
- [ ] **Every alert names its runbook.** An alert with no action is just anxiety.
- [ ] Runbooks: `consumer_lag.md`, `model_rollback.md`, `dlq_triage.md`

### Dashboard (Streamlit)
- [ ] Panel 1 System health — service grid, queue depth sparkline, DLQ counter, last-tick-per-symbol, recent errors
- [ ] Panel 2 Data pipeline — ingestion rate, completeness by symbol, quality history, null rates
- [ ] Panel 3 Model performance — rolling accuracy with the **baseline drawn as a horizontal reference line**, confusion matrix, per-class metrics, **promotion events as vertical annotations** (a visible accuracy step at a promotion boundary is the single most persuasive artifact this project can produce)
- [ ] Panel 4 Drift — PSI heatmap (feature × time), KS stats, live-vs-reference overlays, severity timeline
- [ ] **Read-only DB credentials.** The dashboard computes nothing and writes nothing — metrics are precomputed by Airflow so they exist whether or not anyone is looking.
- [ ] Bounded default time ranges + query caching

## Tests
- [ ] PSI on identical distributions ≈ 0; on a known shifted distribution matches a hand-computed value
- [ ] KS statistic matches `scipy` on a reference sample
- [ ] Synthetic drift injection → severity crosses threshold → alert fires
- [ ] Attribution join respects the `H` lag (off-by-one here is the likeliest bug in the phase)
- [ ] Predictions younger than `H` are **not** resolved
- [ ] Accuracy slicing by model version is correct across a promotion boundary
- [ ] Alert suppression: same condition twice → one alert
- [ ] Sustained-breach logic: single spike does not alert, N consecutive does
- [ ] Dashboard renders an explicit empty state with no data (not a crash)

## Watch out for
- Drift often means a broken pipeline, not a real shift. Check Phase 3-layer symptoms against Phase 1 causes before believing the model has degraded.
