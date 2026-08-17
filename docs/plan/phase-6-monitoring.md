# Phase 6 â€” Monitoring

**Objective:** the system observes itself, including true production accuracy.
**Complexity:** Medium-High Â· **Effort:** ~2 days Â· **Depends on:** Phases 4 + 5, plus elapsed time for outcomes to resolve
**Exit criterion:** injecting synthetic drift into the feature stream produces a visible PSI breach and an alert.

> **Budget note:** this phase overruns. The `H`-lag makes every debug cycle slow. If time is short, cut scope *here* (fewer drift metrics, simpler dashboard) â€” never in Phase 7. Documentation is not the buffer.

## Tasks

### Drift
- [x] `monitoring/drift.py` â€” PSI, two-sample KS, chi-square for categoricals
- [x] Compare live window against the **reference snapshot tied to the Production model** (not against yesterday's live data â€” that measures change, not drift-from-training)
- [x] Severity: PSI <0.1 stable, 0.1â€“0.25 moderate, >0.25 significant (conventional thresholds â€” document them as convention, not law)
- [x] `drift_metrics` table: long not wide, one row per (feature, window), so adding a feature needs no schema change
- [x] `dag_drift_monitoring` every 6h
- [x] **Alert on correlated multi-feature drift, not single-feature.** Single-feature drift is usually noise; alerting on it produces fatigue that gets real signals ignored.

### Performance attribution â€” the part most portfolio projects skip
- [x] `monitoring/performance.py` â€” join predictions older than `H` to realised prices
- [x] `prediction_outcomes` table (separate from `predictions` because it's written later, by a different process)
- [x] Rolling accuracy / macro-F1 **sliced by model version**
- [x] `dag_performance_attribution` hourly, lagged by `H`
- [x] Prediction class-distribution shift vs training prior â€” this fires long before accuracy can, and usually means broken features rather than a regime change

### Quality + alerts
- [x] `monitoring/quality.py` â€” surface Phase 4 check results
- [x] `monitoring/alerts.py` â€” threshold evaluation, dedup/suppression, sustained-breach requirement (N consecutive evaluations, not instantaneous spikes)
- [x] **Every alert names its runbook.** An alert with no action is just anxiety.
- [x] Runbooks: `consumer_lag.md`, `model_rollback.md`, `dlq_triage.md`

### Dashboard (Streamlit)
- [x] Panel 1 System health â€” service grid, queue depth sparkline, DLQ counter, last-tick-per-symbol, recent errors
- [x] Panel 2 Data pipeline â€” ingestion rate, completeness by symbol, quality history, null rates
- [x] Panel 3 Model performance â€” rolling accuracy with the **baseline drawn as a horizontal reference line**, confusion matrix, per-class metrics, **promotion events as vertical annotations** (a visible accuracy step at a promotion boundary is the single most persuasive artifact this project can produce)
- [x] Panel 4 Drift â€” PSI heatmap (feature Ã— time), KS stats, live-vs-reference overlays, severity timeline
- [x] **Read-only DB credentials.** The dashboard computes nothing and writes nothing â€” metrics are precomputed by Airflow so they exist whether or not anyone is looking.
- [x] Bounded default time ranges + query caching

## Tests
- [x] PSI on identical distributions â‰ˆ 0; on a known shifted distribution matches a hand-computed value
- [x] KS statistic matches `scipy` on a reference sample
- [x] Synthetic drift injection â†’ severity crosses threshold â†’ alert fires
- [x] Attribution join respects the `H` lag (off-by-one here is the likeliest bug in the phase)
- [x] Predictions younger than `H` are **not** resolved
- [x] Accuracy slicing by model version is correct across a promotion boundary
- [x] Alert suppression: same condition twice â†’ one alert
- [x] Sustained-breach logic: single spike does not alert, N consecutive does
- [x] Dashboard renders an explicit empty state with no data (not a crash)

## Watch out for
- Drift often means a broken pipeline, not a real shift. Check Phase 3-layer symptoms against Phase 1 causes before believing the model has degraded.
