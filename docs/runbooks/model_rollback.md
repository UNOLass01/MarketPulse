# Runbook — model rollback

**Fired by:** `feature_drift` (correlated multi-feature drift against the
Production model's training reference) and `model_accuracy_degraded`
(realised accuracy below the floor, sliced by model version).

**What it means:** the model currently in Production may no longer be the
right one to serve. *May.* Rolling back is the containment action, not the
first action.

## Rule out the boring cause first

Drift far more often means a broken pipeline than a moved market. Before
touching the registry:

1. Read [`consumer_lag.md`](./consumer_lag.md) and confirm features are
   fresh and complete. Drift computed over a half-empty window is drift in
   the window, not in the world.
2. Check the Phase 4 quality checks:
   ```bash
   curl -s localhost:8000/api/v1/monitoring/quality | jq '.all_passed, .checks[] | select(.passed==false)'
   ```
3. Check *which* features drifted:
   ```bash
   curl -s localhost:8000/api/v1/monitoring/drift | jq '.metrics[] | select(.severity!="stable")'
   ```
   A handful of related features (all the moving averages, say) moving
   together is usually a data problem. Broad drift across unrelated feature
   families is more likely a genuine regime change.

## Confirm the model is actually the problem

```bash
curl -s localhost:8000/api/v1/monitoring/performance | jq '.slices[], .pending_count'
```

Read this carefully:

- `pending_count` high and `resolved_count` low → not enough has resolved yet.
  Predictions inside the horizon are **unknown, not wrong**. Wait.
- Accuracy low on the *current* version but healthy on the previous one, with
  a clean promotion boundary between them → that is the case for rollback.
- Accuracy low on *both* versions → the problem is upstream of the model.
  Rolling back will not help.

Compare against the baselines recorded for the run in MLflow
(<http://localhost:5000>) — a model beating majority/persistence but below the
absolute floor is a different situation from one that has fallen below its own
baselines.

## Rolling back

There is no auto-rollback and no auto-promotion (CLAUDE.md rule #8). This is
deliberate and manual.

1. Find the last known-good version:
   ```bash
   curl -s localhost:8000/api/v1/model/versions | jq
   ```
2. In the MLflow UI, transition that version back to **Production** and the
   current one to **Archived**.
3. Force the API to pick it up without a redeploy:
   ```bash
   curl -sX POST localhost:8000/api/v1/model/refresh | jq
   curl -s localhost:8000/api/v1/model/current | jq '.model_version'
   ```
   The background refresh would do this within
   `MP_SERVING__MODEL_REFRESH_SECONDS` anyway; the POST just removes the wait.

4. Confirm `predictions` rows start carrying the rolled-back `model_version`.
   Performance slices are keyed by version, so the recovery shows up as its
   own slice rather than being averaged into the bad one.

## Afterwards

Do **not** immediately retrain to "fix" it. If the cause was a data problem,
retraining bakes the bad data into the next model. Fix the pipeline, let clean
data accumulate for at least one full feature lookback (24h), then let
`dag_model_retraining` run on its own schedule — its promotion gate will
refuse a candidate that does not beat both the baselines and the incumbent.
