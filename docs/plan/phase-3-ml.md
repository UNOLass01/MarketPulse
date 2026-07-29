# Phase 3 — ML Pipeline

**Objective:** a trained, tracked, registered model with an honest evaluation.
**Complexity:** Medium · **Effort:** ~2 days · **Depends on:** Phase 2 + seeded history
**Exit criterion:** a model sits in MLflow Production stage, fully reproducible from run ID + git SHA, with metrics compared against all baselines — **including if the outcome was "did not promote"**.

## Tasks

### Labeling
- [ ] `ml/labeling.py` — forward return over horizon `H`, 3-class with deadband `θ`:
      UP if return > θ, DOWN if < −θ, else STABLE
- [ ] `θ` derived from the empirical |return| distribution (~0.4 quantile), not picked arbitrarily
- [ ] `H` and `θ` are logged hyperparameters — changing either changes the problem
- [ ] Training query **excludes the most recent H** of data (those rows aren't labelable yet)

### Dataset assembly
- [ ] `ml/dataset.py` — join features to forward-shifted price, filter on `observed_at`
- [ ] Drop rows flagged `insufficient_history` or `has_gap`
- [ ] Temporal split 70/15/15, chronological, with an **embargo gap of H between segments** (a contiguous split leaks H of validation into training)
- [ ] Sufficiency check: minimum row count + class balance sanity → fail loudly, don't train on a thin sample

### Training
- [ ] `ml/train.py` — LightGBM multiclass, early stopping on validation logloss
- [ ] Class weighting for imbalance
- [ ] **No scaler/normaliser** — LightGBM is scale-invariant; skipping it removes a whole class of "scaler wasn't persisted" bugs
- [ ] Hyperparameters from a versioned config file, never hardcoded

### Evaluation
- [ ] `ml/evaluate.py` — baselines: majority class, persistence, random-stratified, **and the incumbent Production model on the same test window**
- [ ] Metrics: accuracy, **macro**-F1, per-class precision/recall, logloss, Brier, confusion matrix
- [ ] Accuracy stratified by time bucket (catches a model that only works in one regime)
- [ ] **Promotion gate:** beats every naive baseline on macro-F1 AND beats/matches incumbent AND no per-class collapse AND passes calibration sanity. Otherwise → Staging with a recorded `rejection_reason`.

### MLflow
- [ ] Tracking server + Postgres backend + **S3-compatible artifact store** (MinIO locally via `MLFLOW_S3_ENDPOINT_URL`, real S3/R2 in prod — config only, no code change)
- [ ] Log per run: params, `H`, `θ`, feature set version + ordered feature list, window boundaries, row counts, class distribution, all metrics, model artifact **with signature**, feature importances, confusion matrix, git SHA
- [ ] Post-training assertion that the artifact actually exists in the store (metadata logging silently succeeding while artifacts fail is a classic MLflow trap)
- [ ] `ml/registry.py` — register version, stage transitions, resolve `models:/marketpulse/Production`
- [ ] Persist reference feature distributions on promotion (Phase 6 drift depends on this)
- [ ] `model_versions` + `training_runs` tables — record rejections too

## Tests
- [ ] **Promotion gate:** a deliberately worse candidate is not promoted
- [ ] Labeling: known price sequence → expected labels; deadband boundary cases exact
- [ ] Split is chronological and non-overlapping; embargo gap present and correct size
- [ ] `assert not shuffled` — a guard test that the split function never reorders
- [ ] Recent-H rows excluded from the training set
- [ ] Insufficient data raises rather than training
- [ ] Baselines compute correctly on a synthetic set with known class prior
- [ ] Registry round-trip: register → transition → resolve returns the right version

## Watch out for
- If accuracy looks impressive (>60% on 3-class), assume leakage and go re-run the Phase 2 leakage tests. ~50–53% against a ~40% baseline is the realistic range.
- Never train without MLflow reachable. An untracked model is unreproducible and therefore worthless.
