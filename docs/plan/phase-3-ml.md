# Phase 3 — ML Pipeline

**Objective:** a trained, tracked, registered model with an honest evaluation.
**Complexity:** Medium · **Effort:** ~2 days · **Depends on:** Phase 2 + seeded history
**Exit criterion:** a model sits in MLflow Production stage, fully reproducible from run ID + git SHA, with metrics compared against all baselines — **including if the outcome was "did not promote"**.

## Tasks

### Labeling
- [x] `ml/labeling.py` — forward return over horizon `H`, 3-class with deadband `θ`:
      UP if return > θ, DOWN if < −θ, else STABLE
- [x] `θ` derived from the empirical |return| distribution (~0.4 quantile), not picked arbitrarily
- [x] `H` and `θ` are logged hyperparameters — changing either changes the problem
- [x] Training query **excludes the most recent H** of data (those rows aren't labelable yet)

### Dataset assembly
- [x] `ml/dataset.py` — join features to forward-shifted price, filter on `observed_at`
- [x] Drop rows flagged `insufficient_history` or `has_gap`
- [x] Temporal split 70/15/15, chronological, with an **embargo gap of H between segments** (a contiguous split leaks H of validation into training)
- [x] Sufficiency check: minimum row count + class balance sanity → fail loudly, don't train on a thin sample

### Training
- [x] `ml/train.py` — LightGBM multiclass, early stopping on validation logloss
- [x] Class weighting for imbalance
- [x] **No scaler/normaliser** — LightGBM is scale-invariant; skipping it removes a whole class of "scaler wasn't persisted" bugs
- [x] Hyperparameters from a versioned config file, never hardcoded

### Evaluation
- [x] `ml/evaluate.py` — baselines: majority class, persistence, random-stratified, **and the incumbent Production model on the same test window**
- [x] Metrics: accuracy, **macro**-F1, per-class precision/recall, logloss, Brier, confusion matrix
- [x] Accuracy stratified by time bucket (catches a model that only works in one regime)
- [x] **Promotion gate:** beats every naive baseline on macro-F1 AND beats/matches incumbent AND no per-class collapse AND passes calibration sanity. Otherwise → Staging with a recorded `rejection_reason`.

### MLflow
- [x] Tracking server + Postgres backend + **S3-compatible artifact store** (MinIO locally via `MLFLOW_S3_ENDPOINT_URL`, real S3/R2 in prod — config only, no code change)
- [x] Log per run: params, `H`, `θ`, feature set version + ordered feature list, window boundaries, row counts, class distribution, all metrics, model artifact **with signature**, feature importances, confusion matrix, git SHA
- [x] Post-training assertion that the artifact actually exists in the store (metadata logging silently succeeding while artifacts fail is a classic MLflow trap)
- [x] `ml/registry.py` — register version, stage transitions, resolve `models:/marketpulse/Production`
- [x] Persist reference feature distributions on promotion (Phase 6 drift depends on this)
- [x] `model_versions` + `training_runs` tables — record rejections too

## Tests
- [x] **Promotion gate:** a deliberately worse candidate is not promoted
- [x] Labeling: known price sequence → expected labels; deadband boundary cases exact
- [x] Split is chronological and non-overlapping; embargo gap present and correct size
- [x] `assert not shuffled` — a guard test that the split function never reorders
- [x] Recent-H rows excluded from the training set
- [x] Insufficient data raises rather than training
- [x] Baselines compute correctly on a synthetic set with known class prior
- [x] Registry round-trip: register → transition → resolve returns the right version

## Watch out for
- If accuracy looks impressive (>60% on 3-class), assume leakage and go re-run the Phase 2 leakage tests. ~50–53% against a ~40% baseline is the realistic range.
- Never train without MLflow reachable. An untracked model is unreproducible and therefore worthless.

## Implementation notes (added on completion)

- Feature-to-model wiring: `storage/repositories/training_data.py` joins
  `features` to `raw_ticks` on `(symbol_id, observed_at = feature_ts)` --
  the only new query this phase adds, filtered on `feature_ts`
  (CLAUDE.md rule #6), never `computed_at`.
- The forward-price join in `ml/dataset.py` uses an as-of merge (not a
  fixed row offset) because tick spacing isn't uniform; its tolerance is
  deliberately *not* defaulted to `H` (a tolerance that wide would silently
  borrow a price up to `2H` late) -- callers pass
  `FeaturesSettings.gap_threshold_seconds`, the same knob the online
  feature pipeline already uses for gap detection.
- A real bug this phase caught: a feature that's legitimately `None` on
  every row of a split (e.g. `roc_1m` when tick cadence is coarser than its
  1-minute window) makes pandas infer that column as `dtype=object`, which
  both LightGBM and MLflow's signature inference reject outright. Fixed by
  forcing every feature column to `float64` after assembly
  (`ml/dataset.py::_rows_to_frame`).
- `ml/registry.py` uses MLflow's model-registry **stage** API
  (Staging/Production/Archived), not the newer alias API -- stages are
  deprecated since MLflow 2.9 but this project's exit criterion is
  explicitly "Production stage", and the pinned server/client version
  (3.15.1, matched in `docker/mlflow.Dockerfile` and `pyproject.toml`)
  still fully supports it. Revisit via an ADR if a future MLflow major
  release removes stages.
- MLflow gets its own Postgres database (`docker/init/001-create-mlflow-db.sh`),
  not `${MP_DB__NAME}` -- MLflow's own backend store creates a table
  literally named `model_versions`, which would collide with this
  project's own table of that name.
- Full stack (`make up` with the new `mlflow`/`minio`/`minio-init`
  services) was brought up and exercised end-to-end during this phase:
  real Postgres, a real MLflow tracking server, and a real MinIO
  S3-compatible artifact store all wired together and verified (model
  artifacts confirmed present in the MinIO bucket via `mc ls`, not just
  assumed from a 200 response).
- `mypy` could not be run in this environment (a pre-existing Windows
  Application Control policy blocks a DLL mypy itself depends on, before
  any project code is even read) -- `ruff` and `black --check` both pass
  clean across `src` and `tests`, and this is unrelated to this phase's
  changes.
