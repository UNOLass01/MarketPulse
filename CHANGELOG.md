# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Phase 3: ML pipeline. `ml/labeling.py` (forward-return deadband labeling,
  data-driven θ), `ml/dataset.py` (as-of price join, insufficient-history/gap
  filtering, chronological 70/15/15 split with an `H` embargo gap, sufficiency
  guards), `ml/train.py` (LightGBM multiclass, class-weighted, no scaler),
  `ml/evaluate.py` (majority/persistence/random-stratified baselines, full
  metric suite, promotion gate), `ml/registry.py` (MLflow tracking + model
  registry, stage transitions, post-log artifact-persisted assertion),
  `ml/pipeline.py` (end-to-end orchestrator). New `training_runs` and
  `model_versions` tables record every run including rejections. MLflow
  tracking server + Postgres backend + MinIO (S3-compatible) artifact store
  added to Docker Compose.
- Phase 0: repository skeleton, configuration, logging, exceptions, Docker Compose (Postgres + RabbitMQ), Alembic, Makefile, pre-commit, CI.
