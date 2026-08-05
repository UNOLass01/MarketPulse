#!/bin/bash
# Runs once, on first init of an empty postgres_data volume (standard
# docker-entrypoint-initdb.d behavior -- re-creating the volume via
# `make reset` is what re-triggers this, not a normal `make up`).
#
# MLflow's SqlAlchemyStore creates its own tables -- including one literally
# named "model_versions" -- so it gets its own database rather than sharing
# ${MP_DB__NAME}, which already has an application table of that same name
# (see storage.models.ModelVersion). Sharing a database would silently
# collide the two.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "${MLFLOW_DB_NAME:-mlflow}" OWNER $POSTGRES_USER;
EOSQL
