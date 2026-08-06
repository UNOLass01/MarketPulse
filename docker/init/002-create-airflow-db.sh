#!/bin/bash
# Runs once, on first init of an empty postgres_data volume (see
# 001-create-mlflow-db.sh's comment -- same trigger condition).
#
# Airflow gets its own database for the same reason MLflow does: its
# SqlAlchemyStore creates its own tables, and at least one of Airflow's own
# table names could plausibly collide with an application or MLflow table
# in the future. One Postgres server, three logically separate databases.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "${AIRFLOW_DB_NAME:-airflow}" OWNER $POSTGRES_USER;
EOSQL
