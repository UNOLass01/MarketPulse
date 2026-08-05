# MLflow tracking server, with a Postgres backend store and an S3-compatible
# (MinIO) artifact store. The official mlflow image ships neither the
# Postgres driver nor an S3 client, so this installs both explicitly rather
# than reaching for Kafka/K8s-style infra this project deliberately avoids.
#
# mlflow's version here must match the `mlflow==` pin in pyproject.toml --
# marketpulse.ml.registry uses the model-registry "stage" API, and a
# client/server version mismatch on a deprecated API is exactly the kind of
# thing that fails silently until promotion day.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==3.15.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.35.24

EXPOSE 5000

# backend-store-uri / artifact destination / credentials all come from env
# (docker-compose.yml), never hardcoded here -- CLAUDE.md rule #12.
CMD ["sh", "-c", "mlflow server \
    --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" \
    --artifacts-destination \"$MLFLOW_ARTIFACTS_DESTINATION\" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000"]
