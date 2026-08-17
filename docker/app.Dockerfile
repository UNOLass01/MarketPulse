# The marketpulse package plus its thin service entrypoints (services/api,
# services/dashboard). One image for both: they share the entire dependency
# set apart from Streamlit, and maintaining two nearly-identical Dockerfiles
# to save one package is a worse trade than the extra ~40MB.
#
# The producer and consumer are deliberately *not* wired into compose (they
# are run directly during development, see docs/plan/phase-1-streaming.md);
# this image can run them too, since it installs the same package.
FROM python:3.11-slim

WORKDIR /app

# libgomp1: LightGBM's compiled lib_lightgbm.so links against OpenMP. The
# API loads a LightGBM model through mlflow.pyfunc, so `import lightgbm`
# happens on the serving path -- without this the container starts and then
# fails on the first model load, which is the worst possible time to find
# out. Same reason docker/airflow.Dockerfile installs it.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# configs/ (ml.config resolves its hyperparameter file relative to the
# installed package's repo layout) and README.md (pyproject's readme field;
# the metadata build fails without it) -- same reasoning as the Airflow image.
COPY pyproject.toml README.md ./
COPY src ./src
COPY services ./services
COPY configs ./configs

# [dashboard] pulls in Streamlit. Installed here rather than in a separate
# stage because the dashboard service runs from this same image.
RUN pip install --no-cache-dir -e ".[dashboard]"

# Non-root: nothing here needs to write to the filesystem, and the API is
# the only process in this project exposed to inbound traffic.
RUN useradd --create-home --uid 10001 marketpulse
USER marketpulse

EXPOSE 8000

# Overridden per service in docker-compose.yml. One worker on purpose: the
# in-process ModelCache is per-process state, and N workers would mean N
# independent caches refreshing on N schedules -- so /api/v1/model/current
# could disagree with the model that just served you. Scaling out means
# scaling containers, each with its own honest cache.
CMD ["uvicorn", "services.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
