# Airflow scheduler + API server + DAG processor, with the marketpulse
# package installed so DAG task callables can `import marketpulse.*`
# directly (CLAUDE.md rule #9: "DAG files are thin callables importing from
# marketpulse.*").
#
# Airflow's version here must match the `apache-airflow==` pin in
# pyproject.toml -- same discipline as docker/mlflow.Dockerfile's
# client/server pin. It's 3.x, not 2.x, and specifically >=3.3.0: that's the
# first apache-airflow-core release that supports sqlalchemy>=2.0, which
# marketpulse's own ORM code (storage/models.py) requires and Airflow 2.x /
# early 3.x can't coexist with in one environment. See ADR 0003.
FROM apache/airflow:3.3.0-python3.11

WORKDIR /opt/airflow

# libgomp1: LightGBM's compiled lib_lightgbm.so is linked against libgomp
# (OpenMP) for its multi-threaded training -- the base image's slim Debian
# layer doesn't carry it, and without this dag_model_retraining fails to
# even *import* (ImportError deep inside `import lightgbm`), which is
# exactly the "broken DAG silently disables scheduling" failure mode the
# phase-4 plan's DAG-import test exists to catch. Caught by that test
# against this exact image, not guessed.
USER root
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
USER airflow

# configs/ (ml/config.py resolves its default hyperparameter file relative
# to the installed package's location, which under an editable install
# means the repo layout has to exist here too, not just src/) and README.md
# (referenced by pyproject.toml's [project] readme field -- the editable
# install's metadata build fails without it).
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src
COPY --chown=airflow:root configs ./configs

# No --no-deps: marketpulse's dependencies are all lower-bound (`>=`), and
# the base image already has apache-airflow's own exact pins installed --
# pip's resolver adds marketpulse's extra packages (pandas, lightgbm,
# mlflow, boto3, ...) without touching anything Airflow needs. Verified
# conflict-free locally (`pip check`) before picking this Airflow version;
# see ADR 0003.
RUN pip install --no-cache-dir -e .
