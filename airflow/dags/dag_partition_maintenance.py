"""dag_partition_maintenance: create upcoming monthly partitions ahead of
time (Phase 4).

A missing partition is an insert error at midnight on the 1st — this DAG
exists purely to make that never happen (phase-4 plan: "watch out for" a
newly deployed DAG with a past start_date; this one has no such risk since
it only ever creates partitions ahead of the current date). All partition
DDL lives in ``marketpulse.storage.partitions`` (CLAUDE.md rule #9);
``ensure_partition`` is itself idempotent (``CREATE TABLE IF NOT EXISTS``),
so a rerun is always safe.
"""

from datetime import date, datetime, timedelta

from _dag_common import DEFAULT_ARGS, logger, mp_engine, mp_settings
from airflow.sdk import DAG, task

from marketpulse.storage.partitions import ensure_partitions_covering

PARTITIONED_TABLES = ("raw_ticks", "features")
MONTHS_AHEAD = 2


@task(
    execution_timeout=timedelta(minutes=5),
    retries=2,
    retry_delay=timedelta(minutes=5),
    retry_exponential_backoff=True,
)
def ensure_upcoming_partitions() -> None:
    settings = mp_settings()
    engine = mp_engine(settings)
    with engine.begin() as connection:
        for table in PARTITIONED_TABLES:
            ensure_partitions_covering(connection, table, date.today(), MONTHS_AHEAD)
    logger.info(
        "partitions ensured",
        extra={"extra_fields": {"tables": PARTITIONED_TABLES, "months_ahead": MONTHS_AHEAD}},
    )


with DAG(
    dag_id="dag_partition_maintenance",
    schedule="@monthly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["storage", "phase-4"],
    doc_md=__doc__,
) as dag:
    ensure_upcoming_partitions()
