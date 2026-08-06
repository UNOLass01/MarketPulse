"""dag_data_archival: export, verify, and drop partitions past hot retention
(Phase 4).

Runs daily; most days find nothing to archive (partitions are monthly-
granularity, this schedule isn't). The sequence is fixed and
one-directional — export -> upload -> verify (row count + checksum) ->
record -> drop — and a partition is never dropped without a prior,
independently verified export. The verify step is non-negotiable; it's what
separates maintenance from data loss (phase-4 plan). All of that lives in
``marketpulse.storage.archival`` (CLAUDE.md rule #9); a verification
failure there raises ``ArchivalVerificationError`` (transient), which
Airflow's own retry — not this file — handles.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

from _dag_common import DEFAULT_ARGS, logger, mp_engine, mp_session_factory, mp_settings
from airflow.sdk import DAG, task

from marketpulse.storage.archival import archivable_partitions, archive_partition, make_s3_client

#: Both range-partitioned tables (storage.models: RawTick, Feature).
PARTITIONED_TABLES = ("raw_ticks", "features")
TMP_DIR = Path("/opt/airflow/tmp/archival")


@task(
    execution_timeout=timedelta(hours=1),
    retries=2,
    retry_delay=timedelta(minutes=10),
    retry_exponential_backoff=True,
)
def archive_hot_partitions() -> list[dict[str, object]]:
    settings = mp_settings()
    engine = mp_engine(settings)
    session_factory = mp_session_factory(settings)
    s3_client = make_s3_client(settings.object_store)

    with engine.connect() as connection:
        to_archive = {
            table: archivable_partitions(
                connection,
                table,
                as_of=date.today(),
                retention_months=settings.object_store.hot_retention_months,
            )
            for table in PARTITIONED_TABLES
        }

    archived = []
    for table, partitions in to_archive.items():
        for year, month in partitions:
            result = archive_partition(
                engine,
                session_factory,
                s3_client,
                table=table,
                year=year,
                month=month,
                bucket=settings.object_store.archive_bucket,
                tmp_dir=TMP_DIR,
            )
            archived.append(
                {
                    "table": result.table,
                    "year": result.year,
                    "month": result.month,
                    "row_count": result.row_count,
                }
            )

    logger.info("archival run complete", extra={"extra_fields": {"archived": archived}})
    return archived


with DAG(
    dag_id="dag_data_archival",
    schedule="0 4 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["storage", "phase-4"],
    doc_md=__doc__,
) as dag:
    archive_hot_partitions()
