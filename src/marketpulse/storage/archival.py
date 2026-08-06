"""Partition export, verify, and drop for ``dag_data_archival`` (Phase 4).

Lives in ``storage/`` rather than ``jobs/`` — this is storage lifecycle
management in the same spirit as ``storage.partitions`` (which already owns
partition create/drop DDL), not a cross-domain wiring job. See ADR 0002.

The sequence is fixed and one-directional: export -> upload -> verify ->
record -> drop. A partition is only ever dropped after its export has been
uploaded *and independently verified* (row count + checksum) *and* recorded
in ``archived_partitions`` — phase-4 plan: "the verify step is
non-negotiable; it's what separates maintenance from data loss". Any
failure before ``drop_partition`` leaves the source partition untouched, so
a retry is always safe.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import ObjectStorageSettings
from marketpulse.exceptions import ArchivalVerificationError
from marketpulse.logging import get_logger
from marketpulse.storage.engine import session_scope
from marketpulse.storage.partitions import drop_partition, list_partitions
from marketpulse.storage.repositories.archival import record_archived_partition

if TYPE_CHECKING:
    # boto3-stubs is a dev-only dependency (type checking only) -- this
    # import must never execute at runtime, only under mypy.
    from mypy_boto3_s3.client import S3Client

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArchivalResult:
    table: str
    year: int
    month: int
    row_count: int
    object_key: str


def make_s3_client(settings: ObjectStorageSettings) -> "S3Client":
    """boto3 S3 client for the archive bucket."""
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
    )


def months_before_cutoff(year: int, month: int, *, as_of: date, retention_months: int) -> bool:
    """True iff ``year``/``month`` is more than ``retention_months`` months
    before ``as_of``'s month. Pure -- the actual archivability decision,
    factored out from the partition-listing I/O so it's unit-testable on
    its own.
    """
    cutoff_ordinal = as_of.year * 12 + (as_of.month - 1) - retention_months
    partition_ordinal = year * 12 + (month - 1)
    return partition_ordinal < cutoff_ordinal


def archivable_partitions(
    connection: Connection, table: str, *, as_of: date, retention_months: int
) -> list[tuple[int, int]]:
    """Existing partitions of ``table`` older than the hot-retention window."""
    return [
        (year, month)
        for year, month in list_partitions(connection, table)
        if months_before_cutoff(year, month, as_of=as_of, retention_months=retention_months)
    ]


def export_partition_to_parquet(
    connection: Connection, table: str, year: int, month: int, dest_dir: Path
) -> tuple[Path, int]:
    """Dump one partition's full contents to a local Parquet file. Returns
    the file path and row count actually written, so the caller never has
    to trust a second, possibly-stale count.
    """
    partition_name = f"{table}_{year:04d}_{month:02d}"
    frame = pd.read_sql(text(f'SELECT * FROM "{partition_name}"'), connection)  # noqa: S608
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{partition_name}.parquet"
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path, len(frame)


def _md5_hex(path: Path) -> str:
    digest = (
        hashlib.md5()
    )  # noqa: S324 -- integrity check against S3's ETag, not security-sensitive
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_and_get_etag(s3_client: "S3Client", bucket: str, key: str, path: Path) -> str:
    """Upload ``path`` to ``bucket``/``key`` as a single part (no multipart
    -- these are monthly partition exports, well under any multipart
    threshold) and return the server's ETag, unquoted. For a single-part
    upload, S3/MinIO's ETag is exactly the MD5 hex digest of the object
    body -- that equality is what :func:`verify_upload` checks.
    """
    with path.open("rb") as handle:
        response = s3_client.put_object(Bucket=bucket, Key=key, Body=handle)
    return response["ETag"].strip('"')


def verify_upload(path: Path, etag: str) -> bool:
    """True iff the uploaded object's ETag matches the local file's MD5 --
    proof the bytes that landed in object storage are the bytes exported,
    not silently truncated or corrupted in transit.
    """
    return _md5_hex(path) == etag


def archive_partition(
    engine: Engine,
    session_factory: sessionmaker[Session],
    s3_client: "S3Client",
    *,
    table: str,
    year: int,
    month: int,
    bucket: str,
    tmp_dir: Path,
) -> ArchivalResult:
    """Export -> upload -> verify -> record -> drop for one partition.

    Raises :class:`~marketpulse.exceptions.ArchivalVerificationError` and
    leaves the partition (and any already-written archive row) untouched if
    verification fails -- fail closed, per the phase-4 plan.
    """
    with engine.connect() as connection:
        local_path, row_count = export_partition_to_parquet(connection, table, year, month, tmp_dir)

    object_key = f"{table}/{year:04d}/{month:02d}.parquet"
    etag = upload_and_get_etag(s3_client, bucket, object_key, local_path)

    if not verify_upload(local_path, etag):
        raise ArchivalVerificationError(
            f"checksum mismatch for {object_key}: local file does not match uploaded ETag"
        )

    checksum = _md5_hex(local_path)
    with session_scope(session_factory) as session:
        record_archived_partition(
            session,
            table_name=table,
            partition_year=year,
            partition_month=month,
            row_count=row_count,
            object_key=object_key,
            checksum=checksum,
            archived_at=datetime.now(UTC),
        )

    with engine.begin() as connection:
        drop_partition(connection, table, year, month)

    local_path.unlink(missing_ok=True)

    logger.info(
        "archived partition",
        extra={
            "extra_fields": {
                "table": table,
                "year": year,
                "month": month,
                "row_count": row_count,
                "object_key": object_key,
            }
        },
    )
    return ArchivalResult(
        table=table, year=year, month=month, row_count=row_count, object_key=object_key
    )
