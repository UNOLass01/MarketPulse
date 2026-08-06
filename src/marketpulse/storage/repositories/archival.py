"""Persistence for ``storage.archival``'s audit trail."""

from datetime import datetime

from sqlalchemy.orm import Session

from marketpulse.storage.models import ArchivedPartition


def record_archived_partition(
    session: Session,
    *,
    table_name: str,
    partition_year: int,
    partition_month: int,
    row_count: int,
    object_key: str,
    checksum: str,
    archived_at: datetime,
) -> int:
    """Record a successfully verified export. Called only after the verify
    step passes (phase-4 plan: "the verify step is non-negotiable; it's what
    separates maintenance from data loss") — a partition drop with no
    corresponding row here means something went wrong before this ran.
    """
    record = ArchivedPartition(
        table_name=table_name,
        partition_year=partition_year,
        partition_month=partition_month,
        row_count=row_count,
        object_key=object_key,
        checksum=checksum,
        archived_at=archived_at,
    )
    session.add(record)
    session.flush()
    return record.id
