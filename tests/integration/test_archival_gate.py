"""Archival's verify-before-drop gate (phase-4 plan test list: "Archival
verify step fails closed on a checksum mismatch (drop must not execute)").

Uses a fake S3 client that always returns a wrong ETag, against a real
Postgres partition — proves ``archive_partition`` raises *before* touching
the partition or writing an audit row, not just that ``verify_upload``
itself returns ``False`` in isolation (already covered in
``tests/unit/test_archival.py``).
"""

from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.exceptions import ArchivalVerificationError
from marketpulse.storage.archival import archive_partition
from marketpulse.storage.engine import session_scope
from marketpulse.storage.models import RawTick
from marketpulse.storage.partitions import ensure_partitions_covering
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id

pytestmark = pytest.mark.integration


class _FakeS3ClientWithWrongEtag:
    """put_object "succeeds" but always reports an ETag that can't match
    the uploaded bytes -- simulates silent corruption in transit.
    """

    def put_object(self, *, Bucket: str, Key: str, Body: object) -> dict[str, str]:  # noqa: N803
        return {"ETag": '"' + "0" * 32 + '"'}


def _seed_one_tick(session_factory: sessionmaker[Session], *, observed_at: datetime) -> None:
    with session_scope(session_factory) as session:
        symbol_id = get_or_create_symbol_id(session, "GATE-USD")
        session.add(
            RawTick(
                observed_at=observed_at,
                symbol_id=symbol_id,
                message_id=uuid4(),
                correlation_id=uuid4(),
                schema_version=1,
                emitted_at=observed_at,
                price=100,
                volume=1,
                source="seed",
            )
        )


def _partition_exists(engine: Engine, table: str, year: int, month: int) -> bool:
    name = f"{table}_{year:04d}_{month:02d}"
    with engine.connect() as connection:
        return (
            connection.execute(
                text(
                    "SELECT count(*) FROM pg_inherits "
                    "JOIN pg_class c ON pg_inherits.inhrelid = c.oid "
                    "WHERE c.relname = :name"
                ),
                {"name": name},
            ).scalar_one()
            > 0
        )


def test_verification_failure_blocks_drop_and_leaves_no_audit_row(
    engine: Engine, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    year, month = 2025, 2
    observed_at = datetime(year, month, 10, tzinfo=UTC)
    with engine.begin() as connection:
        ensure_partitions_covering(connection, "raw_ticks", date(year, month, 1), months_ahead=0)
    _seed_one_tick(session_factory, observed_at=observed_at)

    assert _partition_exists(engine, "raw_ticks", year, month)

    with pytest.raises(ArchivalVerificationError):
        archive_partition(
            engine,
            session_factory,
            _FakeS3ClientWithWrongEtag(),  # type: ignore[arg-type]
            table="raw_ticks",
            year=year,
            month=month,
            bucket="irrelevant-bucket",
            tmp_dir=tmp_path,
        )

    # Fail closed: the partition (and its data) must still be there, and no
    # audit row was written claiming a successful archive.
    assert _partition_exists(engine, "raw_ticks", year, month)
    with session_factory() as session:
        count = session.execute(
            text(
                "SELECT count(*) FROM archived_partitions "
                "WHERE table_name='raw_ticks' AND partition_year=:year AND partition_month=:month"
            ),
            {"year": year, "month": month},
        ).scalar_one()
    assert count == 0
