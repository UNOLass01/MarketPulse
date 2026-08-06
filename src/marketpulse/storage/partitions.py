"""Monthly range-partition management for range-partitioned tables
(``raw_ticks`` on ``observed_at``, ``features`` on ``feature_ts``).
"""

import re
from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Connection

_PARTITION_NAME_RE = re.compile(r"^(?P<table>.+)_(?P<year>\d{4})_(?P<month>\d{2})$")


def _partition_name(table: str, year: int, month: int) -> str:
    return f"{table}_{year:04d}_{month:02d}"


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
    return start, date(end_year, end_month, 1)


def ensure_partition(connection: Connection, table: str, year: int, month: int) -> None:
    """Create ``table``'s partition for ``year``/``month`` if it doesn't already exist."""
    name = _partition_name(table, year, month)
    start, end = _month_bounds(year, month)
    # FOR VALUES FROM/TO does not accept bind parameters (DDL, not DML) —
    # the bounds are internally computed dates, never user input.
    connection.execute(
        text(
            f'CREATE TABLE IF NOT EXISTS "{name}" '
            f"PARTITION OF {table} "
            f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
        )
    )


def ensure_partitions_covering(
    connection: Connection, table: str, start: date, months_ahead: int
) -> None:
    """Ensure ``table`` has partitions from ``start``'s month through ``months_ahead`` beyond it."""
    year, month = start.year, start.month
    for _ in range(months_ahead + 1):
        ensure_partition(connection, table, year, month)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def list_partitions(connection: Connection, table: str) -> list[tuple[int, int]]:
    """``(year, month)`` for every existing partition of ``table``, ascending.

    Reads structure via ``pg_inherits`` (the real source of truth for which
    child tables exist), not just by pattern-matching a naming convention —
    but the year/month themselves are still parsed from the name, since that
    encoding is this project's own and Postgres doesn't track partition
    bounds as a queryable (year, month) pair.
    """
    rows = connection.execute(
        text(
            "SELECT child.relname FROM pg_inherits "
            "JOIN pg_class parent ON pg_inherits.inhparent = parent.oid "
            "JOIN pg_class child ON pg_inherits.inhrelid = child.oid "
            "WHERE parent.relname = :table"
        ),
        {"table": table},
    ).all()

    partitions = []
    for (name,) in rows:
        match = _PARTITION_NAME_RE.match(name)
        if match and match.group("table") == table:
            partitions.append((int(match.group("year")), int(match.group("month"))))
    return sorted(partitions)


def drop_partition(connection: Connection, table: str, year: int, month: int) -> None:
    """Drop ``table``'s partition for ``year``/``month`` — and its data —
    outright. Only ever call this after the caller has independently
    verified the data made it to durable storage elsewhere (see
    ``storage.archival``); this function itself has no way to know that and
    does not check.
    """
    name = _partition_name(table, year, month)
    connection.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
