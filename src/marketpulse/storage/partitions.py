"""Monthly range-partition management for range-partitioned tables
(``raw_ticks`` on ``observed_at``, ``features`` on ``feature_ts``).
"""

from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Connection


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
