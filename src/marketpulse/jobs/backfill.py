"""Feature backfill for ``dag_feature_backfill`` (Phase 4).

Recomputes feature rows from *already-ingested* raw ticks over a date
range — distinct from ``scripts/seed_historical.py``, which fetches new raw
ticks from the provider. This is what you re-run after a
``feature_set_version`` bump, or to fix a detected gap, without re-fetching
anything.

Lives in ``jobs/``, not ``features/`` — ``features/`` must stay pure/no-I/O
(CLAUDE.md rule #1); see ADR 0002 for why this orchestration gets its own
package instead.

Chunked by symbol + day to bound memory (phase-4 plan). The window's warm-up
(``max_history`` before ``start``) is fetched once per symbol; each chunk
after that only fetches its own day, so memory stays proportional to one
``WindowStore`` (already bounded by ``max_buffer_points``) plus one chunk's
ticks, never the whole requested range.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import Settings
from marketpulse.features.pipeline import compute_feature_vector
from marketpulse.features.windows import Observation, WindowStore
from marketpulse.logging import get_logger
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id
from marketpulse.storage.repositories.ticks import list_ticks_in_range

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackfillResult:
    symbol: str
    start: datetime
    end: datetime
    rows_upserted: int


def _day_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def backfill_symbol(
    session_factory: sessionmaker[Session],
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    max_history: timedelta,
    max_buffer_points: int,
    gap_threshold: timedelta,
) -> BackfillResult:
    """Recompute and upsert feature rows for ``symbol`` over ``[start,
    end)``. Idempotent: re-running upserts the same rows via
    ``ON CONFLICT DO NOTHING`` (``features.upsert_feature_vector``), so a
    rerun after a partial failure — or a deliberate full rerun — never
    double-counts or duplicates a row.
    """
    window = WindowStore(max_history=max_history, max_points=max_buffer_points)
    rows_upserted = 0

    with session_scope(session_factory) as session:
        symbol_id = get_or_create_symbol_id(session, symbol)
        for tick in list_ticks_in_range(session, symbol_id, start - max_history, start):
            window.push(
                symbol,
                Observation(
                    observed_at=tick.observed_at, price=float(tick.price), volume=float(tick.volume)
                ),
            )

    for chunk_start, chunk_end in _day_chunks(start, end):
        with session_scope(session_factory) as session:
            for tick in list_ticks_in_range(session, symbol_id, chunk_start, chunk_end):
                window.push(
                    symbol,
                    Observation(
                        observed_at=tick.observed_at,
                        price=float(tick.price),
                        volume=float(tick.volume),
                    ),
                )
                vector = compute_feature_vector(
                    symbol,
                    window.snapshot(symbol),
                    as_of=tick.observed_at,
                    gap_threshold=gap_threshold,
                )
                upsert_feature_vector(session, vector)
                rows_upserted += 1

        logger.info(
            "backfilled chunk",
            extra={
                "extra_fields": {
                    "symbol": symbol,
                    "chunk_start": chunk_start.isoformat(),
                    "chunk_end": chunk_end.isoformat(),
                    "rows_upserted_so_far": rows_upserted,
                }
            },
        )

    return BackfillResult(symbol=symbol, start=start, end=end, rows_upserted=rows_upserted)


def backfill_features(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    start: datetime,
    end: datetime,
    symbols: Sequence[str],
) -> list[BackfillResult]:
    """Entry point ``dag_feature_backfill`` calls — one result per symbol."""
    max_history = timedelta(hours=settings.features.max_history_hours)
    gap_threshold = timedelta(seconds=settings.features.gap_threshold_seconds)
    max_buffer_points = settings.features.max_buffer_points

    return [
        backfill_symbol(
            session_factory,
            symbol,
            start=start,
            end=end,
            max_history=max_history,
            max_buffer_points=max_buffer_points,
            gap_threshold=gap_threshold,
        )
        for symbol in symbols
    ]
