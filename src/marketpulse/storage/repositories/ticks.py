"""Idempotent tick persistence."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.contracts.messages import TickEnvelope
from marketpulse.storage.models import RawTick
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id


def upsert_tick(session: Session, envelope: TickEnvelope, *, source: str = "stream") -> bool:
    """Insert a raw tick row; re-publishing the same envelope is a no-op.

    ``raw_ticks`` carries two unique constraints — ``(message_id,
    observed_at)`` for envelope-level idempotency, and ``(symbol_id,
    observed_at)`` because a provider can legitimately report the same
    ``observed_at`` twice across separate poll cycles (e.g. its own data
    hasn't refreshed yet) even though our poller assigned a new
    ``message_id`` each time. ``ON CONFLICT DO NOTHING`` is left untargeted
    so *either* constraint collapses the insert to a no-op rather than
    raising an ``IntegrityError`` on the one it doesn't name. Returns
    ``True`` iff a new row was actually inserted, so callers can keep other
    idempotent side effects (like pushing into in-memory feature window
    state) from double-processing a duplicate.
    """
    symbol_id = get_or_create_symbol_id(session, envelope.symbol)
    stmt = (
        pg_insert(RawTick)
        .values(
            observed_at=envelope.payload.provider_observed_at,
            symbol_id=symbol_id,
            message_id=envelope.message_id,
            correlation_id=envelope.correlation_id,
            schema_version=envelope.schema_version,
            emitted_at=envelope.emitted_at,
            price=envelope.payload.price,
            volume=envelope.payload.volume,
            source=source,
        )
        .on_conflict_do_nothing()
        .returning(RawTick.id)
    )
    result = session.execute(stmt)
    return result.first() is not None


def list_recent_ticks(session: Session, symbol_id: int, since: datetime) -> list[RawTick]:
    """Ascending ticks for ``symbol_id`` at or after ``since``.

    Used to warm up in-memory feature window state on consumer restart and
    by the offline seed path — never for anything model-relevant, since
    ``observed_at`` (not ``ingested_at``) is the timing field to filter on
    (CLAUDE.md rule #6).
    """
    stmt = (
        select(RawTick)
        .where(RawTick.symbol_id == symbol_id, RawTick.observed_at >= since)
        .order_by(RawTick.observed_at.asc())
    )
    return list(session.execute(stmt).scalars())
