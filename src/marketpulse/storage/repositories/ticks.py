"""Idempotent tick persistence."""

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.contracts.messages import TickEnvelope
from marketpulse.storage.models import RawTick
from marketpulse.storage.repositories.symbols import get_or_create_symbol_id


def upsert_tick(session: Session, envelope: TickEnvelope) -> None:
    """Insert a raw tick row; re-publishing the same envelope is a no-op.

    Idempotency is enforced by the ``(message_id, observed_at)`` unique
    constraint via ``ON CONFLICT DO NOTHING`` — publishing the same envelope
    twice yields exactly one row.
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
        )
        .on_conflict_do_nothing(constraint="uq_raw_ticks_message_observed")
    )
    session.execute(stmt)
