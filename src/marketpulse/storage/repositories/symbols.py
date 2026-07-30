"""Symbol lookups/creation."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.storage.models import Symbol


def get_or_create_symbol_id(session: Session, code: str) -> int:
    """Return the id for ``code``, inserting a new symbol row if needed."""
    stmt = (
        pg_insert(Symbol)
        .values(code=code)
        .on_conflict_do_update(index_elements=["code"], set_={"code": code})
        .returning(Symbol.id)
    )
    result = session.execute(stmt)
    return result.scalar_one()


def list_symbols(session: Session) -> list[Symbol]:
    """All known symbols — used to warm up in-memory feature window state."""
    return list(session.execute(select(Symbol)).scalars())
