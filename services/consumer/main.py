"""Consumer entrypoint: wiring only.

Persists validated ticks and, for each newly-inserted one, its derived
feature vector — computed by the same pure ``features.pipeline`` the offline
seed script uses, so streamed and backfilled feature rows are produced by
identical code (see phase-2 exit criterion: online == offline features).
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from marketpulse.config import get_settings
from marketpulse.contracts.messages import TickEnvelope
from marketpulse.exceptions import TransientError
from marketpulse.features.pipeline import compute_feature_vector
from marketpulse.features.windows import Observation, WindowStore
from marketpulse.logging import get_logger
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.consumer import BaseConsumer
from marketpulse.messaging.serialization import deserialize_envelope
from marketpulse.messaging.topology import QUEUE_PERSIST
from marketpulse.storage.engine import make_engine, make_session_factory, session_scope
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.symbols import list_symbols
from marketpulse.storage.repositories.ticks import list_recent_ticks, upsert_tick

logger = get_logger(__name__)


def _warm_up(session: Session, window_store: WindowStore, max_history: timedelta) -> None:
    """Rebuild in-memory window state from recent DB history.

    Skipping this on restart would silently produce a NaN/insufficient-history
    gap in the next computed feature row for every known symbol.
    """
    since = datetime.now(UTC) - max_history
    for symbol in list_symbols(session):
        ticks = list_recent_ticks(session, symbol.id, since)
        for tick in ticks:
            window_store.push(
                symbol.code,
                Observation(
                    observed_at=tick.observed_at,
                    price=float(tick.price),
                    volume=float(tick.volume),
                ),
            )
        logger.info(
            "warmed up window state",
            extra={"extra_fields": {"symbol": symbol.code, "observations": len(ticks)}},
        )


def main() -> None:
    settings = get_settings()
    get_logger("marketpulse", level=settings.log_level)

    engine = make_engine(settings.db)
    session_factory = make_session_factory(engine)

    max_history = timedelta(hours=settings.features.max_history_hours)
    gap_threshold = timedelta(seconds=settings.features.gap_threshold_seconds)
    window_store = WindowStore(
        max_history=max_history, max_points=settings.features.max_buffer_points
    )

    with session_scope(session_factory) as session:
        _warm_up(session, window_store, max_history)

    def process(envelope: TickEnvelope) -> None:
        try:
            with session_scope(session_factory) as session:
                inserted = upsert_tick(session, envelope)
                if not inserted:
                    # Redelivered duplicate: skip window/feature side effects
                    # too, or a retried message would double-count itself in
                    # the in-memory ring buffer.
                    return

                observed_at = envelope.payload.provider_observed_at
                window_store.push(
                    envelope.symbol,
                    Observation(
                        observed_at=observed_at,
                        price=float(envelope.payload.price),
                        volume=float(envelope.payload.volume),
                    ),
                )
                vector = compute_feature_vector(
                    envelope.symbol,
                    window_store.snapshot(envelope.symbol),
                    as_of=observed_at,
                    gap_threshold=gap_threshold,
                )
                upsert_feature_vector(session, vector)
        except DBAPIError as exc:
            raise TransientError("database unavailable") from exc

    connection_manager = ConnectionManager(settings.rabbitmq.url)
    consumer = BaseConsumer(connection_manager, QUEUE_PERSIST, deserialize_envelope, process)

    try:
        consumer.start()
    except KeyboardInterrupt:
        consumer.stop()
    finally:
        connection_manager.close()
        engine.dispose()
        logger.info("consumer shut down cleanly")


if __name__ == "__main__":
    main()
