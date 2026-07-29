"""Consumer entrypoint: wiring only. Persists validated ticks to Postgres."""

from sqlalchemy.exc import DBAPIError

from marketpulse.config import get_settings
from marketpulse.contracts.messages import TickEnvelope
from marketpulse.exceptions import TransientError
from marketpulse.logging import get_logger
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.consumer import BaseConsumer
from marketpulse.messaging.serialization import deserialize_envelope
from marketpulse.messaging.topology import QUEUE_PERSIST
from marketpulse.storage.engine import make_engine, make_session_factory, session_scope
from marketpulse.storage.repositories.ticks import upsert_tick

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    get_logger("marketpulse", level=settings.log_level)

    engine = make_engine(settings.db)
    session_factory = make_session_factory(engine)

    def process(envelope: TickEnvelope) -> None:
        try:
            with session_scope(session_factory) as session:
                upsert_tick(session, envelope)
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
