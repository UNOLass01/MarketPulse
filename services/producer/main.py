"""Producer entrypoint: wiring only. Polls CoinGecko, publishes tick envelopes."""

import signal
from types import FrameType

from marketpulse.config import get_settings
from marketpulse.ingestion.poller import Poller
from marketpulse.ingestion.providers.coingecko import CoinGeckoProvider
from marketpulse.ingestion.publisher import Publisher
from marketpulse.logging import get_logger
from marketpulse.messaging.connection import ConnectionManager

logger = get_logger(__name__)

SYMBOL_TO_COIN_ID = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
}


def main() -> None:
    settings = get_settings()
    get_logger("marketpulse", level=settings.log_level)

    connection_manager = ConnectionManager(settings.rabbitmq.url)
    connection_manager.channel()  # eagerly connect + declare topology before polling starts

    provider = CoinGeckoProvider(SYMBOL_TO_COIN_ID)
    publisher = Publisher(connection_manager)
    poller = Poller(provider, publisher, list(SYMBOL_TO_COIN_ID))

    def handle_shutdown(signum: int, frame: FrameType | None) -> None:
        logger.info("shutdown signal received, stopping poller")
        poller.stop()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        poller.run()
    finally:
        publisher.close()
        connection_manager.close()
        logger.info("producer shut down cleanly")


if __name__ == "__main__":
    main()
