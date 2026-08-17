"""Backfill historical OHLCV ticks and compute their features offline.

Usage: python scripts/seed_historical.py [--days N]

Writes directly to Postgres (raw ticks tagged ``source='seed'``, plus their
feature rows) without touching RabbitMQ — seeding is a one-shot batch
operation, not a streaming one. Feature computation reuses the exact same
``features.pipeline.compute_feature_vector`` / ``features.windows.WindowStore``
the streaming consumer uses (see ``services/consumer/main.py``), so seeded
(offline) and streamed (online) feature rows are produced by identical code
— this *is* the offline half of the phase-2 online/offline parity guarantee.
"""

import argparse
from datetime import timedelta

from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import get_settings
from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.features.pipeline import compute_feature_vector
from marketpulse.features.windows import Observation, WindowStore
from marketpulse.ingestion.providers.base import Observation as ProviderObservation
from marketpulse.ingestion.providers.coingecko import CoinGeckoProvider
from marketpulse.logging import get_logger
from marketpulse.storage.engine import make_engine, make_session_factory, session_scope
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.ticks import upsert_tick

logger = get_logger(__name__)

SYMBOL_TO_COIN_ID = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=90, help="Days of history to backfill (default: 90)"
    )
    return parser.parse_args()


def _seed_symbol(
    symbol: str,
    # Provider observations (Decimal price/volume), not the float-valued
    # ``features.windows.Observation`` this converts them into below. Two
    # different types share the name; keeping both imports explicit is what
    # makes the conversion at the window.push() call visible.
    observations: list[ProviderObservation],
    *,
    session_factory: sessionmaker[Session],
    max_history: timedelta,
    max_buffer_points: int,
    gap_threshold: timedelta,
) -> None:
    window = WindowStore(max_history=max_history, max_points=max_buffer_points)
    inserted_count = 0

    with session_scope(session_factory) as session:
        for obs in observations:
            envelope = TickEnvelope(
                emitted_at=obs.observed_at,
                symbol=symbol,
                payload=TickPayload(
                    price=obs.price, volume=obs.volume, provider_observed_at=obs.observed_at
                ),
            )
            inserted = upsert_tick(session, envelope, source="seed")
            if not inserted:
                continue
            inserted_count += 1

            observed_at = envelope.payload.provider_observed_at
            window.push(
                symbol,
                Observation(
                    observed_at=observed_at,
                    price=float(obs.price),
                    volume=float(obs.volume),
                ),
            )
            vector = compute_feature_vector(
                symbol,
                window.snapshot(symbol),
                as_of=observed_at,
                gap_threshold=gap_threshold,
            )
            upsert_feature_vector(session, vector)

    logger.info(
        "seeded symbol",
        extra={
            "extra_fields": {
                "symbol": symbol,
                "fetched": len(observations),
                "inserted": inserted_count,
            }
        },
    )


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    get_logger("marketpulse", level=settings.log_level)

    engine = make_engine(settings.db)
    session_factory = make_session_factory(engine)

    max_history = timedelta(hours=settings.features.max_history_hours)
    gap_threshold = timedelta(seconds=settings.features.gap_threshold_seconds)
    max_buffer_points = settings.features.max_buffer_points

    provider = CoinGeckoProvider(SYMBOL_TO_COIN_ID)

    try:
        for symbol in SYMBOL_TO_COIN_ID:
            observations = sorted(
                provider.fetch_history(symbol, days=args.days),
                key=lambda observation: observation.observed_at,
            )
            _seed_symbol(
                symbol,
                observations,
                session_factory=session_factory,
                max_history=max_history,
                max_buffer_points=max_buffer_points,
                gap_threshold=gap_threshold,
            )
    finally:
        engine.dispose()

    logger.info("seed complete")


if __name__ == "__main__":
    main()
