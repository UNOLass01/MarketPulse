"""Envelope schema contract: round-trips, and rejects malformed/naive-datetime input."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from marketpulse.contracts.messages import TickEnvelope, TickPayload
from marketpulse.exceptions import SchemaValidationError
from marketpulse.messaging.serialization import deserialize_envelope, serialize_envelope

pytestmark = pytest.mark.contract


def _envelope() -> TickEnvelope:
    return TickEnvelope(
        emitted_at=datetime.now(UTC),
        symbol="BTC-USD",
        payload=TickPayload(
            price=Decimal("65000.12345678"),
            volume=Decimal("1234.5"),
            provider_observed_at=datetime.now(UTC),
        ),
    )


def test_envelope_round_trips_unchanged() -> None:
    original = _envelope()
    restored = deserialize_envelope(serialize_envelope(original))
    assert restored == original


def test_deserialize_malformed_json_raises_schema_validation_error() -> None:
    with pytest.raises(SchemaValidationError):
        deserialize_envelope(b"not json")


def test_deserialize_missing_field_raises_schema_validation_error() -> None:
    with pytest.raises(SchemaValidationError):
        deserialize_envelope(b'{"symbol": "BTC-USD"}')


def test_naive_emitted_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TickEnvelope(
            emitted_at=datetime(2026, 1, 1),  # naive, no tzinfo
            symbol="BTC-USD",
            payload=TickPayload(
                price=Decimal("1"), volume=Decimal("1"), provider_observed_at=datetime.now(UTC)
            ),
        )


def test_naive_provider_observed_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TickPayload(
            price=Decimal("1"), volume=Decimal("1"), provider_observed_at=datetime(2026, 1, 1)
        )


def test_non_positive_price_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TickPayload(price=Decimal("0"), volume=Decimal("1"), provider_observed_at=datetime.now(UTC))


def test_negative_volume_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TickPayload(
            price=Decimal("1"), volume=Decimal("-1"), provider_observed_at=datetime.now(UTC)
        )
