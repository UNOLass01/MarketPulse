"""Envelope (de)serialization for messages carried over RabbitMQ.

Deserialization failures are always :class:`SchemaValidationError`
(``PermanentError``) so callers can route malformed messages straight to the
dead-letter queue without inspecting the underlying exception type.
"""

from pydantic import ValidationError

from marketpulse.contracts.messages import TickEnvelope
from marketpulse.exceptions import SchemaValidationError

CONTENT_TYPE = "application/json"


def serialize_envelope(envelope: TickEnvelope) -> bytes:
    """Serialize an envelope to its wire representation."""
    return envelope.model_dump_json().encode("utf-8")


def deserialize_envelope(data: bytes) -> TickEnvelope:
    """Parse a wire message back into a :class:`TickEnvelope`.

    Raises:
        SchemaValidationError: the payload is not valid JSON or does not
            conform to the envelope schema.
    """
    try:
        return TickEnvelope.model_validate_json(data)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc
