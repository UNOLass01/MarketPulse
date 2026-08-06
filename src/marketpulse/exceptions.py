"""Exception hierarchy.

Every exception raised by ``marketpulse`` code is either a ``TransientError``
(retry may succeed — network blip, DB unavailable, broker timeout) or a
``PermanentError`` (retry will never succeed — malformed message, schema
violation, missing config). Handlers branch on this split, not on exception
identity, so new error types must subclass one of the two.
"""


class MarketPulseError(Exception):
    """Base class for all application-raised errors."""


class TransientError(MarketPulseError):
    """Retryable: the same operation may succeed later without changes."""


class PermanentError(MarketPulseError):
    """Not retryable: the operation will fail again unless the input changes."""


class ConfigurationError(PermanentError):
    """Invalid or missing configuration."""


class ConnectionUnavailableError(TransientError):
    """A downstream dependency (DB, broker) could not be reached."""


class SchemaValidationError(PermanentError):
    """A message or record did not conform to its contract."""


class ArchivalVerificationError(TransientError):
    """A partition export's row count or checksum didn't verify against
    object storage after upload. Transient, not permanent — an upload can
    fail to be a bit-for-bit fresh export for reasons (network corruption,
    a concurrent write mid-export) that a retry can resolve. Raising this
    must always happen *before* the partition is dropped, never after.
    """
