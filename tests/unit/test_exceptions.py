import pytest

from marketpulse.exceptions import (
    ConfigurationError,
    ConnectionUnavailableError,
    MarketPulseError,
    PermanentError,
    SchemaValidationError,
    TransientError,
)

pytestmark = pytest.mark.unit


def test_exception_hierarchy() -> None:
    assert issubclass(TransientError, MarketPulseError)
    assert issubclass(PermanentError, MarketPulseError)
    assert not issubclass(TransientError, PermanentError)
    assert not issubclass(PermanentError, TransientError)

    assert isinstance(ConnectionUnavailableError(), TransientError)
    assert isinstance(ConfigurationError(), PermanentError)
    assert isinstance(SchemaValidationError(), PermanentError)

    assert not isinstance(ConnectionUnavailableError(), PermanentError)
    assert not isinstance(ConfigurationError(), TransientError)
