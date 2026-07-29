import json

import pytest

from marketpulse.logging import get_logger, set_correlation_id

pytestmark = pytest.mark.unit


def test_logger_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    set_correlation_id("test-correlation-id")
    logger = get_logger("marketpulse.test", level="DEBUG")

    logger.info("hello world")

    out = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(out)

    assert record["message"] == "hello world"
    assert record["level"] == "INFO"
    assert record["correlation_id"] == "test-correlation-id"


def test_logger_reuses_handlers() -> None:
    first = get_logger("marketpulse.reuse")
    second = get_logger("marketpulse.reuse")

    assert first is second
    assert len(second.handlers) == 1
