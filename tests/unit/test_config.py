import pytest
from pydantic import ValidationError

from marketpulse.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_config_loads_from_env(env_vars: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MP_ENV", "test")
    monkeypatch.setenv("MP_DB__HOST", "db.internal")
    monkeypatch.setenv("MP_RABBITMQ__PORT", "5673")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.env == "test"
    assert settings.db.host == "db.internal"
    assert settings.db.user == "marketpulse"
    assert settings.rabbitmq.port == 5673


def test_config_rejects_invalid(env_vars: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MP_ENV", "not-a-real-environment")

    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_config_is_cached(env_vars: None) -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
