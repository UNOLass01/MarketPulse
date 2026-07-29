"""Shared fixtures for all test tiers (unit, contract, integration, e2e)."""

import pytest

from marketpulse.config import Settings, get_settings


@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the minimum required MP_ env vars so Settings() constructs cleanly."""
    values = {
        "MP_DB__USER": "marketpulse",
        "MP_DB__PASSWORD": "marketpulse",
        "MP_DB__NAME": "marketpulse",
        "MP_RABBITMQ__USER": "marketpulse",
        "MP_RABBITMQ__PASSWORD": "marketpulse",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def settings(env_vars: None) -> Settings:
    get_settings.cache_clear()
    return get_settings()
