"""Application configuration.

All settings are sourced from environment variables prefixed ``MP_`` (see
``.env.example``). Nested settings use a double underscore delimiter, e.g.
``MP_DB__HOST``. Validation happens at construction time so a bad or missing
value fails at startup, not on first use.
"""

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Postgres connection settings."""

    host: str = "localhost"
    port: int = 5432
    user: str
    password: str
    name: str

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RabbitMQSettings(BaseModel):
    """RabbitMQ connection settings."""

    host: str = "localhost"
    port: int = 5672
    user: str
    password: str
    vhost: str = "/"

    @property
    def url(self) -> str:
        return f"amqp://{self.user}:{self.password}@{self.host}:{self.port}{self.vhost}"


class Settings(BaseSettings):
    """Root application settings, assembled from ``MP_``-prefixed env vars."""

    model_config = SettingsConfigDict(
        env_prefix="MP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="local", pattern="^(local|test|ci|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    db: DatabaseSettings
    rabbitmq: RabbitMQSettings


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton, parsed once."""
    return Settings()  # type: ignore[call-arg]
