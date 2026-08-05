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


class FeaturesSettings(BaseModel):
    """Feature-layer window state tuning.

    ``max_history_hours`` must exceed the widest feature lookback
    (currently 24h, ``realised_vol_24h``) with real margin — see
    ``features.windows.DEFAULT_MAX_HISTORY`` for why an exact match makes
    that window's coverage check impossible to satisfy.
    """

    max_history_hours: float = 27.0
    max_buffer_points: int = 10_000
    gap_threshold_seconds: float = 120.0


class MLflowSettings(BaseModel):
    """MLflow tracking + model registry connection settings.

    ``tracking_uri`` points at the tracking server (Postgres-backed, with a
    proxied S3/MinIO artifact store — see ``docker/docker-compose.yml``), never
    at a bare local ``mlruns/`` directory: an untracked-server run can't be
    resolved by ``registry_model_name``, and Phase 4/5 both depend on a
    reachable server (CLAUDE.md: "never train without MLflow reachable").
    """

    tracking_uri: str = "http://localhost:5000"
    registry_model_name: str = "marketpulse"
    experiment_name: str = "marketpulse"


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
    features: FeaturesSettings = Field(default_factory=FeaturesSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton, parsed once."""
    return Settings()  # type: ignore[call-arg]
