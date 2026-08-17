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


class ServingSettings(BaseModel):
    """Phase 5 API behaviour.

    ``max_feature_age_seconds`` is the staleness guard: a prediction served
    off features older than this is refused with 503 rather than returned,
    because a caller cannot tell a stale prediction from a fresh one (see
    ``docs/plan/phase-5-api.md``). It should comfortably exceed the normal
    tick cadence — the default is 12x ``MonitoringSettings``'s expected 10s
    interval, so ordinary jitter never trips it.

    ``model_refresh_seconds`` is how often the background task re-checks the
    registry for a newly promoted Production version. It bounds how long the
    exit criterion ("promotion changes model_version with no redeploy")
    takes to become visible, nothing more — a failed refresh keeps serving
    the model already in memory.
    """

    max_feature_age_seconds: float = Field(default=120.0, gt=0)
    model_refresh_seconds: float = Field(default=60.0, gt=0)
    default_page_size: int = Field(default=100, gt=0)
    # Hard ceiling: a caller asking for more gets 422, never a full-table
    # scan dressed up as pagination.
    max_page_size: int = Field(default=500, gt=0)
    max_history_days: float = Field(default=30.0, gt=0)


class MonitoringSettings(BaseModel):
    """Thresholds for ``monitoring.quality``'s data-quality checks (Phase 4)
    plus ``monitoring.drift`` / ``monitoring.performance`` / ``monitoring.alerts``
    (Phase 6).

    The Phase 4 checks are deliberately simple sanity checks; the Phase 6
    fields below are the rigorous distribution-shift statistics, compared
    against the reference snapshot tied to the Production model
    (``model_versions.reference_feature_stats``), not against yesterday's
    live data.
    """

    freshness_max_lag_minutes: float = 10.0
    expected_tick_interval_seconds: float = 10.0
    completeness_window_hours: float = 1.0
    completeness_min_ratio: float = Field(default=0.95, gt=0, le=1)
    distribution_window_hours: float = 1.0
    distribution_reference_window_hours: float = 24.0
    distribution_max_relative_shift: float = Field(default=0.5, gt=0)

    # --- Phase 6: drift ---------------------------------------------------
    #: Conventional PSI severity cuts (<0.1 stable, 0.1-0.25 moderate,
    #: >0.25 significant). These are industry convention, not law — they are
    #: settings precisely so a symbol whose features are genuinely noisier
    #: can be tuned without editing ``monitoring.drift``.
    psi_moderate_threshold: float = Field(default=0.10, gt=0)
    psi_significant_threshold: float = Field(default=0.25, gt=0)
    drift_window_hours: float = Field(default=6.0, gt=0)
    drift_bins: int = Field(default=10, gt=1)
    #: A drift alert requires this many features breaching at once. Single-
    #: feature drift is usually noise; alerting on it produces the fatigue
    #: that gets real signals ignored (phase-6 plan).
    drift_min_correlated_features: int = Field(default=3, gt=0)

    # --- Phase 6: performance attribution ---------------------------------
    performance_window_hours: float = Field(default=24.0, gt=0)
    #: Minimum resolved predictions before rolling accuracy is trusted
    #: enough to alert on. Below this, metrics are still computed and stored
    #: (they are the record), just never used to fire.
    performance_min_resolved: int = Field(default=50, gt=0)
    performance_min_accuracy: float = Field(default=0.34, ge=0, le=1)
    #: Max total-variation distance between the live predicted-class mix and
    #: the training prior before it counts as a breach. Fires long before
    #: accuracy can, and usually means broken features, not a regime change.
    prediction_distribution_max_shift: float = Field(default=0.30, gt=0, le=1)

    # --- Phase 6: alerting ------------------------------------------------
    #: N consecutive breaching evaluations before an alert fires. 1 would
    #: make every instantaneous spike an alert.
    alert_sustained_evaluations: int = Field(default=2, gt=0)
    #: Re-firing window: the same dedup key seen again inside this many
    #: minutes updates the open alert instead of raising a second one.
    alert_suppression_minutes: float = Field(default=360.0, gt=0)


class ObjectStorageSettings(BaseModel):
    """S3-compatible object storage the app talks to directly (archival
    exports). Distinct from MLflow's artifact store: MLflow's tracking
    server proxies artifact reads/writes so the app never needs S3
    credentials for that path (see ``MLflowSettings``) — this is only for
    ``storage.archival``, which writes Parquet exports straight to the
    bucket itself.
    """

    endpoint_url: str = "http://localhost:9000"
    access_key: str = "marketpulse"
    secret_key: str = "marketpulse-minio"
    archive_bucket: str = "marketpulse-archive"
    # dag_data_archival: a partition older than this many months is
    # eligible for export + drop. See storage.archival.archivable_partitions.
    hot_retention_months: int = Field(default=6, gt=0)


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
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    object_store: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton, parsed once."""
    return Settings()  # type: ignore[call-arg]
