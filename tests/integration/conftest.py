"""Integration fixtures: real Postgres + RabbitMQ via testcontainers.

Schema is created directly from ``storage.models.Base`` (the same source of
truth the Alembic migration is generated from) rather than by shelling out to
Alembic — the migration DDL itself is exercised separately against the dev
compose stack.
"""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from marketpulse.config import DatabaseSettings, RabbitMQSettings
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.topology import QUEUE_DEAD, QUEUE_PERSIST, QUEUE_RETRY
from marketpulse.storage.engine import make_engine, make_session_factory
from marketpulse.storage.models import Base
from marketpulse.storage.partitions import ensure_partitions_covering

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_container() -> Iterator[RabbitMqContainer]:
    with RabbitMqContainer("rabbitmq:3.13-management-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def db_settings(postgres_container: PostgresContainer) -> DatabaseSettings:
    return DatabaseSettings(
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(postgres_container.port)),
        user=postgres_container.username,
        password=postgres_container.password,
        name=postgres_container.dbname,
    )


@pytest.fixture(scope="session")
def rabbitmq_settings(rabbitmq_container: RabbitMqContainer) -> RabbitMQSettings:
    return RabbitMQSettings(
        host=rabbitmq_container.get_container_host_ip(),
        port=int(rabbitmq_container.get_exposed_port(rabbitmq_container.port)),
        user=rabbitmq_container.username,
        password=rabbitmq_container.password,
    )


@pytest.fixture(scope="session")
def engine(db_settings: DatabaseSettings) -> Iterator[Engine]:
    eng = make_engine(db_settings)
    Base.metadata.create_all(eng)
    with eng.begin() as connection:
        today = datetime.now(UTC).date()
        ensure_partitions_covering(connection, "raw_ticks", today, months_ahead=1)
        ensure_partitions_covering(connection, "features", today, months_ahead=1)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


@pytest.fixture(autouse=True)
def _clean_database(engine: Engine) -> Iterator[None]:
    yield
    with engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE TABLE raw_ticks, features, symbols, "
                "training_runs, model_versions RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture
def connection_manager(rabbitmq_settings: RabbitMQSettings) -> Iterator[ConnectionManager]:
    manager = ConnectionManager(rabbitmq_settings.url)
    manager.channel()  # eagerly connect + declare topology
    yield manager
    manager.close()


@pytest.fixture(autouse=True)
def _clean_queues(connection_manager: ConnectionManager) -> Iterator[None]:
    yield
    channel = connection_manager.channel()
    for queue in (QUEUE_PERSIST, QUEUE_RETRY, QUEUE_DEAD):
        channel.queue_purge(queue)
