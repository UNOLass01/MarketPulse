"""Shared plumbing for Phase 4 DAGs: default task args plus marketpulse
settings/engine/session-factory helpers.

Not business logic (CLAUDE.md rule #9) — just avoids repeating the same
five lines of wiring in every DAG file. Airflow adds a DAG file's own
directory to ``sys.path`` while parsing it, so sibling DAG files can import
this as a plain top-level module (``from _dag_common import ...``); the
leading underscore keeps it out of anything that might otherwise scan
``airflow/dags/`` looking for DAG-defining files.
"""

from datetime import timedelta
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import Settings, get_settings
from marketpulse.logging import get_logger
from marketpulse.storage.engine import make_engine, make_session_factory

logger = get_logger(__name__)


def mp_settings() -> Settings:
    return get_settings()


def mp_engine(settings: Settings) -> Engine:
    return make_engine(settings.db)


def mp_session_factory(settings: Settings) -> sessionmaker[Session]:
    return make_session_factory(mp_engine(settings))


def log_task_failure(context: dict[str, Any]) -> None:
    """Default ``on_failure_callback`` — a structured log entry, not a new
    alerting channel (CLAUDE.md: never add a new service without asking).
    Airflow's own UI/retry history is the primary place to look; this makes
    a failure greppable in the same JSON log stream as every other
    marketpulse process.
    """
    task_instance = context.get("task_instance")
    dag = context.get("dag")
    logger.error(
        "task failed",
        extra={
            "extra_fields": {
                "dag_id": dag.dag_id if dag is not None else None,
                "task_id": getattr(task_instance, "task_id", None),
                "try_number": getattr(task_instance, "try_number", None),
            }
        },
    )


#: Applied to every task via ``default_args=DEFAULT_ARGS`` on each DAG.
#: retries=2 with exponential backoff, an explicit execution_timeout every
#: task inherits (individual tasks may override it for longer-running work),
#: and on_failure_callback -- the phase-4 plan's cross-cutting task rules.
DEFAULT_ARGS: dict[str, Any] = {
    "owner": "marketpulse",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": log_task_failure,
}
