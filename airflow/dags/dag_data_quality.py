"""dag_data_quality: hourly data-quality checks (Phase 4).

Freshness, completeness, validity, and distribution sanity for every known
symbol — see ``marketpulse.monitoring.quality`` for the checks themselves
and why they're split this way; freshness in particular is "the
highest-value task in the whole portfolio" (phase-4 plan) since it catches
a silently-dead producer. All business logic lives there; this file only
wires it into a schedule and branches to an alert path on failure
(CLAUDE.md rule #9).
"""

from datetime import datetime, timedelta

from _dag_common import DEFAULT_ARGS, logger, mp_session_factory, mp_settings
from airflow.sdk import DAG, task

from marketpulse.monitoring.quality import run_data_quality_checks


@task(execution_timeout=timedelta(minutes=10))
def run_checks() -> list[dict[str, object]]:
    settings = mp_settings()
    results = run_data_quality_checks(mp_session_factory(settings), settings.monitoring)
    return [{"check_name": r.check_name, "symbol": r.symbol, "passed": r.passed} for r in results]


@task.branch(execution_timeout=timedelta(minutes=1))
def branch_on_failure(results: list[dict[str, object]]) -> str:
    return "alert_on_failure" if any(not r["passed"] for r in results) else "checks_passed"


@task(execution_timeout=timedelta(minutes=1))
def checks_passed() -> None:
    logger.info("data quality checks passed")


@task(execution_timeout=timedelta(minutes=1))
def alert_on_failure(results: list[dict[str, object]]) -> None:
    failed = [r for r in results if not r["passed"]]
    logger.error("data quality check(s) failed", extra={"extra_fields": {"failed_checks": failed}})


with DAG(
    dag_id="dag_data_quality",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["quality", "phase-4"],
    doc_md=__doc__,
) as dag:
    check_results = run_checks()
    branch = branch_on_failure(check_results)
    passed = checks_passed()
    failed = alert_on_failure(check_results)
    branch >> [passed, failed]
