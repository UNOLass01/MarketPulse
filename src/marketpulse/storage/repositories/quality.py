"""Persistence for ``monitoring.quality`` check results, plus the read the
retraining gate uses.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketpulse.storage.models import QualityCheck

#: Every check ``dag_data_quality`` runs -- the gate requires a passing,
#: recent row for each of these, not just "no failures seen" (a check that
#: never ran must not silently read as passing).
ALL_CHECK_NAMES = ("freshness", "completeness", "validity", "distribution")


def record_quality_check(
    session: Session,
    *,
    check_name: str,
    symbol: str | None,
    passed: bool,
    details: dict[str, object],
    window_start: datetime,
    window_end: datetime,
) -> int:
    """Insert one check result row. Never upserted -- every run is a new,
    independently-auditable row, mirroring ``training_runs``' "record
    rejections too" pattern.
    """
    check = QualityCheck(
        check_name=check_name,
        symbol=symbol,
        passed=passed,
        details=details,
        window_start=window_start,
        window_end=window_end,
    )
    session.add(check)
    session.flush()
    return check.id


def latest_checks_passed(session: Session, *, since: datetime) -> bool:
    """True iff, for every check in :data:`ALL_CHECK_NAMES`, the most recent
    run at or after ``since`` exists and passed. A missing or stale check
    fails closed -- ``dag_model_retraining``'s gate must never promote on
    the *absence* of a bad signal.
    """
    for check_name in ALL_CHECK_NAMES:
        stmt = (
            select(QualityCheck.passed)
            .where(QualityCheck.check_name == check_name, QualityCheck.checked_at >= since)
            .order_by(QualityCheck.checked_at.desc())
            .limit(1)
        )
        result = session.execute(stmt).scalar_one_or_none()
        if result is not True:
            return False
    return True
