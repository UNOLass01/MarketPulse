"""Persistence for ``alerts`` (Phase 6).

The *decision* about whether a breach should fire lives in
``monitoring.alerts`` as a pure function; this module only loads the current
state for a dedup key and writes back whatever that function decided. Keeping
them apart is what lets the suppression and sustained-breach rules — the two
easiest things in a monitoring system to get subtly wrong — be tested with no
database at all.
"""

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketpulse.storage.models import Alert

STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"


def get_open_alert(session: Session, dedup_key: str) -> Alert | None:
    """The currently-open alert for ``dedup_key``, if any.

    At most one can exist: every write path either updates this row or
    resolves it, never inserts a second one alongside.
    """
    stmt = (
        select(Alert)
        .where(Alert.dedup_key == dedup_key, Alert.status == STATUS_OPEN)
        .order_by(Alert.first_breached_at.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def open_alert(
    session: Session,
    *,
    alert_name: str,
    dedup_key: str,
    severity: str,
    runbook: str,
    details: Mapping[str, object],
    consecutive_breaches: int,
    first_breached_at: datetime,
    fired_at: datetime | None,
    updated_at: datetime,
) -> int:
    """Insert a new open alert. ``runbook`` is required by the schema, so
    there is no way to reach this function without one.
    """
    alert = Alert(
        alert_name=alert_name,
        dedup_key=dedup_key,
        severity=severity,
        status=STATUS_OPEN,
        runbook=runbook,
        details=dict(details),
        consecutive_breaches=consecutive_breaches,
        first_breached_at=first_breached_at,
        fired_at=fired_at,
        updated_at=updated_at,
    )
    session.add(alert)
    session.flush()
    return alert.id


def update_alert(
    session: Session,
    alert: Alert,
    *,
    details: Mapping[str, object],
    consecutive_breaches: int,
    fired_at: datetime | None,
    updated_at: datetime,
) -> None:
    """Advance an existing open alert in place — this is the suppression
    path. The same condition breaching again bumps the counter and refreshes
    the detail payload; it does not create a second alert.
    """
    alert.details = dict(details)
    alert.consecutive_breaches = consecutive_breaches
    if fired_at is not None and alert.fired_at is None:
        alert.fired_at = fired_at
    alert.updated_at = updated_at


def resolve_alert(session: Session, alert: Alert, *, resolved_at: datetime) -> None:
    alert.status = STATUS_RESOLVED
    alert.resolved_at = resolved_at
    alert.updated_at = resolved_at


def list_open_alerts(session: Session) -> list[Alert]:
    stmt = select(Alert).where(Alert.status == STATUS_OPEN).order_by(Alert.first_breached_at.desc())
    return list(session.execute(stmt).scalars())


def list_alerts_since(session: Session, *, since: datetime) -> list[Alert]:
    stmt = select(Alert).where(Alert.updated_at >= since).order_by(Alert.updated_at.desc())
    return list(session.execute(stmt).scalars())
