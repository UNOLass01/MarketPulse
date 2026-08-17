"""Phase 6 exit criterion, end to end against real Postgres.

    "Injecting synthetic drift into the feature stream produces a visible
    PSI breach and an alert."

Both halves are asserted: the ``drift_metrics`` rows (visible) and the
``alerts`` row (an alert), including the sustained-breach rule that makes the
first evaluation *not* fire and the dedup rule that keeps a persistent
condition to a single alert.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import MonitoringSettings
from marketpulse.contracts.features import FeatureVector
from marketpulse.features.registry import FEATURE_NAMES
from marketpulse.monitoring.alerts import (
    RULE_FEATURE_DRIFT,
    evaluate_and_record,
)
from marketpulse.monitoring.drift import (
    SEVERITY_SIGNIFICANT,
    SEVERITY_STABLE,
    run_drift_monitoring,
    worst_severity,
)
from marketpulse.storage.engine import session_scope
from marketpulse.storage.repositories.alerts import get_open_alert, list_open_alerts
from marketpulse.storage.repositories.drift import (
    latest_drift_computed_at,
    list_drift_metrics,
)
from marketpulse.storage.repositories.features import upsert_feature_vector
from marketpulse.storage.repositories.ml_registry import (
    record_model_version,
    record_training_run,
)

pytestmark = pytest.mark.integration

SYMBOL = "BTC-USD"
MODEL_NAME = "marketpulse"
NOW = datetime.now(UTC).replace(microsecond=0)

#: Reference distribution the "Production model" was trained on.
REFERENCE_MEAN = 100.0
REFERENCE_STD = 5.0

MONITORING = MonitoringSettings(
    drift_window_hours=6.0,
    drift_bins=10,
    drift_min_correlated_features=3,
    alert_sustained_evaluations=2,
    alert_suppression_minutes=360.0,
)


def _promote_reference_model(session: Session) -> None:
    """Register a Production model carrying a reference feature snapshot.

    Drift is only meaningful against the distribution the *live* model was
    trained on, so this is a prerequisite rather than test scaffolding.
    """
    stats = {
        name: {
            "mean": REFERENCE_MEAN,
            "std": REFERENCE_STD,
            "min": REFERENCE_MEAN - 4 * REFERENCE_STD,
            "max": REFERENCE_MEAN + 4 * REFERENCE_STD,
            "p50": REFERENCE_MEAN,
        }
        for name in FEATURE_NAMES
    }
    training_run_id = record_training_run(
        session,
        mlflow_run_id="run-abc",
        feature_set_version=1,
        config_version=1,
        horizon_minutes=15.0,
        theta=0.002,
        train_row_count=1000,
        validation_row_count=200,
        test_row_count=200,
        train_class_distribution={"DOWN": 300, "STABLE": 400, "UP": 300},
        window_start=NOW - timedelta(days=7),
        window_end=NOW - timedelta(days=1),
        candidate_metrics={"accuracy": 0.5},
        baseline_metrics={"majority": {"accuracy": 0.4}},
        incumbent_metrics=None,
        promoted=True,
        rejection_reason=None,
        git_sha=None,
        started_at=NOW - timedelta(days=1),
        finished_at=NOW - timedelta(days=1),
    )
    record_model_version(
        session,
        training_run_id=training_run_id,
        mlflow_model_name=MODEL_NAME,
        mlflow_model_version="3",
        stage="Production",
        reference_feature_stats=stats,
        promoted_at=NOW - timedelta(days=1),
    )


def _write_feature_window(
    session: Session, *, mean: float, count: int = 400, end: datetime | None = None
) -> None:
    """Write a live feature window ending at ``end`` (default ``NOW``).

    Every registered feature gets the same distribution so a drift injection
    moves all of them together — which is the *correlated* multi-feature
    condition the alert rule requires, not a single noisy feature.
    """
    window_end = end or NOW
    rng = np.random.default_rng(11)
    for index in range(count):
        feature_ts = window_end - timedelta(seconds=(count - index) * 10)
        values: dict[str, float | None] = {
            name: float(rng.normal(mean, REFERENCE_STD)) for name in FEATURE_NAMES
        }
        upsert_feature_vector(
            session,
            FeatureVector(
                symbol=SYMBOL,
                feature_ts=feature_ts,
                feature_set_version=1,
                feature_values=values,
                insufficient_history=False,
                has_gap=False,
            ),
        )


# --- drift computation ----------------------------------------------------


def test_a_quiet_window_produces_stable_metrics_and_no_alert(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _promote_reference_model(session)
        _write_feature_window(session, mean=REFERENCE_MEAN)

    results = run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)

    assert results, "drift should have evaluated every feature"
    assert worst_severity(results) == SEVERITY_STABLE

    evaluate_and_record(session_factory, MONITORING, drift_results=results, now=NOW)

    with session_factory() as session:
        assert list_open_alerts(session) == []


def test_injected_synthetic_drift_produces_a_visible_psi_breach(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _promote_reference_model(session)
        # The injection: a 4-sigma shift across every feature at once.
        _write_feature_window(session, mean=REFERENCE_MEAN + 4 * REFERENCE_STD)

    results = run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)

    assert worst_severity(results) == SEVERITY_SIGNIFICANT

    with session_factory() as session:
        rows = list_drift_metrics(session, since=NOW - timedelta(hours=1))
        computed_at = latest_drift_computed_at(session)

    # Visible: persisted, long-not-wide, tied to the model it was compared to.
    assert computed_at is not None
    psi_rows = [row for row in rows if row.metric_name == "psi"]
    assert {row.feature_name for row in psi_rows} == set(FEATURE_NAMES)
    assert all(row.reference_model_version == "3" for row in psi_rows)
    assert all(row.severity == SEVERITY_SIGNIFICANT for row in psi_rows)
    assert all(row.metric_value > MONITORING.psi_significant_threshold for row in psi_rows)
    # KS rows carry a p-value; PSI rows do not.
    assert all(row.p_value is None for row in psi_rows)
    assert all(row.p_value is not None for row in rows if row.metric_name == "ks")


def test_rerunning_drift_over_the_same_window_updates_rather_than_duplicates(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _promote_reference_model(session)
        _write_feature_window(session, mean=REFERENCE_MEAN)

    run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)
    run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)

    with session_factory() as session:
        rows = list_drift_metrics(session, since=NOW - timedelta(hours=1))

    # Two metrics per feature, once -- not twice.
    assert len(rows) == len(FEATURE_NAMES) * 2


def test_drift_without_a_production_reference_writes_nothing(
    session_factory: sessionmaker[Session],
) -> None:
    # An empty drift_metrics table is what makes the API report "never
    # evaluated" instead of "no breaches". Absence of a signal is never a
    # passing signal.
    with session_scope(session_factory) as session:
        _write_feature_window(session, mean=REFERENCE_MEAN)

    assert run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW) == []

    with session_factory() as session:
        assert latest_drift_computed_at(session) is None


# --- alerting -------------------------------------------------------------


def test_sustained_drift_fires_exactly_one_alert(
    session_factory: sessionmaker[Session],
) -> None:
    """The exit criterion's second half, plus both anti-fatigue rules."""
    with session_scope(session_factory) as session:
        _promote_reference_model(session)
        _write_feature_window(session, mean=REFERENCE_MEAN + 4 * REFERENCE_STD)

    results = run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)

    # Evaluation 1: a spike. Recorded, but nobody is woken up.
    evaluate_and_record(session_factory, MONITORING, drift_results=results, now=NOW)
    with session_factory() as session:
        alert = get_open_alert(session, RULE_FEATURE_DRIFT.name)
    assert alert is not None
    assert alert.fired_at is None
    assert alert.consecutive_breaches == 1

    # Evaluation 2: sustained. Now it fires.
    second = NOW + timedelta(hours=6)
    evaluate_and_record(session_factory, MONITORING, drift_results=results, now=second)
    with session_factory() as session:
        alert = get_open_alert(session, RULE_FEATURE_DRIFT.name)
    assert alert is not None
    assert alert.fired_at is not None
    assert alert.consecutive_breaches == 2
    assert alert.severity == RULE_FEATURE_DRIFT.severity
    # Every alert names its runbook.
    assert alert.runbook == RULE_FEATURE_DRIFT.runbook
    assert len(alert.details["drifted_features"]) >= MONITORING.drift_min_correlated_features

    # Evaluation 3: still broken, inside the suppression window. One alert.
    evaluate_and_record(
        session_factory, MONITORING, drift_results=results, now=second + timedelta(hours=1)
    )
    with session_factory() as session:
        assert len(list_open_alerts(session)) == 1


def test_a_cleared_condition_resolves_the_open_alert(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as session:
        _promote_reference_model(session)
        _write_feature_window(session, mean=REFERENCE_MEAN + 4 * REFERENCE_STD)

    drifted = run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)
    evaluate_and_record(session_factory, MONITORING, drift_results=drifted, now=NOW)
    evaluate_and_record(
        session_factory, MONITORING, drift_results=drifted, now=NOW + timedelta(hours=6)
    )

    with session_factory() as session:
        assert len(list_open_alerts(session)) == 1

    # The pipeline is fixed: the live window matches the reference again.
    # Written to end at the later evaluation time, so it actually falls
    # inside that run's drift window rather than behind it.
    later = NOW + timedelta(hours=12)
    with session_scope(session_factory) as session:
        session.execute(text("TRUNCATE TABLE features CASCADE"))
    with session_scope(session_factory) as session:
        _write_feature_window(session, mean=REFERENCE_MEAN, end=later)

    recovered = run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=later)
    assert recovered, "the recovery window must actually contain feature rows"
    assert worst_severity(recovered) == SEVERITY_STABLE

    evaluate_and_record(session_factory, MONITORING, drift_results=recovered, now=later)

    with session_factory() as session:
        assert list_open_alerts(session) == []


def test_a_drift_run_with_no_live_data_does_not_resolve_an_open_alert(
    session_factory: sessionmaker[Session],
) -> None:
    """Absence of a signal is never treated as a passing signal.

    If the feature pipeline dies, drift monitoring produces *no* results —
    and that must not read as "drift cleared". An open alert stays open until
    something actually demonstrates recovery.
    """
    with session_scope(session_factory) as session:
        _promote_reference_model(session)
        _write_feature_window(session, mean=REFERENCE_MEAN + 4 * REFERENCE_STD)

    drifted = run_drift_monitoring(session_factory, MONITORING, MODEL_NAME, now=NOW)
    evaluate_and_record(session_factory, MONITORING, drift_results=drifted, now=NOW)
    evaluate_and_record(
        session_factory, MONITORING, drift_results=drifted, now=NOW + timedelta(hours=6)
    )
    with session_factory() as session:
        assert len(list_open_alerts(session)) == 1

    # A much later window with no feature rows in it at all.
    silent = run_drift_monitoring(
        session_factory, MONITORING, MODEL_NAME, now=NOW + timedelta(days=2)
    )
    assert silent == []

    evaluate_and_record(
        session_factory, MONITORING, drift_results=silent, now=NOW + timedelta(days=2)
    )

    with session_factory() as session:
        assert len(list_open_alerts(session)) == 1
