"""Dashboard panel data (Phase 6).

The requirement being tested is the unglamorous one: **a fresh install with
no data renders an explicit empty state, not a crash.** That is the failure
mode that makes a portfolio dashboard embarrassing, and it is checked here
per panel rather than by loading the Streamlit app.
"""

from datetime import UTC, datetime, timedelta

import pytest

from marketpulse.monitoring.dashboard import (
    DriftCell,
    DriftPanel,
    ModelPerformancePanel,
    SymbolFreshness,
    SystemHealthPanel,
    build_drift,
    build_system_health,
    stale_symbols,
)
from marketpulse.monitoring.drift import (
    SEVERITY_MODERATE,
    SEVERITY_SIGNIFICANT,
    SEVERITY_STABLE,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
STALE_AFTER = timedelta(seconds=120)


class FakeAlert:
    def __init__(self, severity: str) -> None:
        self.severity = severity
        self.first_breached_at = NOW


class FakeDriftRow:
    def __init__(
        self,
        feature_name: str,
        metric_name: str,
        value: float,
        severity: str,
        window_end: datetime,
    ) -> None:
        self.feature_name = feature_name
        self.metric_name = metric_name
        self.metric_value = value
        self.severity = severity
        self.window_end = window_end
        self.window_start = window_end - timedelta(hours=6)
        self.computed_at = window_end
        self.p_value = None
        self.reference_model_version = "3"


class FakeSession:
    """Session stand-in whose only job is to satisfy the repository calls the
    panel builders make. Those calls are monkeypatched per test.
    """


# --- empty states ---------------------------------------------------------


def test_system_health_renders_an_explicit_empty_state_with_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marketpulse.monitoring.dashboard as dashboard_module

    monkeypatch.setattr(dashboard_module, "latest_observed_at_per_symbol", lambda _s: {})
    monkeypatch.setattr(dashboard_module, "latest_feature_ts_per_symbol", lambda _s: {})
    monkeypatch.setattr(dashboard_module, "list_open_alerts", lambda _s: [])

    panel = build_system_health(FakeSession(), now=NOW, stale_after=STALE_AFTER)  # type: ignore[arg-type]

    assert panel.is_empty is True
    assert panel.empty_reason  # a stated reason, not just a blank panel
    assert panel.symbols == []


def test_drift_panel_empty_state_says_it_never_ran_not_that_it_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The distinction that matters: an empty drift table means the monitor
    # has not run, which is more worrying than a clean one, not less.
    import marketpulse.monitoring.dashboard as dashboard_module

    monkeypatch.setattr(dashboard_module, "list_drift_metrics", lambda _s, **_k: [])

    panel = build_drift(FakeSession(), now=NOW, window=timedelta(hours=24))  # type: ignore[arg-type]

    assert panel.is_empty is True
    assert "not run" in (panel.empty_reason or "")
    assert panel.worst_severity == SEVERITY_STABLE


def test_every_panel_dataclass_supports_an_empty_construction() -> None:
    # A panel that can only be built from data is a panel that crashes on a
    # fresh install.
    assert SystemHealthPanel([], None, 0, 0, True, "empty").is_empty
    assert ModelPerformancePanel([], [], None, 0, True, "empty").is_empty
    assert DriftPanel([], {}, [], SEVERITY_STABLE, True, "empty").is_empty


# --- system health --------------------------------------------------------


def test_system_health_flags_stale_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    import marketpulse.monitoring.dashboard as dashboard_module

    monkeypatch.setattr(
        dashboard_module,
        "latest_observed_at_per_symbol",
        lambda _s: {"BTC-USD": NOW - timedelta(seconds=5), "ETH-USD": NOW - timedelta(hours=3)},
    )
    monkeypatch.setattr(
        dashboard_module,
        "latest_feature_ts_per_symbol",
        lambda _s: {"BTC-USD": NOW - timedelta(seconds=5), "ETH-USD": NOW - timedelta(hours=3)},
    )
    monkeypatch.setattr(
        dashboard_module,
        "list_open_alerts",
        lambda _s: [FakeAlert("critical"), FakeAlert("warning")],
    )
    monkeypatch.setattr(dashboard_module, "latest_prediction_at", lambda _s: NOW)

    panel = build_system_health(FakeSession(), now=NOW, stale_after=STALE_AFTER)  # type: ignore[arg-type]

    assert panel.is_empty is False
    assert stale_symbols(panel) == ["ETH-USD"]
    assert panel.open_alert_count == 2
    assert panel.critical_alert_count == 1


def test_a_symbol_with_ticks_but_no_features_counts_as_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "Never produced a feature row" is not a healthier state than "produced
    # and went quiet".
    import marketpulse.monitoring.dashboard as dashboard_module

    monkeypatch.setattr(
        dashboard_module, "latest_observed_at_per_symbol", lambda _s: {"NEW-USD": NOW}
    )
    monkeypatch.setattr(dashboard_module, "latest_feature_ts_per_symbol", lambda _s: {})
    monkeypatch.setattr(dashboard_module, "list_open_alerts", lambda _s: [])
    monkeypatch.setattr(dashboard_module, "latest_prediction_at", lambda _s: None)

    panel = build_system_health(FakeSession(), now=NOW, stale_after=STALE_AFTER)  # type: ignore[arg-type]

    assert stale_symbols(panel) == ["NEW-USD"]
    assert panel.symbols[0].feature_age_seconds is None


def test_symbol_freshness_reports_ages_in_seconds() -> None:
    row = SymbolFreshness("BTC-USD", NOW, NOW, 0.0, 0.0, False)
    assert row.tick_age_seconds == 0.0
    assert row.is_stale is False


# --- drift panel ----------------------------------------------------------


def test_drift_panel_builds_a_heatmap_and_a_severity_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marketpulse.monitoring.dashboard as dashboard_module

    earlier = NOW - timedelta(hours=6)
    rows = [
        FakeDriftRow("ma_5m", "psi", 0.05, SEVERITY_STABLE, earlier),
        FakeDriftRow("rsi_15m", "psi", 0.15, SEVERITY_MODERATE, earlier),
        FakeDriftRow("ma_5m", "psi", 0.40, SEVERITY_SIGNIFICANT, NOW),
        FakeDriftRow("ma_5m", "ks", 0.30, SEVERITY_SIGNIFICANT, NOW),
    ]
    monkeypatch.setattr(dashboard_module, "list_drift_metrics", lambda _s, **_k: rows)

    panel = build_drift(FakeSession(), now=NOW, window=timedelta(hours=24))  # type: ignore[arg-type]

    assert panel.is_empty is False
    # Heatmap is PSI only -- one cell per (feature, window).
    assert len(panel.heatmap) == 3
    assert all(isinstance(cell, DriftCell) for cell in panel.heatmap)
    assert panel.ks_by_feature == {"ma_5m": 0.30}
    # The timeline takes the worst severity in each window.
    assert panel.severity_timeline == [
        (earlier, SEVERITY_MODERATE),
        (NOW, SEVERITY_SIGNIFICANT),
    ]
    assert panel.worst_severity == SEVERITY_SIGNIFICANT
