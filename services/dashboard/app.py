"""Streamlit dashboard: rendering shell.

Run with ``streamlit run services/dashboard/app.py``.

Every value shown here comes from ``marketpulse.monitoring.dashboard``, which
does read-only aggregation over metrics Airflow already computed. This file
draws; it does not calculate, and it never writes. Point it at read-only
database credentials (``MP_DB__USER``) — nothing here needs more, and a
dashboard with write access is an outage waiting for a bad query.

Queries are cached (``st.cache_data``) with a bounded TTL so a page left open
on a wall display does not re-query Postgres on every rerender.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import get_settings
from marketpulse.monitoring.dashboard import Dashboard, build_dashboard
from marketpulse.storage.engine import make_engine, make_session_factory

REFRESH_TTL_SECONDS = 60


@st.cache_resource
def _session_factory() -> sessionmaker[Session]:
    """One small pool for the whole app. Cached as a *resource*, not data:
    Streamlit reruns the script top-to-bottom on every interaction, and a
    fresh engine per rerun would leak connections until Postgres refused
    new ones.
    """
    settings = get_settings()
    return make_session_factory(make_engine(settings.db, pool_size=2, max_overflow=2))


@st.cache_data(ttl=REFRESH_TTL_SECONDS)
def _load(window_hours: int) -> Dashboard:
    return build_dashboard(_session_factory(), get_settings(), window=timedelta(hours=window_hours))


def _empty(reason: str) -> None:
    """The explicit empty state. Never a blank panel and never a traceback —
    on a fresh install with nothing ingested, every panel should say what is
    missing.
    """
    st.info(f"No data to show — {reason}.")


def render_system_health(dashboard: Dashboard) -> None:
    st.subheader("1 · System health")
    panel = dashboard.system_health
    if panel.is_empty:
        _empty(panel.empty_reason or "nothing ingested yet")
        return

    left, middle, right = st.columns(3)
    stale = [row for row in panel.symbols if row.is_stale]
    left.metric("Symbols", len(panel.symbols), delta=f"-{len(stale)} stale" if stale else None)
    middle.metric("Open alerts", panel.open_alert_count)
    right.metric("Critical alerts", panel.critical_alert_count)

    st.caption(
        "Last prediction served: "
        + (panel.last_prediction_at.isoformat() if panel.last_prediction_at else "never")
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "last tick": row.last_tick_at,
                    "last feature": row.last_feature_ts,
                    "feature age (s)": row.feature_age_seconds,
                    "stale": row.is_stale,
                }
                for row in panel.symbols
            ]
        ),
        use_container_width=True,
    )


def render_data_pipeline(dashboard: Dashboard) -> None:
    st.subheader("2 · Data pipeline")
    panel = dashboard.data_pipeline
    if panel.is_empty:
        _empty(panel.empty_reason or "no ticks in the window")
        return

    if panel.ingestion_rate:
        st.caption("Ingestion rate (ticks per hour)")
        st.bar_chart(
            pd.DataFrame(panel.ingestion_rate, columns=["bucket", "ticks"]).set_index("bucket")
        )

    left, right = st.columns(2)
    with left:
        st.caption("Completeness by symbol")
        st.dataframe(
            pd.DataFrame(
                sorted(panel.completeness_by_symbol.items()), columns=["symbol", "completeness"]
            ),
            use_container_width=True,
        )
    with right:
        st.caption("Null rate by feature")
        st.dataframe(
            pd.DataFrame(
                sorted(panel.null_rate_by_feature.items()), columns=["feature", "null rate"]
            ),
            use_container_width=True,
        )

    if panel.quality_history:
        st.caption("Quality check history")
        st.dataframe(
            pd.DataFrame(panel.quality_history, columns=["checked at", "check", "passed"]),
            use_container_width=True,
        )


def render_model_performance(dashboard: Dashboard) -> None:
    st.subheader("3 · Model performance")
    panel = dashboard.model_performance
    if panel.is_empty:
        _empty(panel.empty_reason or "no resolved outcomes yet")
        return

    st.caption(
        f"{panel.pending_count} prediction(s) still inside the horizon — "
        "unresolved, not incorrect."
    )

    frame = pd.DataFrame(
        [
            {
                "model version": s.model_version,
                "resolved": s.resolved_count,
                "accuracy": s.accuracy,
                "macro F1": s.macro_f1,
            }
            for s in panel.slices
        ]
    ).set_index("model version")
    st.bar_chart(frame[["accuracy"]])

    # The baseline as a horizontal reference line, and promotions as
    # vertical annotations: an accuracy step at a promotion boundary is the
    # single most persuasive thing this project can show, and it is only
    # legible with both drawn in.
    if panel.baseline_accuracy is not None:
        st.caption(f"Baseline accuracy reference: {panel.baseline_accuracy:.3f}")
    if panel.promotions:
        st.caption("Promotion events")
        st.dataframe(
            pd.DataFrame(
                [(p.promoted_at, p.model_version) for p in panel.promotions],
                columns=["promoted at", "version"],
            ),
            use_container_width=True,
        )

    st.dataframe(frame, use_container_width=True)

    for slice_ in panel.slices:
        with st.expander(f"Confusion matrix — version {slice_.model_version}"):
            st.dataframe(pd.DataFrame(slice_.confusion_matrix).T, use_container_width=True)
            st.caption("Rows: actual · Columns: predicted")
            st.dataframe(
                pd.DataFrame(sorted(slice_.per_class_f1.items()), columns=["class", "F1"]),
                use_container_width=True,
            )


def render_drift(dashboard: Dashboard) -> None:
    st.subheader("4 · Drift")
    panel = dashboard.drift
    if panel.is_empty:
        # Explicitly *not* "no drift detected". An empty drift table means
        # the monitor has not run.
        _empty(panel.empty_reason or "drift monitoring has not run")
        return

    st.metric("Worst severity in window", panel.worst_severity)

    heat = pd.DataFrame(
        [(c.window_end, c.feature_name, c.psi) for c in panel.heatmap],
        columns=["window", "feature", "psi"],
    )
    if not heat.empty:
        st.caption("PSI heatmap (feature × window)")
        st.dataframe(
            heat.pivot_table(index="feature", columns="window", values="psi"),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.caption("KS statistic by feature")
        st.dataframe(
            pd.DataFrame(sorted(panel.ks_by_feature.items()), columns=["feature", "KS"]),
            use_container_width=True,
        )
    with right:
        st.caption("Severity timeline")
        st.dataframe(
            pd.DataFrame(panel.severity_timeline, columns=["window", "severity"]),
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="MarketPulse", layout="wide")
    st.title("MarketPulse — monitoring")

    window_hours = st.sidebar.selectbox(
        "Window", options=[6, 24, 72, 168], index=1, format_func=lambda h: f"last {h}h"
    )
    st.sidebar.caption(
        f"Read-only. Metrics are precomputed by Airflow; cached for {REFRESH_TTL_SECONDS}s."
    )
    if st.sidebar.button("Refresh now"):
        _load.clear()

    dashboard = _load(int(window_hours))
    for warning in dashboard.warnings:
        st.warning(warning)
    st.caption(f"Generated at {dashboard.generated_at.isoformat()} (UTC now: {datetime.now(UTC)})")

    render_system_health(dashboard)
    render_data_pipeline(dashboard)
    render_model_performance(dashboard)
    render_drift(dashboard)


if __name__ == "__main__":
    main()
