"""Prometheus text exposition (Phase 5's ``/metrics``)."""

import pytest

from marketpulse.monitoring.metrics import (
    MetricFamily,
    MetricsRegistry,
    Sample,
    render,
)

pytestmark = pytest.mark.unit


def test_render_emits_help_type_and_samples() -> None:
    text = render(
        [
            MetricFamily(
                name="requests_total",
                metric_type="counter",
                help_text="Total requests.",
                samples=[Sample({"path": "/health"}, 3.0)],
            )
        ]
    )
    assert text.splitlines() == [
        "# HELP requests_total Total requests.",
        "# TYPE requests_total counter",
        'requests_total{path="/health"} 3.0',
    ]


def test_a_family_with_no_samples_still_declares_itself() -> None:
    # A metric that vanishes from a scrape is indistinguishable from a failed
    # scrape; a declared metric with no series is unambiguously "nothing has
    # happened yet".
    text = render([MetricFamily("x_total", "counter", "X.", [])])
    assert "# TYPE x_total counter" in text
    assert "x_total{" not in text


def test_labels_render_in_sorted_order_so_scrapes_are_comparable() -> None:
    text = render(
        [
            MetricFamily(
                "m",
                "gauge",
                "",
                [Sample({"z": "1", "a": "2", "m": "3"}, 1.0)],
            )
        ]
    )
    assert 'm{a="2",m="3",z="1"} 1.0' in text


def test_label_values_are_escaped() -> None:
    text = render([MetricFamily("m", "gauge", "", [Sample({"p": 'a"b\\c\nd'}, 1.0)])])
    assert 'p="a\\"b\\\\c\\nd"' in text


def test_unlabelled_samples_render_without_braces() -> None:
    text = render([MetricFamily("m", "gauge", "", [Sample({}, 7.0)])])
    assert "m 7.0" in text


def test_counters_accumulate_and_are_not_reset_by_reading() -> None:
    # Resetting on read would make two scrapers silently halve each other's
    # numbers.
    registry = MetricsRegistry()
    registry.increment("hits_total", labels={"path": "/a"})
    registry.increment("hits_total", labels={"path": "/a"})

    assert 'hits_total{path="/a"} 2.0' in registry.render()
    assert 'hits_total{path="/a"} 2.0' in registry.render()

    registry.increment("hits_total", labels={"path": "/a"})
    assert 'hits_total{path="/a"} 3.0' in registry.render()


def test_gauges_replace_rather_than_accumulate() -> None:
    registry = MetricsRegistry()
    registry.set_gauge("temp", 10.0)
    registry.set_gauge("temp", 4.0)
    assert "temp 4.0" in registry.render()


def test_distinct_label_sets_are_distinct_series() -> None:
    registry = MetricsRegistry()
    registry.increment("hits_total", labels={"path": "/a"})
    registry.increment("hits_total", labels={"path": "/b"})

    text = registry.render()
    assert 'hits_total{path="/a"} 1.0' in text
    assert 'hits_total{path="/b"} 1.0' in text
    # One HELP/TYPE header for the family, not one per series.
    assert text.count("# TYPE hits_total counter") == 1


def test_help_text_is_kept_from_the_first_registration() -> None:
    registry = MetricsRegistry()
    registry.increment("x_total", help_text="The real description.")
    registry.increment("x_total", help_text="")
    assert "# HELP x_total The real description." in registry.render()


def test_render_output_is_stable_across_calls() -> None:
    registry = MetricsRegistry()
    for path in ("/z", "/a", "/m"):
        registry.increment("hits_total", labels={"path": path})
    assert registry.render() == registry.render()


def test_empty_registry_renders_an_empty_document() -> None:
    assert MetricsRegistry().render() == "\n"
