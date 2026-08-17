"""Prometheus text exposition format, rendered by hand.

Deliberately not ``prometheus_client``. The API exposes a handful of
counters and gauges it already tracks in memory; pulling in a library with
its own global registry, multiprocess mode, and collector lifecycle to
render forty lines of text would be more moving parts than the thing being
monitored (CLAUDE.md: the architecture is deliberately small).

The format itself is stable and trivial::

    # HELP <name> <description>
    # TYPE <name> <counter|gauge>
    <name>{<label>="<value>",...} <number>

Everything here is pure — samples in, text out — so the exposition can be
asserted on in a unit test without an HTTP client.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock

_ESCAPES = {"\\": "\\\\", "\n": "\\n", '"': '\\"'}


def _escape_label_value(value: str) -> str:
    return "".join(_ESCAPES.get(char, char) for char in value)


def _render_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    # Sorted so the same sample set always renders byte-identically --
    # otherwise a diff of two scrapes is dominated by key reordering.
    pairs = ",".join(f'{key}="{_escape_label_value(labels[key])}"' for key in sorted(labels))
    return "{" + pairs + "}"


@dataclass(frozen=True)
class Sample:
    labels: Mapping[str, str]
    value: float


@dataclass(frozen=True)
class MetricFamily:
    name: str
    metric_type: str
    help_text: str
    samples: Sequence[Sample]


def render(families: Iterable[MetricFamily]) -> str:
    """Render metric families as Prometheus text exposition.

    A family with no samples still emits its HELP/TYPE header. That is not
    cosmetic: a metric that vanishes from the scrape is indistinguishable
    from a scrape that failed, whereas a declared metric with no series is
    unambiguously "nothing has happened yet".
    """
    lines: list[str] = []
    for family in families:
        lines.append(f"# HELP {family.name} {family.help_text}")
        lines.append(f"# TYPE {family.name} {family.metric_type}")
        for sample in family.samples:
            lines.append(f"{family.name}{_render_labels(sample.labels)} {sample.value}")
    return "\n".join(lines) + "\n"


@dataclass
class MetricsRegistry:
    """Tiny in-process counter/gauge store for the API.

    Counters only ever increase and are never reset by a scrape — Prometheus
    handles restarts via the counter-reset rule, and resetting on read would
    make two scrapers silently halve each other's numbers.
    """

    _counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    _gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    _help: dict[str, str] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    @staticmethod
    def _key(
        name: str, labels: Mapping[str, str] | None
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((labels or {}).items()))

    def increment(
        self,
        name: str,
        *,
        help_text: str = "",
        labels: Mapping[str, str] | None = None,
        amount: float = 1.0,
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount
            self._help.setdefault(name, help_text)

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        help_text: str = "",
        labels: Mapping[str, str] | None = None,
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value
            self._help.setdefault(name, help_text)

    def families(self) -> list[MetricFamily]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            help_texts = dict(self._help)

        families: list[MetricFamily] = []
        for metric_type, store in (("counter", counters), ("gauge", gauges)):
            by_name: dict[str, list[Sample]] = {}
            for (name, labels), value in store.items():
                by_name.setdefault(name, []).append(Sample(dict(labels), value))
            for name in sorted(by_name):
                families.append(
                    MetricFamily(
                        name=name,
                        metric_type=metric_type,
                        help_text=help_texts.get(name, ""),
                        samples=sorted(by_name[name], key=lambda s: sorted(s.labels.items())),
                    )
                )
        return families

    def render(self) -> str:
        return render(self.families())
