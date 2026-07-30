"""Bounded in-memory window state — pure, no I/O.

Only ring buffers and plain slicing/coverage helpers live here. Rebuilding
this state from Postgres on consumer restart (the "warm-up" step) is wiring
code that lives in ``services/consumer/main.py`` and ``scripts/seed_historical.py``:
it queries ``storage.repositories.ticks`` and feeds the results in through
``push``, one observation at a time, exactly as the streaming path does.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

# Must exceed the widest feature lookback (realised_vol_24h's 24h window —
# see features.pipeline.WINDOW_24H) with real margin. Coverage requires an
# observation strictly older than "as_of - window" to prove the window is
# fully populated; if retention equalled the window exactly, eviction would
# always discard that observation first, so the 24h window could never
# report sufficient coverage.
DEFAULT_MAX_HISTORY = timedelta(hours=27)
DEFAULT_MAX_POINTS = 10_000


@dataclass(frozen=True)
class Observation:
    """A single (timestamp, price, volume) point feeding the feature window."""

    observed_at: datetime
    price: float
    volume: float


class SymbolWindow:
    """Bounded ring buffer of observations for a single symbol.

    Bounded two ways so memory is O(max_window), never O(time): a hard cap
    on point count (``max_points``, via ``deque(maxlen=...)``) and a
    time-based eviction to ``max_history`` on every push.
    """

    def __init__(
        self,
        *,
        max_history: timedelta = DEFAULT_MAX_HISTORY,
        max_points: int = DEFAULT_MAX_POINTS,
    ) -> None:
        self._max_history = max_history
        self._buffer: deque[Observation] = deque(maxlen=max_points)

    def push(self, observation: Observation) -> None:
        self._buffer.append(observation)
        self._evict_older_than(observation.observed_at - self._max_history)

    def _evict_older_than(self, cutoff: datetime) -> None:
        while self._buffer and self._buffer[0].observed_at < cutoff:
            self._buffer.popleft()

    def snapshot(self, as_of: datetime | None = None) -> list[Observation]:
        """Ascending observations at or before ``as_of`` (default: all held).

        Filtering here — rather than trusting the caller to have already
        truncated — is what makes downstream feature computation immune to
        look-ahead even if a future point is already sitting in the buffer.
        """
        if as_of is None:
            return list(self._buffer)
        return [obs for obs in self._buffer if obs.observed_at <= as_of]

    def __len__(self) -> int:
        return len(self._buffer)


class WindowStore:
    """Per-symbol :class:`SymbolWindow` registry, created lazily on first push."""

    def __init__(
        self,
        *,
        max_history: timedelta = DEFAULT_MAX_HISTORY,
        max_points: int = DEFAULT_MAX_POINTS,
    ) -> None:
        self._max_history = max_history
        self._max_points = max_points
        self._windows: dict[str, SymbolWindow] = {}

    def push(self, symbol: str, observation: Observation) -> None:
        self._window_for(symbol).push(observation)

    def snapshot(self, symbol: str, as_of: datetime | None = None) -> list[Observation]:
        window = self._windows.get(symbol)
        if window is None:
            return []
        return window.snapshot(as_of)

    def _window_for(self, symbol: str) -> SymbolWindow:
        window = self._windows.get(symbol)
        if window is None:
            window = SymbolWindow(max_history=self._max_history, max_points=self._max_points)
            self._windows[symbol] = window
        return window


def slice_window(
    observations: Sequence[Observation], *, as_of: datetime, window: timedelta
) -> list[Observation]:
    """Ascending observations with ``as_of - window < observed_at <= as_of``."""
    start = as_of - window
    return [obs for obs in observations if start < obs.observed_at <= as_of]


def has_full_coverage(
    observations: Sequence[Observation],
    *,
    as_of: datetime,
    window: timedelta,
    min_points: int = 1,
) -> bool:
    """Whether a ``window``-sized feature can be trusted at ``as_of``.

    Requires both that the earliest observation on record predates the
    window's start (otherwise the window is truncated by how long we've been
    watching this symbol, not by an actual absence of movement) and that at
    least ``min_points`` observations fall inside it.

    The default of 1 deliberately does not require two points inside the
    window: tick density varies by source (a live poll every ~10s vs.
    hourly-granularity backfilled history), so a window narrower than the
    data's actual spacing is common and legitimate, not a coverage gap.
    Aggregation features (moving averages) are meaningful from a single
    point; delta-based features (returns, RSI, volatility) still guard their
    own minimum length in ``features.technical`` and fall back to ``None``
    there regardless of what this function decides.
    """
    if not observations:
        return False
    within = slice_window(observations, as_of=as_of, window=window)
    if len(within) < min_points:
        return False
    return observations[0].observed_at <= as_of - window


def detect_gap(
    observations: Sequence[Observation],
    *,
    as_of: datetime,
    window: timedelta,
    threshold: timedelta,
) -> bool:
    """True if any consecutive pair within ``window`` is more than ``threshold`` apart."""
    within = slice_window(observations, as_of=as_of, window=window)
    for previous, current in zip(within, within[1:], strict=False):
        if current.observed_at - previous.observed_at > threshold:
            return True
    return False
