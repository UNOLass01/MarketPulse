"""Serving-side prediction: guards, inference, and the in-process model cache.

This is the logic half of Phase 5. ``services/api`` is wiring around it
(CLAUDE.md rule #9's spirit: deleting ``services/`` should leave nothing
interesting behind), which is what makes the guards below testable without
an HTTP client, a database, or a tracking server.

Three things happen here and nothing else:

1. **Guards.** A prediction is refused — loudly, with the numbers attached —
   when the underlying features are too old (:class:`FeaturesStaleError`) or
   were produced by a different feature set than the model was trained on
   (:class:`FeatureSchemaMismatchError`). Both refusals are strictly better
   than a plausible-looking answer the caller cannot audit.
2. **Inference.** Feature values are projected into the registry's canonical
   column order (``features.registry.ordered_values``), never into whatever
   order a dict or a database row happened to yield.
3. **Caching.** :class:`ModelCache` holds the loaded model so request latency
   never includes an MLflow round-trip, and so API availability is not
   coupled to the tracking server's.

No feature is *computed* here (CLAUDE.md rule #5) — values arrive already
persisted and are only reordered.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from marketpulse.exceptions import PermanentError, TransientError
from marketpulse.features.registry import FEATURE_NAMES, ordered_values
from marketpulse.logging import get_logger
from marketpulse.ml.labeling import INDEX_TO_LABEL, LABELS

logger = get_logger(__name__)


class ModelUnavailableError(TransientError):
    """No Production model is loaded.

    Transient by classification: a model can be promoted at any moment and
    the next refresh will pick it up, so the caller's retry is meaningful.
    """


class FeaturesStaleError(TransientError):
    """The newest stored feature row is older than the staleness threshold.

    Carries the actual age so the refusal is actionable — "no prediction" and
    "no prediction, the data is 40 minutes old" are very different messages
    to be woken up by.
    """

    def __init__(self, *, symbol: str, feature_ts: datetime, age_seconds: float, max_age: float):
        super().__init__(
            f"features for {symbol} are {age_seconds:.1f}s old, "
            f"exceeding the {max_age:.1f}s staleness threshold"
        )
        self.symbol = symbol
        self.feature_ts = feature_ts
        self.age_seconds = age_seconds
        self.max_age_seconds = max_age

    def details(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "feature_ts": self.feature_ts.isoformat(),
            "feature_age_seconds": self.age_seconds,
            "max_feature_age_seconds": self.max_age_seconds,
        }


class FeatureSchemaMismatchError(PermanentError):
    """A stored feature row's ``feature_set_version`` differs from the one the
    loaded model was trained against.

    Permanent, not transient: retrying serves the same mismatched row again.
    This is the guard that catches a retrain which changed the feature set
    without a coordinated deploy — the failure mode where every number the
    API returns is confidently wrong.
    """

    def __init__(self, *, symbol: str, feature_set_version: int, model_feature_set_version: int):
        super().__init__(
            f"features for {symbol} are feature_set_version="
            f"{feature_set_version} but the loaded model expects "
            f"{model_feature_set_version}"
        )
        self.symbol = symbol
        self.feature_set_version = feature_set_version
        self.model_feature_set_version = model_feature_set_version

    def details(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "feature_set_version": self.feature_set_version,
            "model_feature_set_version": self.model_feature_set_version,
        }


class ModelSignatureMismatchError(PermanentError):
    """A model's logged input signature is not the registry's feature set.

    Raised at *load* time, never per request. A model whose columns disagree
    with ``features.registry`` would feed every value into the wrong slot and
    still return a confident-looking probability vector, so it must be
    rejected before it can serve anything.
    """


class SupportsPredict(Protocol):
    """The slice of an MLflow pyfunc model this module actually uses.

    Typed as a Protocol so tests can pass a five-line stub instead of
    standing up a tracking server to exercise a guard.
    """

    def predict(self, data: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class LoadedModel:
    """A Production model resolved from the registry, plus the metadata the
    guards need. Immutable: a refresh builds a *new* instance rather than
    mutating this one, which is what makes the cache swap safe.
    """

    model: SupportsPredict
    version: str
    feature_set_version: int
    #: The model's own input column order, read from its logged signature.
    #: Should equal ``FEATURE_NAMES``; :func:`validate_feature_names` is what
    #: turns "should" into "verified at load time".
    feature_names: tuple[str, ...]
    run_id: str | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Prediction:
    symbol: str
    label: str
    probabilities: dict[str, float]
    model_version: str
    feature_set_version: int
    feature_ts: datetime
    feature_age_seconds: float
    predicted_at: datetime
    latency_ms: float


@dataclass(frozen=True)
class FeatureSnapshot:
    """One stored feature row, as read back from ``storage.repositories``.

    A plain value object rather than the ORM row so this module stays
    DB-free and every test below can build one by hand.
    """

    symbol: str
    feature_ts: datetime
    feature_set_version: int
    feature_values: dict[str, float | None]
    insufficient_history: bool
    has_gap: bool


def feature_age_seconds(feature_ts: datetime, *, now: datetime) -> float:
    """Age of a feature row in seconds. Negative ages (a clock skew between
    the writer and this process) are clamped to 0 rather than reported as
    "from the future", which would read as impossibly fresh and silently
    disarm the staleness guard.
    """
    return max((now - feature_ts).total_seconds(), 0.0)


def check_staleness(snapshot: FeatureSnapshot, *, now: datetime, max_age: timedelta) -> float:
    """Return the row's age, raising :class:`FeaturesStaleError` past ``max_age``."""
    age = feature_age_seconds(snapshot.feature_ts, now=now)
    if age > max_age.total_seconds():
        raise FeaturesStaleError(
            symbol=snapshot.symbol,
            feature_ts=snapshot.feature_ts,
            age_seconds=age,
            max_age=max_age.total_seconds(),
        )
    return age


def check_feature_schema(snapshot: FeatureSnapshot, loaded: LoadedModel) -> None:
    """Raise :class:`FeatureSchemaMismatchError` unless the row's feature set
    version matches the loaded model's.
    """
    if snapshot.feature_set_version != loaded.feature_set_version:
        raise FeatureSchemaMismatchError(
            symbol=snapshot.symbol,
            feature_set_version=snapshot.feature_set_version,
            model_feature_set_version=loaded.feature_set_version,
        )


def validate_feature_names(feature_names: Sequence[str]) -> None:
    """Assert a model's signature columns are exactly the registry's, in order.

    Called at load time, not per request: a model whose signature disagrees
    with ``features.registry`` must never reach the serving path at all,
    because every prediction it makes would silently feed values into the
    wrong columns.
    """
    actual = tuple(feature_names)
    if actual != FEATURE_NAMES:
        raise ModelSignatureMismatchError(
            f"model signature columns {actual} do not match the feature "
            f"registry's {FEATURE_NAMES}"
        )


def build_feature_frame(snapshots: Sequence[FeatureSnapshot]) -> pd.DataFrame:
    """Project stored feature values into the registry's canonical column order.

    ``ordered_values`` raises on a missing registered name rather than
    filling it with ``None``: a feature the pipeline never computed is a bug
    and must not be indistinguishable from a legitimate
    insufficient-history null (CLAUDE.md rule #7).

    The explicit ``astype("float64")`` mirrors ``ml.dataset._rows_to_frame``
    for the same reason it exists there — an all-null column infers as
    ``object`` dtype, which LightGBM rejects outright.
    """
    records = [
        dict(zip(FEATURE_NAMES, ordered_values(snapshot.feature_values), strict=True))
        for snapshot in snapshots
    ]
    frame = pd.DataFrame.from_records(records, columns=list(FEATURE_NAMES))
    return frame.astype("float64")


def probabilities_to_mapping(row: Sequence[float]) -> dict[str, float]:
    """Map a raw probability row onto class labels via ``labels.INDEX_TO_LABEL``.

    Never zips against a locally-written label tuple — class index order is
    defined in exactly one place, the same discipline ``features.registry``
    applies to column order.
    """
    if len(row) != len(LABELS):
        raise ValueError(
            f"model returned {len(row)} probabilities but there are {len(LABELS)} classes"
        )
    return {INDEX_TO_LABEL[index]: float(value) for index, value in enumerate(row)}


def _as_proba_matrix(raw: Any, *, n_rows: int) -> np.ndarray:
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(n_rows, -1)
    if matrix.shape != (n_rows, len(LABELS)):
        raise ValueError(
            f"model returned predictions of shape {matrix.shape}, "
            f"expected {(n_rows, len(LABELS))}"
        )
    return matrix


def predict_batch(
    loaded: LoadedModel,
    snapshots: Sequence[FeatureSnapshot],
    *,
    now: datetime,
    max_feature_age: timedelta,
) -> list[Prediction]:
    """Guard, then score every snapshot in a single model call.

    Guards run over the whole batch *before* inference, so a stale or
    schema-mismatched symbol fails the request rather than being quietly
    dropped from an otherwise-successful response. A partial batch would let
    a caller iterate the list and never notice a symbol went missing.
    """
    if not snapshots:
        return []

    started = datetime.now(UTC)
    ages = [check_staleness(snapshot, now=now, max_age=max_feature_age) for snapshot in snapshots]
    for snapshot in snapshots:
        check_feature_schema(snapshot, loaded)

    frame = build_feature_frame(snapshots)
    matrix = _as_proba_matrix(loaded.model.predict(frame), n_rows=len(snapshots))

    predicted_at = datetime.now(UTC)
    # One elapsed measurement for the batch, attributed to each row: the
    # model call dominates and is genuinely shared, so splitting it per row
    # would invent precision that isn't there.
    latency_ms = (predicted_at - started).total_seconds() * 1000.0

    predictions = []
    for snapshot, age, row in zip(snapshots, ages, matrix, strict=True):
        probabilities = probabilities_to_mapping(row)
        predictions.append(
            Prediction(
                symbol=snapshot.symbol,
                label=INDEX_TO_LABEL[int(row.argmax())],
                probabilities=probabilities,
                model_version=loaded.version,
                feature_set_version=loaded.feature_set_version,
                feature_ts=snapshot.feature_ts,
                feature_age_seconds=age,
                predicted_at=predicted_at,
                latency_ms=latency_ms,
            )
        )
    return predictions


def predict_one(
    loaded: LoadedModel,
    snapshot: FeatureSnapshot,
    *,
    now: datetime,
    max_feature_age: timedelta,
) -> Prediction:
    return predict_batch(loaded, [snapshot], now=now, max_feature_age=max_feature_age)[0]


@dataclass(frozen=True)
class RefreshResult:
    changed: bool
    previous_version: str | None
    current_version: str | None
    error: str | None = None


class ModelCache:
    """Holds the Production model in memory, refreshed in the background.

    Two properties matter and both are deliberate:

    * **The swap is atomic and never drops an in-flight request.** A refresh
      loads the new model into a local, then rebinds one attribute. Readers
      take a single reference via :attr:`current` and keep using it for the
      whole request, so a request that started on v3 finishes on v3 even if
      v4 lands mid-flight.
    * **A failed refresh keeps the old model.** The previous reference is
      only replaced once a new one has been fully constructed. Never fail
      open into an unmodelled state; record the error and keep serving.

    ``loader`` is injected rather than calling MLflow directly so the whole
    class is testable with a function that returns a stub.
    """

    def __init__(
        self,
        loader: Callable[[], LoadedModel | None],
        *,
        refresh_interval: timedelta | None = None,
    ) -> None:
        self._loader = loader
        self._refresh_interval = refresh_interval
        self._current: LoadedModel | None = None
        self._last_error: str | None = None
        # Guards refresh() against itself (background tick racing a manual
        # POST /model/refresh), not the read path -- readers take a lock-free
        # single-attribute read on purpose.
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    @property
    def last_refresh_error(self) -> str | None:
        return self._last_error

    def require(self) -> LoadedModel:
        """The loaded model, or :class:`ModelUnavailableError`.

        Used by prediction routes; ``/health`` and ``/ready`` use
        :attr:`current` instead, since "no model" is a state they report on
        rather than fail on.
        """
        loaded = self._current
        if loaded is None:
            raise ModelUnavailableError(
                "no Production model is loaded"
                + (f" (last refresh error: {self._last_error})" if self._last_error else "")
            )
        return loaded

    def refresh(self) -> RefreshResult:
        """Re-resolve the Production model, swapping it in only on success."""
        with self._refresh_lock:
            previous = self._current
            previous_version = previous.version if previous else None
            try:
                candidate = self._loader()
            except Exception as exc:  # noqa: BLE001 - any loader failure is survivable
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "model refresh failed; keeping previously loaded model",
                    extra={
                        "extra_fields": {
                            "error": self._last_error,
                            "retained_model_version": previous_version,
                        }
                    },
                )
                return RefreshResult(
                    changed=False,
                    previous_version=previous_version,
                    current_version=previous_version,
                    error=self._last_error,
                )

            if candidate is None:
                # Nothing in Production. Distinct from a *failure*: there is
                # no error to report, and if a model was previously loaded we
                # keep serving it rather than going dark on an empty registry.
                self._last_error = None
                return RefreshResult(
                    changed=False,
                    previous_version=previous_version,
                    current_version=previous_version,
                )

            self._last_error = None
            if previous is not None and previous.version == candidate.version:
                return RefreshResult(
                    changed=False,
                    previous_version=previous_version,
                    current_version=candidate.version,
                )

            self._current = candidate  # the atomic swap
            logger.info(
                "model version swapped",
                extra={
                    "extra_fields": {
                        "previous_model_version": previous_version,
                        "model_version": candidate.version,
                        "feature_set_version": candidate.feature_set_version,
                    }
                },
            )
            return RefreshResult(
                changed=True,
                previous_version=previous_version,
                current_version=candidate.version,
            )

    def start_background_refresh(self, interval: timedelta | None = None) -> None:
        """Poll the registry on ``interval`` until :meth:`stop` is called.

        Waits on an :class:`threading.Event` rather than sleeping, so
        shutdown is immediate instead of taking up to a full interval.
        """
        resolved = interval or self._refresh_interval
        if resolved is None:
            raise ValueError("no refresh interval configured")
        if self._thread is not None:
            return

        seconds = resolved.total_seconds()

        def _loop() -> None:
            while not self._stop.wait(seconds):
                self.refresh()

        self._stop.clear()
        self._thread = threading.Thread(target=_loop, name="model-refresh", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
