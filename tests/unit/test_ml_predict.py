"""Serving guards, inference, and the model cache (Phase 5).

The guards are the point of this file. A prediction served off stale features
or against a mismatched feature set is worse than no prediction at all,
because the caller cannot tell — so each refusal is asserted on directly,
including the numbers it must carry in its body.
"""

import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from marketpulse.ml.labeling import LABEL_DOWN, LABEL_STABLE, LABEL_UP, LABELS
from marketpulse.ml.predict import (
    FeatureSchemaMismatchError,
    FeatureSnapshot,
    FeaturesStaleError,
    LoadedModel,
    ModelCache,
    ModelSignatureMismatchError,
    ModelUnavailableError,
    build_feature_frame,
    check_staleness,
    feature_age_seconds,
    predict_batch,
    predict_one,
    probabilities_to_mapping,
    validate_feature_names,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
MAX_AGE = timedelta(seconds=120)


class StubModel:
    """Returns a fixed probability row per input row, and records what it saw.

    ``seen`` is what makes the feature-ordering test possible: the assertion
    is not "the prediction was right", it is "the model was handed columns in
    the registry's order".
    """

    def __init__(self, row: list[float] | None = None) -> None:
        self.row = row or [0.1, 0.2, 0.7]
        self.seen: pd.DataFrame | None = None

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        self.seen = data.copy()
        return np.array([self.row] * len(data))


def make_values(**overrides: float | None) -> dict[str, float | None]:
    values: dict[str, float | None] = {name: 1.0 for name in FEATURE_NAMES}
    values.update(overrides)
    return values


def make_snapshot(
    *,
    symbol: str = "BTC-USD",
    age_seconds: float = 10.0,
    feature_set_version: int = FEATURE_SET_VERSION,
    values: dict[str, float | None] | None = None,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol=symbol,
        feature_ts=NOW - timedelta(seconds=age_seconds),
        feature_set_version=feature_set_version,
        feature_values=values or make_values(),
        insufficient_history=False,
        has_gap=False,
    )


def make_loaded(
    *, version: str = "3", feature_set_version: int = FEATURE_SET_VERSION, model: object = None
) -> LoadedModel:
    return LoadedModel(
        model=model or StubModel(),  # type: ignore[arg-type]
        version=version,
        feature_set_version=feature_set_version,
        feature_names=FEATURE_NAMES,
    )


# --- staleness guard ------------------------------------------------------


def test_fresh_features_pass_the_staleness_guard() -> None:
    age = check_staleness(make_snapshot(age_seconds=30), now=NOW, max_age=MAX_AGE)
    assert age == pytest.approx(30.0)


def test_stale_features_raise_with_the_age_in_the_error() -> None:
    snapshot = make_snapshot(age_seconds=2400)  # 40 minutes

    with pytest.raises(FeaturesStaleError) as excinfo:
        check_staleness(snapshot, now=NOW, max_age=MAX_AGE)

    error = excinfo.value
    assert error.age_seconds == pytest.approx(2400.0)
    assert error.max_age_seconds == pytest.approx(120.0)
    # The body must state the age -- a bare "unavailable" leaves the caller
    # unable to distinguish a blip from a dead pipeline.
    details = error.details()
    assert details["feature_age_seconds"] == pytest.approx(2400.0)
    assert details["max_feature_age_seconds"] == pytest.approx(120.0)
    assert "2400" in str(error) or "2400.0" in str(error)


def test_staleness_boundary_is_inclusive() -> None:
    # Exactly at the threshold is still served; only strictly older refuses.
    assert check_staleness(make_snapshot(age_seconds=120), now=NOW, max_age=MAX_AGE) == 120.0
    with pytest.raises(FeaturesStaleError):
        check_staleness(make_snapshot(age_seconds=120.5), now=NOW, max_age=MAX_AGE)


def test_future_feature_ts_clamps_to_zero_rather_than_reading_as_fresh() -> None:
    # Clock skew between writer and reader must not produce a negative age --
    # "impossibly fresh" would silently disarm the guard.
    future = NOW + timedelta(seconds=45)
    assert feature_age_seconds(future, now=NOW) == 0.0


def test_batch_refuses_entirely_when_one_symbol_is_stale() -> None:
    # Not a partial response: silently dropping the stale symbol would let a
    # caller iterate the list and never notice it went missing.
    snapshots = [make_snapshot(symbol="BTC-USD"), make_snapshot(symbol="ETH-USD", age_seconds=9999)]

    with pytest.raises(FeaturesStaleError) as excinfo:
        predict_batch(make_loaded(), snapshots, now=NOW, max_feature_age=MAX_AGE)

    assert excinfo.value.symbol == "ETH-USD"


# --- schema guard ---------------------------------------------------------


def test_feature_set_version_mismatch_is_refused_not_silently_predicted() -> None:
    loaded = make_loaded(feature_set_version=2)
    snapshot = make_snapshot(feature_set_version=1)

    with pytest.raises(FeatureSchemaMismatchError) as excinfo:
        predict_one(loaded, snapshot, now=NOW, max_feature_age=MAX_AGE)

    details = excinfo.value.details()
    assert details["feature_set_version"] == 1
    assert details["model_feature_set_version"] == 2


def test_matching_feature_set_version_predicts() -> None:
    prediction = predict_one(
        make_loaded(feature_set_version=1),
        make_snapshot(feature_set_version=1),
        now=NOW,
        max_feature_age=MAX_AGE,
    )
    assert prediction.feature_set_version == 1


def test_model_signature_must_match_the_registry() -> None:
    validate_feature_names(FEATURE_NAMES)  # the happy path

    with pytest.raises(ModelSignatureMismatchError):
        validate_feature_names(tuple(reversed(FEATURE_NAMES)))
    with pytest.raises(ModelSignatureMismatchError):
        validate_feature_names(FEATURE_NAMES[:-1])


# --- feature ordering -----------------------------------------------------


def test_feature_frame_uses_registry_order_not_dict_order() -> None:
    # The DB returns JSONB whose key order is not guaranteed. Reverse it to
    # prove the frame's columns come from features.registry regardless.
    reversed_values = dict(reversed(list(make_values().items())))
    assert list(reversed_values) != list(FEATURE_NAMES)

    frame = build_feature_frame([make_snapshot(values=reversed_values)])
    assert list(frame.columns) == list(FEATURE_NAMES)


def test_model_receives_columns_in_registry_order_even_when_db_reorders() -> None:
    model = StubModel()
    scrambled = dict(sorted(make_values().items()))  # alphabetical, not registry order
    assert list(scrambled) != list(FEATURE_NAMES)

    predict_one(
        make_loaded(model=model),
        make_snapshot(values=scrambled),
        now=NOW,
        max_feature_age=MAX_AGE,
    )

    assert model.seen is not None
    assert list(model.seen.columns) == list(FEATURE_NAMES)


def test_a_missing_registered_feature_raises_rather_than_being_filled() -> None:
    # A feature the pipeline never computed is a bug, and must not be
    # indistinguishable from a legitimate insufficient-history null.
    incomplete = make_values()
    del incomplete[FEATURE_NAMES[0]]

    with pytest.raises(KeyError):
        build_feature_frame([make_snapshot(values=incomplete)])


def test_all_null_feature_column_is_float_not_object_dtype() -> None:
    # LightGBM and MLflow signature inference both reject object dtype, even
    # when it holds only floats and None.
    frame = build_feature_frame([make_snapshot(values=make_values(roc_1m=None))])
    assert frame["roc_1m"].dtype == "float64"


# --- probabilities --------------------------------------------------------


def test_probabilities_are_keyed_by_the_canonical_label_order() -> None:
    mapping = probabilities_to_mapping([0.1, 0.2, 0.7])
    assert mapping == {LABEL_DOWN: 0.1, LABEL_STABLE: 0.2, LABEL_UP: 0.7}


def test_wrong_probability_width_raises() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        probabilities_to_mapping([0.5, 0.5])


def test_prediction_reports_all_three_classes_and_the_argmax_label() -> None:
    prediction = predict_one(
        make_loaded(model=StubModel([0.6, 0.3, 0.1])),
        make_snapshot(age_seconds=15),
        now=NOW,
        max_feature_age=MAX_AGE,
    )
    assert prediction.label == LABEL_DOWN
    assert set(prediction.probabilities) == set(LABELS)
    assert prediction.feature_age_seconds == pytest.approx(15.0)
    assert prediction.model_version == "3"
    assert prediction.latency_ms >= 0.0


def test_batch_predicts_every_symbol_in_one_model_call() -> None:
    model = StubModel()
    snapshots = [make_snapshot(symbol=s) for s in ("BTC-USD", "ETH-USD", "SOL-USD")]

    predictions = predict_batch(
        make_loaded(model=model), snapshots, now=NOW, max_feature_age=MAX_AGE
    )

    assert [p.symbol for p in predictions] == ["BTC-USD", "ETH-USD", "SOL-USD"]
    assert model.seen is not None and len(model.seen) == 3


def test_empty_batch_is_not_an_error() -> None:
    assert predict_batch(make_loaded(), [], now=NOW, max_feature_age=MAX_AGE) == []


# --- model cache ----------------------------------------------------------


def test_require_raises_when_no_model_is_loaded() -> None:
    cache = ModelCache(lambda: None)
    with pytest.raises(ModelUnavailableError):
        cache.require()


def test_refresh_loads_and_reports_the_change() -> None:
    cache = ModelCache(lambda: make_loaded(version="7"))

    result = cache.refresh()

    assert result.changed is True
    assert result.previous_version is None
    assert result.current_version == "7"
    assert cache.require().version == "7"


def test_refresh_swaps_version_without_disturbing_an_in_flight_reference() -> None:
    versions = iter([make_loaded(version="1"), make_loaded(version="2")])
    cache = ModelCache(lambda: next(versions))
    cache.refresh()

    # A request takes its reference once, then keeps using it. A promotion
    # landing mid-request must not change the model under its feet.
    in_flight = cache.require()
    result = cache.refresh()

    assert result.changed is True
    assert result.previous_version == "1"
    assert result.current_version == "2"
    assert in_flight.version == "1"  # untouched
    assert cache.require().version == "2"  # next request sees the new one


def test_same_version_is_not_reported_as_a_change() -> None:
    cache = ModelCache(lambda: make_loaded(version="4"))
    cache.refresh()
    assert cache.refresh().changed is False


def test_failed_refresh_retains_the_previous_model() -> None:
    state = {"fail": False}

    def loader() -> LoadedModel | None:
        if state["fail"]:
            raise ConnectionError("mlflow unreachable")
        return make_loaded(version="5")

    cache = ModelCache(loader)
    cache.refresh()
    state["fail"] = True

    result = cache.refresh()

    # Never fail open into an unmodelled state: the old model keeps serving
    # and the error is recorded for /model/current to surface.
    assert result.changed is False
    assert result.error is not None and "mlflow unreachable" in result.error
    assert cache.require().version == "5"
    assert cache.last_refresh_error is not None


def test_a_successful_refresh_clears_a_previous_error() -> None:
    state = {"fail": True}

    def loader() -> LoadedModel | None:
        if state["fail"]:
            raise ConnectionError("down")
        return make_loaded(version="6")

    cache = ModelCache(loader)
    cache.refresh()
    assert cache.last_refresh_error is not None

    state["fail"] = False
    cache.refresh()
    assert cache.last_refresh_error is None


def test_empty_registry_is_not_an_error_and_keeps_any_loaded_model() -> None:
    state: dict[str, LoadedModel | None] = {"value": make_loaded(version="9")}
    cache = ModelCache(lambda: state["value"])
    cache.refresh()

    state["value"] = None  # nothing in Production any more
    result = cache.refresh()

    assert result.error is None
    assert cache.last_refresh_error is None
    assert cache.require().version == "9"


def test_background_refresh_picks_up_a_new_version_and_stops_cleanly() -> None:
    loaded = threading.Event()
    versions = iter(["1", "2", "2", "2", "2"])

    def loader() -> LoadedModel | None:
        try:
            version = next(versions)
        except StopIteration:
            version = "2"
        if version == "2":
            loaded.set()
        return make_loaded(version=version)

    cache = ModelCache(loader)
    cache.refresh()
    assert cache.require().version == "1"

    cache.start_background_refresh(timedelta(milliseconds=20))
    try:
        assert loaded.wait(timeout=5.0), "background refresh never ran"
        # Give the swap a moment to land after the loader returned.
        for _ in range(100):
            if cache.require().version == "2":
                break
            threading.Event().wait(0.01)
        assert cache.require().version == "2"
    finally:
        cache.stop()

    assert cache._thread is None  # noqa: SLF001 - asserting shutdown really joined


def test_start_background_refresh_requires_an_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        ModelCache(lambda: None).start_background_refresh()
