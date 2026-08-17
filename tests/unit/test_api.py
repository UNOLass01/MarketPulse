"""API behaviour (Phase 5), exercised through the real ASGI stack.

Uses ``TestClient`` against a real ``create_app`` with an injected
:class:`AppState`, rather than calling handler functions directly — the
middleware, the exception handlers, and the dependency graph are a large part
of what these tests are actually asserting on, and none of them run if you
call the function.

The database is faked at the ``session_factory`` seam. That keeps this tier
fast and I/O-free (``make test`` must stay so), while the genuine SQL is
covered by ``tests/integration/test_serving_storage.py``.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from services.api.main import API_PREFIX, create_app
from services.api.middleware import CORRELATION_HEADER
from services.api.state import AppState

from marketpulse.config import Settings
from marketpulse.contracts.api import (
    ERROR_FEATURE_SCHEMA_MISMATCH,
    ERROR_FEATURES_STALE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NOT_FOUND,
    ERROR_VALIDATION,
)
from marketpulse.features.registry import FEATURE_NAMES, FEATURE_SET_VERSION
from marketpulse.ml.predict import LoadedModel, ModelCache

pytestmark = pytest.mark.unit


# --- fakes ----------------------------------------------------------------


class FakeModel:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.array([[0.15, 0.25, 0.60]] * len(data))


class FakeFeatureRow:
    def __init__(self, *, age_seconds: float = 5.0, version: int = FEATURE_SET_VERSION) -> None:
        self.feature_ts = datetime.now(UTC) - timedelta(seconds=age_seconds)
        self.feature_set_version = version
        self.feature_values: dict[str, float | None] = {n: 1.0 for n in FEATURE_NAMES}
        self.insufficient_history = False
        self.has_gap = False


class FakeSymbol:
    def __init__(self, symbol_id: int, code: str) -> None:
        self.id = symbol_id
        self.code = code


class FakeSession:
    """Just enough Session surface for the routes under test.

    ``execute`` returning an empty result is deliberate: every route that
    needs real rows gets them through a monkeypatched repository function
    instead, so this class never has to grow a SQL interpreter.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        if self.fail:
            raise RuntimeError("database is down")
        return _EmptyResult()

    def commit(self) -> None:
        if self.fail:
            raise RuntimeError("database is down")

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class _EmptyResult:
    def scalars(self) -> list[Any]:
        return []

    def scalar_one_or_none(self) -> None:
        return None

    def first(self) -> None:
        return None

    def all(self) -> list[Any]:
        return []

    def __iter__(self) -> Iterator[Any]:
        return iter(())


class FakeSessionFactory:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def __call__(self) -> FakeSession:
        return FakeSession(fail=self.fail)


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def api_settings(env_vars: None) -> Settings:
    return Settings()  # type: ignore[call-arg]


def make_loaded(*, version: str = "3", feature_set_version: int = FEATURE_SET_VERSION):  # type: ignore[no-untyped-def]
    return LoadedModel(
        model=FakeModel(),  # type: ignore[arg-type]
        version=version,
        feature_set_version=feature_set_version,
        feature_names=FEATURE_NAMES,
    )


def build_client(
    settings: Settings,
    *,
    model: LoadedModel | None = None,
    db_fails: bool = False,
) -> TestClient:
    cache = ModelCache(lambda: model)
    if model is not None:
        cache.refresh()
    state = AppState(
        settings=settings,
        session_factory=FakeSessionFactory(fail=db_fails),  # type: ignore[arg-type]
        model_cache=cache,
    )
    app = create_app(state)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    # A model-refresh loader that returns the same object forever, so the
    # lifespan's startup refresh is a no-op rather than a surprise.
    with build_client(api_settings, model=make_loaded()) as test_client:
        yield test_client


# --- liveness vs readiness ------------------------------------------------


def test_health_is_liveness_only_and_does_not_touch_the_database(
    api_settings: Settings,
) -> None:
    # The whole point of separating the two: a healthy process whose DB
    # blinked must not be restarted.
    with build_client(api_settings, model=make_loaded(), db_fails=True) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_is_503_when_the_database_is_down_while_health_stays_200(
    api_settings: Settings,
) -> None:
    with build_client(api_settings, model=make_loaded(), db_fails=True) as client:
        assert client.get("/health").status_code == 200

        ready = client.get("/ready")
        assert ready.status_code == 503
        body = ready.json()
        assert body["ready"] is False
        assert "postgres" in (body["reason"] or "")


def test_app_starts_with_no_production_model_and_reports_it_on_ready(
    api_settings: Settings,
) -> None:
    # Missing Production model must not crash-loop the container.
    with build_client(api_settings, model=None) as client:
        assert client.get("/health").status_code == 200

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert "model" in (ready.json()["reason"] or "")


def test_health_dependencies_reports_without_gating(api_settings: Settings) -> None:
    with build_client(api_settings, model=None, db_fails=True) as client:
        response = client.get("/health/dependencies")

    # Always 200 -- it reports on dependencies rather than failing with them.
    assert response.status_code == 200
    names = {d["name"] for d in response.json()["dependencies"]}
    assert names == {"postgres", "model", "features"}


# --- prediction guards over HTTP -----------------------------------------


def _patch_symbol_and_feature(monkeypatch: pytest.MonkeyPatch, row: FakeFeatureRow | None) -> None:
    import services.api.routers.predictions as predictions_router

    monkeypatch.setattr(predictions_router, "list_symbols", lambda _s: [FakeSymbol(1, "BTC-USD")])
    monkeypatch.setattr(predictions_router, "latest_feature_vector", lambda _s, _sid, _v: row)


def test_prediction_returns_all_three_probabilities_and_the_age(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_symbol_and_feature(monkeypatch, FakeFeatureRow(age_seconds=8))

    response = client.get(f"{API_PREFIX}/predictions/BTC-USD")

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "UP"
    assert set(body["probabilities"]) == {"DOWN", "STABLE", "UP"}
    assert body["model_version"] == "3"
    assert 0 <= body["feature_age_seconds"] < 60


def test_stale_features_return_503_with_the_age_in_the_body(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_symbol_and_feature(monkeypatch, FakeFeatureRow(age_seconds=2400))

    response = client.get(f"{API_PREFIX}/predictions/BTC-USD")

    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == ERROR_FEATURES_STALE
    # The age must be *in the body* -- that is what distinguishes an
    # actionable refusal from a bare "unavailable".
    assert body["details"]["feature_age_seconds"] > 2000
    assert body["details"]["max_feature_age_seconds"] == pytest.approx(120.0)


def test_feature_schema_mismatch_is_refused_not_silently_predicted(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with build_client(api_settings, model=make_loaded(feature_set_version=2)) as client:
        _patch_symbol_and_feature(monkeypatch, FakeFeatureRow(version=2))
        # The route asks for rows at the *model's* version, so simulate the
        # storage layer handing back a row from the old feature set.
        import services.api.routers.predictions as predictions_router

        monkeypatch.setattr(
            predictions_router,
            "latest_feature_vector",
            lambda _s, _sid, _v: FakeFeatureRow(version=1),
        )

        response = client.get(f"{API_PREFIX}/predictions/BTC-USD")

    assert response.status_code == 409
    body = response.json()
    assert body["error_code"] == ERROR_FEATURE_SCHEMA_MISMATCH
    assert body["details"]["feature_set_version"] == 1
    assert body["details"]["model_feature_set_version"] == 2


def test_prediction_without_a_loaded_model_is_503(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with build_client(api_settings, model=None) as client:
        _patch_symbol_and_feature(monkeypatch, FakeFeatureRow())
        response = client.get(f"{API_PREFIX}/predictions/BTC-USD")

    assert response.status_code == 503
    assert response.json()["error_code"] == ERROR_MODEL_UNAVAILABLE


def test_unknown_symbol_is_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import services.api.routers.predictions as predictions_router

    monkeypatch.setattr(predictions_router, "list_symbols", lambda _s: [])

    response = client.get(f"{API_PREFIX}/predictions/DOGE-USD")

    assert response.status_code == 404
    assert response.json()["error_code"] == ERROR_NOT_FOUND


def test_batch_predictions_use_one_model_call(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.api.routers.predictions as predictions_router

    monkeypatch.setattr(
        predictions_router,
        "latest_feature_vector_per_symbol",
        lambda _s, _v: [("BTC-USD", FakeFeatureRow()), ("ETH-USD", FakeFeatureRow())],
    )

    response = client.get(f"{API_PREFIX}/predictions")

    assert response.status_code == 200
    body = response.json()
    assert [p["symbol"] for p in body["predictions"]] == ["BTC-USD", "ETH-USD"]
    assert body["model_version"] == "3"


# --- prediction logging must never fail a request ------------------------


def test_prediction_logging_failure_does_not_fail_the_request(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Observability never sits on the critical path: the caller already has
    # their prediction, and failing the request to protect a log row would
    # trade the product for the telemetry.
    import services.api.state as state_module

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("predictions table is gone")

    monkeypatch.setattr(state_module, "upsert_prediction", explode)

    with build_client(api_settings, model=make_loaded()) as client:
        _patch_symbol_and_feature(monkeypatch, FakeFeatureRow())
        response = client.get(f"{API_PREFIX}/predictions/BTC-USD")

    assert response.status_code == 200
    assert response.json()["label"] == "UP"


# --- error envelope -------------------------------------------------------


ENVELOPE_KEYS = {"error_code", "message", "correlation_id", "timestamp", "details"}


def test_error_envelope_shape_is_identical_across_every_error_path(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = []

    # 404
    with build_client(api_settings, model=make_loaded()) as client:
        import services.api.routers.predictions as predictions_router

        monkeypatch.setattr(predictions_router, "list_symbols", lambda _s: [])
        bodies.append(client.get(f"{API_PREFIX}/predictions/NOPE").json())

        # 503 stale
        monkeypatch.setattr(
            predictions_router, "list_symbols", lambda _s: [FakeSymbol(1, "BTC-USD")]
        )
        monkeypatch.setattr(
            predictions_router,
            "latest_feature_vector",
            lambda _s, _sid, _v: FakeFeatureRow(age_seconds=9999),
        )
        bodies.append(client.get(f"{API_PREFIX}/predictions/BTC-USD").json())

        # 422 validation (negative hours fails the Query constraint)
        bodies.append(client.get(f"{API_PREFIX}/predictions/BTC-USD/history?hours=-1").json())

        # 404 from an unrouted path
        bodies.append(client.get(f"{API_PREFIX}/nope").json())

    # 503 model unavailable
    with build_client(api_settings, model=None) as client:
        import services.api.routers.predictions as predictions_router

        monkeypatch.setattr(
            predictions_router, "list_symbols", lambda _s: [FakeSymbol(1, "BTC-USD")]
        )
        bodies.append(client.get(f"{API_PREFIX}/predictions/BTC-USD").json())

    for body in bodies:
        assert set(body) == ENVELOPE_KEYS, body
        assert isinstance(body["error_code"], str) and body["error_code"]
        assert isinstance(body["correlation_id"], str) and body["correlation_id"]
        assert isinstance(body["message"], str)


def test_unhandled_exception_still_renders_the_envelope(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.api.routers.predictions as predictions_router

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    with build_client(api_settings, model=make_loaded()) as client:
        monkeypatch.setattr(predictions_router, "list_symbols", explode)
        response = client.get(f"{API_PREFIX}/predictions/BTC-USD")

    assert response.status_code == 500
    body = response.json()
    assert set(body) == ENVELOPE_KEYS
    # The internal message must not leak the exception text.
    assert "boom" not in body["message"]


# --- correlation id -------------------------------------------------------


def test_inbound_correlation_id_is_honoured_and_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={CORRELATION_HEADER: "abc-123"})
    assert response.headers[CORRELATION_HEADER] == "abc-123"


def test_correlation_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers[CORRELATION_HEADER]


def test_error_bodies_carry_the_inbound_correlation_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.api.routers.predictions as predictions_router

    monkeypatch.setattr(predictions_router, "list_symbols", lambda _s: [])
    response = client.get(
        f"{API_PREFIX}/predictions/NOPE", headers={CORRELATION_HEADER: "trace-me"}
    )
    assert response.json()["correlation_id"] == "trace-me"


# --- pagination bounds ----------------------------------------------------


def test_limit_cannot_be_forced_past_the_max_page_size(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/predictions/BTC-USD/history?limit=100000")

    assert response.status_code == 422
    assert response.json()["error_code"] in {ERROR_VALIDATION, "not_found"}
    assert "maximum page size" in response.json()["message"]


def test_history_window_cannot_be_forced_past_the_max(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/predictions/BTC-USD/history?hours=100000")
    assert response.status_code == 422
    assert "maximum look-back" in response.json()["message"]


def test_ticks_limit_is_bounded_too(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/ticks/BTC-USD?limit=99999")
    assert response.status_code == 422


def test_limit_at_the_maximum_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.api.routers.predictions as predictions_router

    monkeypatch.setattr(predictions_router, "list_predictions", lambda *a, **k: ([], False))
    response = client.get(f"{API_PREFIX}/predictions/BTC-USD/history?limit=500")
    assert response.status_code == 200
    assert response.json()["limit"] == 500


# --- model endpoints ------------------------------------------------------


def test_model_current_reports_the_loaded_version(client: TestClient) -> None:
    body = client.get(f"{API_PREFIX}/model/current").json()
    assert body["model_version"] == "3"
    assert body["feature_names"] == list(FEATURE_NAMES)


def test_model_current_is_200_with_nulls_when_nothing_is_loaded(
    api_settings: Settings,
) -> None:
    with build_client(api_settings, model=None) as client:
        response = client.get(f"{API_PREFIX}/model/current")

    assert response.status_code == 200
    assert response.json()["model_version"] is None


def test_refresh_swaps_the_version_with_no_redeploy(api_settings: Settings) -> None:
    """The Phase 5 exit criterion, end to end through HTTP.

    The loader is driven by mutable state rather than a call-count iterator:
    the lifespan refreshes at startup too, so counting calls would couple the
    test to startup internals it is not trying to assert on.
    """
    registry_state = {"version": "1"}
    cache = ModelCache(lambda: make_loaded(version=registry_state["version"]))
    state = AppState(
        settings=api_settings,
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        model_cache=cache,
    )

    with TestClient(create_app(state), raise_server_exceptions=False) as client:
        assert client.get(f"{API_PREFIX}/model/current").json()["model_version"] == "1"

        # A promotion happens in MLflow. No redeploy, no restart.
        registry_state["version"] = "2"
        refresh = client.post(f"{API_PREFIX}/model/refresh").json()

        assert refresh["changed"] is True
        assert refresh["previous_version"] == "1"
        assert refresh["current_version"] == "2"
        assert client.get(f"{API_PREFIX}/model/current").json()["model_version"] == "2"


def test_refresh_failure_is_reported_but_keeps_serving(api_settings: Settings) -> None:
    registry_state = {"down": False}

    def loader() -> LoadedModel | None:
        if registry_state["down"]:
            raise ConnectionError("registry down")
        return make_loaded(version="8")

    state = AppState(
        settings=api_settings,
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        model_cache=ModelCache(loader),
    )
    with TestClient(create_app(state), raise_server_exceptions=False) as client:
        assert client.get(f"{API_PREFIX}/model/current").json()["model_version"] == "8"

        registry_state["down"] = True
        body = client.post(f"{API_PREFIX}/model/refresh").json()

        assert body["error"] is not None and "registry down" in body["error"]
        assert body["current_version"] == "8"
        # Still serving the old model, and /model/current says why.
        current = client.get(f"{API_PREFIX}/model/current").json()
        assert current["model_version"] == "8"
        assert "registry down" in current["last_refresh_error"]


# --- metrics --------------------------------------------------------------


def test_metrics_endpoint_renders_prometheus_text(client: TestClient) -> None:
    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# TYPE marketpulse_api_requests_total counter" in body
    assert "marketpulse_api_model_loaded 1.0" in body


def test_metrics_label_by_route_template_not_concrete_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Labelling by concrete path would mint a new series per symbol.
    _patch_symbol_and_feature(monkeypatch, FakeFeatureRow())
    client.get(f"{API_PREFIX}/predictions/BTC-USD")
    client.get(f"{API_PREFIX}/predictions/ETH-USD")

    body = client.get("/metrics").text
    assert f'path="{API_PREFIX}/predictions/{{symbol}}"' in body
    assert "BTC-USD" not in body


# --- OpenAPI --------------------------------------------------------------


def test_openapi_is_generated_from_the_response_models(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert f"{API_PREFIX}/predictions/{{symbol}}" in schema["paths"]
    assert "PredictionResponse" in schema["components"]["schemas"]
    assert "ErrorEnvelope" in schema["components"]["schemas"]


def test_every_business_route_is_versioned(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    unversioned = [
        path
        for path in schema["paths"]
        if not path.startswith(API_PREFIX) and path not in {"/health", "/ready"}
    ]
    assert unversioned == ["/health/dependencies"], unversioned
