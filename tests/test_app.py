import app as app_module

app = app_module.app


def test_home_renders_with_production_security_headers():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Team-A" in response.data
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-cache"


def test_signal_contract_and_api_cache_policy():
    response = app.test_client().get("/api/signal")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.get_json()
    assert payload["symbol"] == "BTC / USD"
    assert len(payload["history"]) == 168
    assert len(payload["signals"]) == 3
    assert sum(model["weight"] for model in payload["signals"][0]["models"]) == 1


def test_status_exposes_operational_metadata():
    payload = app.test_client().get("/api/status").get_json()
    assert payload["serving"] is True
    assert payload["version"] == app_module.APP_VERSION
    assert payload["mode"] in {"demo", "ml"}


def test_health_and_readiness():
    client = app.test_client()
    assert client.get("/health").get_json()["ok"] is True
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.get_json()["ready"] is True


def test_retrain_requires_admin_token_when_configured(monkeypatch):
    monkeypatch.setattr(app_module, "ADMIN_TOKEN", "production-secret")
    response = app.test_client().post("/api/retrain")
    assert response.status_code == 401
    assert response.get_json()["error"] == "A valid admin token is required"


def test_unknown_api_route_is_json():
    response = app.test_client().get("/api/missing")
    assert response.status_code == 404
    assert response.is_json


def test_vercel_web_layer_proxies_to_persistent_backend(monkeypatch):
    class BackendResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b'{"status":"ready","serving":true,"mode":"ml"}'

    captured = {}

    def fake_urlopen(upstream, timeout):
        captured["url"] = upstream.full_url
        captured["timeout"] = timeout
        return BackendResponse()

    monkeypatch.setattr(app_module, "BACKEND_URL", "https://ml.example.com")
    monkeypatch.setattr(app_module, "urlopen", fake_urlopen)
    response = app.test_client().get("/api/status")

    assert response.status_code == 200
    assert response.get_json()["mode"] == "ml"
    assert captured == {"url": "https://ml.example.com/api/status", "timeout": 25}
