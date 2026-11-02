from app import app


def test_home_renders():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"BTC Oracle" in response.data


def test_signal_contract():
    client = app.test_client()
    response = client.get("/api/signal")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["symbol"] == "BTC / USD"
    assert len(payload["history"]) == 168
    assert len(payload["signals"]) == 3
    assert sum(model["weight"] for model in payload["signals"][0]["models"]) == 1


def test_health():
    response = app.test_client().get("/health")
    assert response.get_json() == {"ok": True}
