import numpy as np
import pandas as pd

from oracle.config import OracleConfig
from oracle.engine import OracleEngine


class MeanRegressor:
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def fit(self, features, target):
        self.value = float(np.mean(target)) * self.multiplier
        return self

    def predict(self, features):
        return np.full(len(features), self.value)


def synthetic_candles(rows=900):
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    trend = np.arange(rows) * 18
    close = pd.Series(70_000 + trend + np.sin(np.arange(rows) / 13) * 250, index=index)
    return pd.DataFrame({
        "open": close.shift().fillna(close.iloc[0]),
        "high": close + 180,
        "low": close - 170,
        "close": close,
        "volume": 2000 + (np.arange(rows) % 50) * 20,
    }, index=index)


def synthetic_provider(symbol, days):
    return synthetic_candles()


def model_factory():
    return {"Fast": MeanRegressor(.8), "Balanced": MeanRegressor(1), "Slow": MeanRegressor(1.2)}


def test_engine_trains_weighted_ensemble_and_serves_contract():
    config = OracleConfig(
        horizons={"4 hours": 4, "24 hours": 24},
        sequence_length=12,
        minimum_training_rows=200,
        refresh_seconds=3600,
    )
    engine = OracleEngine(config, synthetic_provider, model_factory)
    engine.train()
    status = engine.status()
    assert status["status"] == "ready"
    assert status["serving"] is True
    assert status["modelTrainedAt"]
    assert status["modelDataAsOf"]
    assert status["nextRetrainAt"]

    result = engine.snapshot()
    assert result["mode"] == "live"
    assert result["predictionGeneratedAt"]
    assert result["modelDataAsOf"] == status["modelDataAsOf"]
    assert len(result["history"]) == 168
    assert len(result["signals"]) == 2
    for signal in result["signals"]:
        assert signal["call"] in {"BUY", "SELL", "HOLD"}
        assert 0 <= signal["confidence"] <= 1
        assert signal["low"] < signal["target"] < signal["high"]
        assert abs(sum(model["weight"] for model in signal["models"]) - 1) < 1e-10
        assert signal["samples"]["train"] > signal["samples"]["validation"]


def test_retraining_consumes_new_candles_and_replaces_predictions():
    rows = {"value": 900}

    def evolving_provider(symbol, days):
        return synthetic_candles(rows["value"])

    config = OracleConfig(
        horizons={"4 hours": 4},
        sequence_length=12,
        minimum_training_rows=200,
        refresh_seconds=3600,
    )
    engine = OracleEngine(config, evolving_provider, model_factory)
    engine.train()
    first = engine.snapshot()

    rows["value"] += 24
    engine.train()
    second = engine.snapshot()

    assert second["modelDataAsOf"] > first["modelDataAsOf"]
    assert second["price"] != first["price"]
    assert second["predictionGeneratedAt"] >= first["predictionGeneratedAt"]
