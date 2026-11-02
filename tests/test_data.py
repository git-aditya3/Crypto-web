import numpy as np
import pandas as pd
import pytest

from oracle import data
from oracle.data import add_features, validate_candles


def candles(rows=240):
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series(50_000 + np.linspace(0, 1800, rows) + np.sin(np.arange(rows) / 8) * 100, index=index)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 120,
        "low": close - 110,
        "close": close,
        "volume": 1000 + np.arange(rows) % 37 * 12,
    }, index=index)


def test_features_are_finite_after_warmup_and_leak_free():
    frame = candles()
    full = add_features(frame)
    prefix = add_features(frame.iloc[:180])
    assert len(full.columns) == 20
    assert full.iloc[100:].notna().all().all()
    pd.testing.assert_series_equal(full.loc[prefix.index[-1]], prefix.iloc[-1])


def test_validation_rejects_non_positive_prices():
    frame = candles(20)
    frame.iloc[3, frame.columns.get_loc("close")] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_candles(frame)


def test_live_fetch_merges_fresh_coinbase_tail(monkeypatch):
    index = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=20), periods=20, freq="h")
    close = pd.Series(np.linspace(90_000, 91_000, 20), index=index)
    yahoo = pd.DataFrame(
        {"open": close, "high": close + 10, "low": close - 10, "close": close, "volume": 1000},
        index=index,
    )
    recent = yahoo.iloc[-5:].copy()
    recent.loc[recent.index[-1], "close"] = 92_000
    monkeypatch.setattr(data, "_fetch_yahoo", lambda symbol, days: yahoo)
    monkeypatch.setattr(data, "_fetch_coinbase", lambda symbol, days, full_history=False: recent)

    result = data.fetch_bitcoin()
    assert result.close.iloc[-1] == 92_000
    assert result.attrs["source"] == "Yahoo Finance + Coinbase"


def test_live_fetch_reports_all_provider_failures(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("provider offline")

    monkeypatch.setattr(data, "_fetch_yahoo", fail)
    monkeypatch.setattr(data, "_fetch_coinbase", fail)
    with pytest.raises(RuntimeError, match="All market-data providers failed"):
        data.fetch_bitcoin()
