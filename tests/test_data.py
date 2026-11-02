import numpy as np
import pandas as pd
import pytest

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
