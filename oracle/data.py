from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def fetch_bitcoin(symbol: str = "BTC-USD", lookback_days: int = 729) -> pd.DataFrame:
    """Download hourly candles from Yahoo Finance.

    yfinance is imported lazily so feature engineering and tests do not require a
    network client. Yahoo limits intraday history to roughly 730 days.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("Install ML dependencies with: pip install -r requirements-ml.txt") from exc

    frame = yf.download(
        symbol,
        period=f"{lookback_days}d",
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError(f"No hourly candles returned for {symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return validate_candles(frame)


def validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a candle frame before it enters the model."""
    if frame.empty:
        raise ValueError("Market data is empty")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}")
    if frame.index.has_duplicates:
        frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free momentum, trend, volatility, and volume features."""
    frame = validate_candles(frame)
    close, high, low, volume = frame.close, frame.high, frame.low, frame.volume
    log_close = np.log(close)
    returns = log_close.diff()
    features = pd.DataFrame(index=frame.index)

    for lag in (1, 2, 3, 6, 12, 24, 48):
        features[f"return_{lag}"] = log_close.diff(lag)
    features["volatility_12"] = returns.rolling(12).std()
    features["volatility_48"] = returns.rolling(48).std()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    features["rsi_14"] = (100 - 100 / (1 + gain / (loss + 1e-12))) / 100

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    features["macd"] = macd / close
    features["macd_histogram"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    mean_20, std_20 = close.rolling(20).mean(), close.rolling(20).std()
    features["bollinger_position"] = (close - mean_20) / (2 * std_20 + 1e-12)
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    features["atr_14"] = true_range.rolling(14).mean() / close
    features["ema_ratio"] = close.ewm(span=24, adjust=False).mean() / close.ewm(span=96, adjust=False).mean() - 1
    features["volume_zscore"] = (volume - volume.rolling(48).mean()) / (volume.rolling(48).std() + 1e-12)
    features["hour_sin"] = np.sin(2 * np.pi * features.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features.index.hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * features.index.dayofweek / 7)
    return features.replace([np.inf, -np.inf], np.nan)


DataProvider = Callable[[str, int], pd.DataFrame]
