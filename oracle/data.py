from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return validate_candles(frame)


def _fetch_yahoo(symbol: str, lookback_days: int, attempts: int = 2) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on runtime package
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            frame = yf.download(
                symbol,
                period=f"{lookback_days}d",
                interval="1h",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20,
            )
            if frame.empty:
                raise RuntimeError(f"No hourly candles returned for {symbol}")
            return _normalize_frame(frame)
        except Exception as exc:  # noqa: BLE001 - yfinance exposes varied transport errors
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Yahoo Finance failed after {attempts} attempts: {last_error}") from last_error


def _coinbase_page(product: str, start: datetime, end: datetime) -> pd.DataFrame:
    query = urlencode({"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()})
    request = Request(
        f"{COINBASE_CANDLES_URL.format(product=product)}?{query}",
        headers={"User-Agent": "Team-A-BTC/1.0", "Accept": "application/json"},
    )
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Coinbase returned no candles")
    # Coinbase rows: [time, low, high, open, close, volume].
    frame = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
    frame.index = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    return _normalize_frame(frame)


def _fetch_coinbase(symbol: str, lookback_days: int, full_history: bool = False) -> pd.DataFrame:
    product = symbol
    if product != "BTC-USD":
        raise ValueError("Coinbase fallback currently supports BTC-USD only")
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    requested_hours = lookback_days * 24 if full_history else min(250, lookback_days * 24)
    start_limit = end - timedelta(hours=requested_hours)
    pages: list[pd.DataFrame] = []
    page_end = end
    # Coinbase permits at most 300 candles per request; 290 avoids boundary ambiguity.
    while page_end > start_limit:
        page_start = max(start_limit, page_end - timedelta(hours=290))
        pages.append(_coinbase_page(product, page_start, page_end))
        page_end = page_start
        if full_history and page_end > start_limit:
            time.sleep(0.12)  # stay below the public REST burst limit
    frame = pd.concat(pages).sort_index()
    return validate_candles(frame.loc[~frame.index.duplicated(keep="last")])


def fetch_bitcoin(symbol: str = "BTC-USD", lookback_days: int = 729) -> pd.DataFrame:
    """Fetch hourly candles with provider failover and a fresh exchange tail.

    Yahoo supplies deep history efficiently. Coinbase supplies the latest exchange
    candles and becomes a full-history fallback if Yahoo is unavailable. This
    arrangement avoids a single market-data provider becoming a production outage.
    """
    yahoo_error: Exception | None = None
    try:
        history = _fetch_yahoo(symbol, lookback_days)
        source = "Yahoo Finance"
    except Exception as exc:  # noqa: BLE001 - fail over on any provider failure
        yahoo_error = exc
        try:
            history = _fetch_coinbase(symbol, lookback_days, full_history=True)
            source = "Coinbase (fallback)"
        except Exception as coinbase_error:
            raise RuntimeError(
                f"All market-data providers failed. Yahoo: {yahoo_error}; Coinbase: {coinbase_error}"
            ) from coinbase_error

    tail_error: Exception | None = None
    try:
        recent = _fetch_coinbase(symbol, lookback_days, full_history=False)
        history = pd.concat([history, recent]).sort_index()
        history = history.loc[~history.index.duplicated(keep="last")]
        source = f"{source} + Coinbase"
    except Exception as exc:  # noqa: BLE001 - deep history remains valid
        tail_error = exc

    history = validate_candles(history)
    age = datetime.now(UTC) - history.index[-1].to_pydatetime()
    if age > timedelta(hours=4):
        raise RuntimeError(f"Latest {source} candle is stale ({age.total_seconds() / 3600:.1f} hours old)")
    history.attrs["source"] = source
    errors = [str(error) for error in (yahoo_error, tail_error) if error]
    history.attrs["provider_error"] = "; ".join(errors) or None
    return history


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
