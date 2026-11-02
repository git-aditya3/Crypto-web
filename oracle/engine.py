from __future__ import annotations

import threading
import time
from datetime import timedelta
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from .config import OracleConfig
from .data import DataProvider, add_features, fetch_bitcoin, validate_candles
from .models import ModelFactory, build_models


class OracleEngine:
    """Train, validate, and serve a confidence-weighted model ensemble."""

    def __init__(
        self,
        config: OracleConfig | None = None,
        data_provider: DataProvider = fetch_bitcoin,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or OracleConfig()
        self.data_provider = data_provider
        self.model_factory = model_factory or partial(
            build_models,
            sequence_length=self.config.sequence_length,
            random_state=self.config.random_state,
        )
        self._lock = threading.RLock()
        self._training_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._bundles: dict[str, dict[str, Any]] = {}
        self._candles: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._fetched_at = 0.0
        self._last_refresh_error: str | None = None
        self._maintenance_started = False
        self._status: dict[str, Any] = {
            "status": "idle", "message": "Models have not been trained", "progress": 0
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _set_status(self, status: str, message: str, progress: int) -> None:
        with self._lock:
            self._status = {"status": status, "message": message, "progress": progress}

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        candles = validate_candles(
            self.data_provider(self.config.symbol, self.config.lookback_days).copy()
        )
        features = add_features(candles)
        return candles, features

    def train(self) -> None:
        """Train all horizons synchronously and atomically publish the result."""
        if not self._training_lock.acquire(blocking=False):
            return
        try:
            self._set_status("training", "Downloading hourly BTC candles", 5)
            candles, features = self._load_data()
            log_price = np.log(candles.close)
            bundles: dict[str, dict[str, Any]] = {}
            horizon_count = len(self.config.horizons)

            for horizon_index, (label, hours) in enumerate(self.config.horizons.items()):
                base_progress = 12 + int(76 * horizon_index / horizon_count)
                self._set_status("training", f"Training {label} models", base_progress)
                target = log_price.shift(-hours) - log_price
                dataset = features.assign(target=target).dropna()
                if len(dataset) < self.config.minimum_training_rows:
                    raise ValueError(
                        f"Only {len(dataset)} usable rows for {label}; "
                        f"at least {self.config.minimum_training_rows} are required"
                    )
                values = dataset[features.columns].to_numpy(dtype=np.float64)
                expected = dataset.target.to_numpy(dtype=np.float64)
                split = int(len(values) * (1 - self.config.validation_fraction))
                train_end = split - hours  # purge observations whose target overlaps validation
                if train_end <= self.config.sequence_length:
                    raise ValueError(f"Not enough purged training data for {label}")
                train_x, train_y = values[:train_end], expected[:train_end]
                validation_x, validation_y = values[split:], expected[split:]

                fitted: dict[str, Any] = {}
                predictions: dict[str, np.ndarray] = {}
                metrics: dict[str, dict[str, float]] = {}
                errors: list[str] = []
                for model_name, model in self.model_factory().items():
                    try:
                        model.fit(train_x, train_y)
                        prediction = np.asarray(model.predict(validation_x), dtype=float).reshape(-1)
                        if len(prediction) != len(validation_y) or not np.isfinite(prediction).all():
                            raise ValueError("model returned invalid predictions")
                        rmse = float(np.sqrt(np.mean(np.square(prediction - validation_y))))
                        accuracy = float(np.mean(np.sign(prediction) == np.sign(validation_y)))
                        fitted[model_name] = model
                        predictions[model_name] = prediction
                        metrics[model_name] = {"rmse": max(rmse, 1e-12), "accuracy": accuracy}
                    # Third-party estimators can raise library-specific exceptions;
                    # isolate any failed estimator so healthy models can still train.
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
                if len(fitted) < 2:
                    detail = "; ".join(errors) or "model factory returned fewer than two models"
                    raise RuntimeError(f"Could not build an ensemble for {label}: {detail}")

                raw_weights = {
                    name: max(metric["accuracy"] - 0.5, 0.01) / metric["rmse"]
                    for name, metric in metrics.items()
                }
                total_weight = sum(raw_weights.values())
                weights = {name: value / total_weight for name, value in raw_weights.items()}
                ensemble = sum(weights[name] * predictions[name] for name in fitted)
                bundles[label] = {
                    "hours": hours,
                    "models": fitted,
                    "weights": weights,
                    "metrics": metrics,
                    "sigma": max(float(np.std(ensemble - validation_y)), 1e-8),
                    "accuracy": float(np.mean(np.sign(ensemble) == np.sign(validation_y))),
                    "warnings": errors,
                    "samples": {"train": len(train_x), "validation": len(validation_x)},
                }

            with self._lock:
                self._candles = candles
                self._features = features
                self._bundles = bundles
                self._fetched_at = time.time()
                self._last_refresh_error = None
            self._set_status("ready", "Models online", 100)
            self._start_market_maintenance()
        except Exception as exc:
            self._set_status("error", f"{type(exc).__name__}: {exc}", 0)
            raise
        finally:
            self._training_lock.release()

    def retrain(self) -> bool:
        """Start non-blocking training for web/API clients."""
        if self._training_lock.locked() or self.status()["status"] == "training":
            return False
        self._set_status("training", "Training queued", 1)

        def run() -> None:
            try:
                self.train()
            except Exception:  # noqa: BLE001
                # train() already publishes the detailed failure through status().
                return

        threading.Thread(target=run, name="btc-oracle-training", daemon=True).start()
        return True

    def _start_market_maintenance(self) -> None:
        """Keep market candles current even when no browser session is open."""
        with self._lock:
            if self._maintenance_started:
                return
            self._maintenance_started = True

        def maintain() -> None:
            while True:
                time.sleep(max(self.config.refresh_seconds, 30))
                self._refresh_if_stale()

        threading.Thread(target=maintain, name="btc-market-maintenance", daemon=True).start()

    def _refresh_if_stale(self) -> None:
        """Refresh candles in the background while continuing to serve cached data."""
        with self._lock:
            stale = time.time() - self._fetched_at > self.config.refresh_seconds
        if not stale or self._refresh_lock.locked() or self._training_lock.locked():
            return

        def refresh() -> None:
            if not self._refresh_lock.acquire(blocking=False):
                return
            try:
                candles, features = self._load_data()
                with self._lock:
                    self._candles, self._features = candles, features
                    self._fetched_at = time.time()
                    self._last_refresh_error = None
            except Exception as exc:  # noqa: BLE001 - stale cache remains available
                with self._lock:
                    self._last_refresh_error = f"{type(exc).__name__}: {exc}"
                    # Back off before another request retries a failing provider.
                    self._fetched_at = time.time() - self.config.refresh_seconds + 60
            finally:
                self._refresh_lock.release()

        threading.Thread(target=refresh, name="btc-market-refresh", daemon=True).start()

    def snapshot(self) -> dict[str, Any]:
        """Generate live calls without refitting the validated model bundles."""
        with self._lock:
            if not self._bundles:
                raise RuntimeError("Models are not ready")
        self._refresh_if_stale()
        with self._lock:
            candles = self._candles.copy()  # type: ignore[union-attr]
            features = self._features.copy()  # type: ignore[union-attr]
            bundles = dict(self._bundles)
            refresh_error = self._last_refresh_error

        recent_features = features.dropna().iloc[-self.config.sequence_length :]
        if recent_features.empty:
            raise RuntimeError("No complete feature row is available")
        price = float(candles.close.iloc[-1])
        as_of = candles.index[-1]
        day_reference = float(candles.close.iloc[-25]) if len(candles) >= 25 else float(candles.close.iloc[0])
        signals: list[dict[str, Any]] = []

        for label, bundle in bundles.items():
            model_returns = {
                name: float(np.asarray(model.predict(recent_features.to_numpy(dtype=float))).reshape(-1)[-1])
                for name, model in bundle["models"].items()
            }
            expected_return = sum(bundle["weights"][name] * value for name, value in model_returns.items())
            direction = np.sign(expected_return)
            agreement = sum(
                bundle["weights"][name]
                for name, value in model_returns.items()
                if np.sign(value) == direction
            )
            confidence = float(np.clip(agreement * bundle["accuracy"], 0, 1))
            enough_edge = abs(expected_return) >= self.config.minimum_signal_sigma * bundle["sigma"]
            if not enough_edge or confidence < self.config.minimum_confidence:
                call = "HOLD"
            else:
                call = "BUY" if expected_return > 0 else "SELL"
            target = price * np.exp(expected_return)
            low = price * np.exp(expected_return - self.config.interval_z * bundle["sigma"])
            high = price * np.exp(expected_return + self.config.interval_z * bundle["sigma"])
            hours = bundle["hours"]
            signals.append({
                "id": f"{hours}h" if hours < 72 else f"{hours // 24}d",
                "label": label,
                "hours": hours,
                "call": call,
                "confidence": confidence,
                "expectedMove": float(np.expm1(expected_return)),
                "target": round(float(target), 2),
                "low": round(float(low), 2),
                "high": round(float(high), 2),
                "end": (as_of + timedelta(hours=hours)).isoformat(),
                "validationAccuracy": bundle["accuracy"],
                "samples": bundle["samples"],
                "models": [
                    {
                        "name": name,
                        "weight": bundle["weights"][name],
                        "accuracy": bundle["metrics"][name]["accuracy"],
                        "rmse": bundle["metrics"][name]["rmse"],
                        "target": round(float(price * np.exp(model_returns[name])), 2),
                    }
                    for name in bundle["models"]
                ],
            })

        history = candles.close.iloc[-self.config.history_hours :]
        return {
            "mode": "live",
            "symbol": self.config.symbol.replace("-", " / "),
            "dataSource": candles.attrs.get("source", "Configured market provider"),
            "dataAgeSeconds": max(0, int((pd.Timestamp.now(tz="UTC") - as_of).total_seconds())),
            "refreshWarning": refresh_error,
            "price": round(price, 2),
            "change24h": round((price / day_reference - 1) * 100, 2),
            "asOf": as_of.isoformat(),
            "history": [
                {"time": timestamp.isoformat(), "price": round(float(value), 2)}
                for timestamp, value in history.items()
            ],
            "signals": signals,
        }
