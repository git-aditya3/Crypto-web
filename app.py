"""BTC Oracle web application.

The demo data provider keeps the web experience usable while the ML engine is being
built. Replace ``DemoOracle`` with an adapter exposing ``snapshot()`` and
``retrain()`` when the prediction engine is ready.
"""
from __future__ import annotations

import math
import os
import random
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import Flask, jsonify, render_template

app = Flask(__name__)


@dataclass(frozen=True)
class Horizon:
    name: str
    hours: int
    move: float
    confidence: float
    call: str


class DemoOracle:
    """Deterministic market snapshot used by the first web milestone."""

    horizons = (
        Horizon("4 hours", 4, 0.0084, 0.72, "BUY"),
        Horizon("24 hours", 24, 0.0218, 0.68, "BUY"),
        Horizon("3 days", 72, -0.0136, 0.54, "HOLD"),
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = {"status": "ready", "message": "Models online", "progress": 100}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def retrain(self) -> bool:
        with self._lock:
            if self._status["status"] == "training":
                return False
            self._status = {"status": "training", "message": "Refreshing market models", "progress": 12}
        threading.Thread(target=self._finish_retrain, daemon=True).start()
        return True

    def _finish_retrain(self) -> None:
        # Short staged task demonstrates the API lifecycle without blocking requests.
        import time
        for progress, message in ((38, "Preparing features"), (67, "Validating ensemble"), (91, "Building forecast")):
            time.sleep(0.7)
            with self._lock:
                self._status = {"status": "training", "message": message, "progress": progress}
        time.sleep(0.5)
        with self._lock:
            self._status = {"status": "ready", "message": "Models online", "progress": 100}

    def snapshot(self) -> dict[str, Any]:
        rng = random.Random(21000)
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=167)
        price = 102_430.0
        points: list[dict[str, Any]] = []
        current = 98_180.0
        for index in range(168):
            cycle = math.sin(index / 9) * 110 + math.sin(index / 25) * 190
            current += rng.gauss(24, 155) + cycle * 0.045
            points.append({"time": (start + timedelta(hours=index)).isoformat(), "price": round(current, 2)})
        # Keep the display's final quote and chart perfectly aligned.
        shift = price - points[-1]["price"]
        for point in points:
            point["price"] = round(point["price"] + shift, 2)

        signals = []
        model_specs = (
            ("XGBoost", 0.42, 0.573),
            ("Random Forest", 0.34, 0.551),
            ("LSTM", 0.24, 0.538),
        )
        for h in self.horizons:
            target = price * (1 + h.move)
            spread = price * (0.012 + h.hours / 72 * 0.019)
            models = [
                {
                    "name": name,
                    "weight": weight,
                    "accuracy": accuracy,
                    "target": round(target * (1 + (i - 1) * 0.0032), 2),
                }
                for i, (name, weight, accuracy) in enumerate(model_specs)
            ]
            signals.append({
                "id": h.name.split()[0] + ("h" if h.hours < 72 else "d"),
                "label": h.name,
                "hours": h.hours,
                "call": h.call,
                "confidence": h.confidence,
                "expectedMove": h.move,
                "target": round(target, 2),
                "low": round(target - spread, 2),
                "high": round(target + spread, 2),
                "end": (now + timedelta(hours=h.hours)).isoformat(),
                "validationAccuracy": 0.559,
                "models": models,
            })
        return {
            "mode": "demo",
            "symbol": "BTC / USD",
            "dataSource": "Deterministic demo feed",
            "dataAgeSeconds": 0,
            "refreshWarning": None,
            "price": price,
            "change24h": 2.41,
            "asOf": now.isoformat(),
            "history": points,
            "signals": signals,
        }


def create_oracle():
    """Select demo or live ML mode without coupling the web layer to training."""
    if os.getenv("BTC_ORACLE_MODE", "demo").lower() != "ml":
        return DemoOracle()
    from oracle import OracleEngine

    engine = OracleEngine()
    engine.retrain()
    return engine


oracle = create_oracle()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify(oracle.status())


@app.get("/api/signal")
def api_signal():
    if oracle.status()["status"] != "ready":
        return jsonify({"error": "Forecast is being refreshed"}), 503
    try:
        return jsonify(oracle.snapshot())
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/retrain")
def api_retrain():
    started = oracle.retrain()
    return jsonify({"started": started, **oracle.status()}), 202 if started else 200


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
