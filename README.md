# BTC Oracle Web

A responsive Flask dashboard for confidence-weighted Bitcoin market forecasts. This first web milestone includes the complete UI, API contract, interactive forecast horizons, custom canvas chart, model weighting view, and retraining lifecycle.

> The current API serves deterministic demo forecasts so the web experience works before the ML engine is connected. It is clearly labelled in the interface.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://localhost:5000>.

## API

- `GET /api/status` — model status and training progress
- `GET /api/signal` — market history and all forecast horizons
- `POST /api/retrain` — start a non-blocking model refresh
- `GET /health` — health check

## Structure

```text
app.py                  Flask app and temporary demo provider
static/css/app.css      responsive dashboard styles
static/js/app.js        rendering, chart, and API integration
templates/index.html    semantic dashboard markup
tests/test_app.py       route and API contract tests
```

## ML core

The first machine-learning core is available in `oracle/`:

- `data.py` validates hourly candles and computes 20 leak-free technical/time features.
- `models.py` builds a boosted tree, random forest, and neural sequence model. XGBoost and PyTorch are preferred when installed; explicitly named sklearn fallbacks keep development portable.
- `engine.py` performs a purged, chronological 80/20 split per horizon, derives confidence weights from directional accuracy and RMSE, and generates calls and 80% price intervals.
- Training runs in a background thread and model bundles are replaced atomically.

The website stays in demo mode by default. To run the real engine:

```bash
pip install -r requirements-ml.txt
BTC_ORACLE_MODE=ml python app.py
```

Live mode downloads up to 729 days of hourly BTC-USD candles and starts training immediately. Yahoo Finance provides deep history, while Coinbase overlays the latest exchange candles and acts as a full-history failover. The browser requests a new snapshot every minute; the engine refreshes candles every five minutes without blocking requests and keeps serving the last verified data if a provider is temporarily unavailable. The API exposes `dataSource`, `dataAgeSeconds`, and `refreshWarning` so freshness is observable.

The API status endpoint reports progress or a detailed error. Installing PyTorch is optional and documented in `requirements-ml.txt`; without it, the API honestly reports the `Neural Network` fallback rather than labelling it LSTM.

## Hosting with live data

A production `Dockerfile` and Render Blueprint are included. Both set `BTC_ORACLE_MODE=ml` and run one Gunicorn worker with four threads. A single worker is intentional: it avoids training duplicate model ensembles and wasting memory.

### Docker

```bash
docker build -t btc-oracle .
docker run --rm -p 5000:5000 -e BTC_ORACLE_MODE=ml btc-oracle
```

For another hosting provider, use:

```bash
pip install -r requirements.txt
BTC_ORACLE_MODE=ml gunicorn --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:$PORT app:app
```

The host must allow outbound HTTPS to Yahoo Finance and Coinbase. If Yahoo is blocked, Coinbase automatically supplies the history. Do not deploy with multiple workers unless model training is moved to a separate shared worker service.

## Next milestone

Add artifact persistence and walk-forward backtesting with trading fees/slippage, then benchmark the ensemble before changing any signal thresholds.

## Disclaimer

Educational use only. Not financial advice.
