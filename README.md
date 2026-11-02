# Team-A Bitcoin Market Intelligence

A production-packaged Flask dashboard for confidence-weighted Bitcoin market forecasts. It includes a responsive interface, resilient live-data providers, an ML ensemble, operational health checks, protected retraining, container deployment, and automated CI.

> Local development uses clearly labelled deterministic demo forecasts by default. Hosted configurations enable the live ML engine.

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
- `POST /api/retrain` — start a non-blocking model refresh; requires `X-Admin-Token` when configured
- `GET /health` and `GET /health/live` — process liveness checks
- `GET /health/ready` — model/data readiness check

## Structure

```text
app.py                  Flask app, API, security headers, and health checks
oracle/                 data providers, features, models, and ensemble engine
static/                 responsive styles and browser application
templates/index.html    semantic dashboard markup
tests/                  API, data, feature, and engine tests
Dockerfile              non-root production container with health check
render.yaml             live-mode Render deployment blueprint
```

## ML core

The first machine-learning core is available in `oracle/`:

- `data.py` validates hourly candles and computes 20 leak-free technical/time features.
- `models.py` builds a boosted tree, random forest, and neural sequence model. XGBoost and PyTorch are preferred when installed; explicitly named sklearn fallbacks keep development portable.
- `engine.py` performs a purged, chronological 80/20 split per horizon, derives confidence weights from directional accuracy and RMSE, and generates calls and 80% price intervals.
- Training runs in a background thread, model bundles are replaced atomically, and hosted models retrain every six hours on the newest candles while the previous bundle continues serving.

The website stays in demo mode by default. To run the real engine:

```bash
pip install -r requirements-ml.txt
BTC_ORACLE_MODE=ml python app.py
```

Live mode downloads up to 729 days of hourly BTC-USD candles and starts training immediately. Yahoo Finance provides deep history, while Coinbase overlays the latest exchange candles and acts as a full-history failover. The engine refreshes candles every five minutes even when nobody has the dashboard open. Each API snapshot runs the trained ensemble against the newest feature window, and the browser requests a new prediction every minute. Every six hours the engine refits and validates all horizon models on the expanded history, then atomically replaces the previous bundles. A failed refresh or retraining attempt leaves the last verified data and models online.

The API exposes `dataSource`, `dataAgeSeconds`, `marketUpdatedAt`, `modelTrainedAt`, `modelDataAsOf`, `nextRetrainAt`, `predictionGeneratedAt`, and `refreshWarning` so the complete live-data and self-training lifecycle is observable.

The API status endpoint reports progress or a detailed error. Installing PyTorch is optional and documented in `requirements-ml.txt`; without it, the API honestly reports the `Neural Network` fallback rather than labelling it LSTM.

## Hosting with live data

A production `Dockerfile` and Render Blueprint are included. Both set `BTC_ORACLE_MODE=ml` and run one Gunicorn worker with four threads. A single worker is intentional: it avoids training duplicate model ensembles and wasting memory.

### Docker

```bash
docker build -t team-a-btc .
docker run --rm -p 5000:5000 \
  -e BTC_ORACLE_MODE=ml \
  -e TEAM_A_ADMIN_TOKEN="$(openssl rand -hex 32)" \
  team-a-btc
```

For another hosting provider, use:

```bash
pip install -r requirements.txt
BTC_ORACLE_MODE=ml gunicorn --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:$PORT app:app
```

The host must allow outbound HTTPS to Yahoo Finance and Coinbase. If Yahoo is blocked, Coinbase automatically supplies the history. Do not deploy with multiple workers unless model training is moved to a separate shared worker service.

Set a long random `TEAM_A_ADMIN_TOKEN` in production. The Retrain button prompts administrators for it and stores it only in that browser tab's session storage. Render generates the secret automatically. Security headers, no-store API caching, stale-while-refresh market data, liveness/readiness endpoints, graceful Gunicorn recycling, and a non-root container user are enabled by default.

## Quality checks

```bash
pip install -r requirements-dev.txt
ruff check app.py oracle tests
pytest -q
node --check static/js/app.js
```

GitHub Actions runs the Python checks for every push and pull request.

## Next milestone

Add artifact persistence and walk-forward backtesting with trading fees/slippage, then benchmark the ensemble before changing any signal thresholds.

## Disclaimer

Educational use only. Not financial advice.
