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

## Next milestone

Implement the data/features/models engine from `BTC_Oracle_Project.md`, then replace `DemoOracle` with an engine adapter that exposes the same `status()`, `snapshot()`, and `retrain()` interface.

## Disclaimer

Educational use only. Not financial advice.
