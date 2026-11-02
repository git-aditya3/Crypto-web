# Deploy Team-A as a Web Application

Team-A is packaged for Vercel, Docker, Render, and any Python host that supports a Procfile. Python 3.12 is used consistently across all deployment targets.

## Required production settings

| Variable | Value | Purpose |
|---|---:|---|
| `BTC_ORACLE_MODE` | `ml` | Enables live data and trained predictions |
| `TEAM_A_ADMIN_TOKEN` | long random secret | Protects manual retraining |
| `MARKET_REFRESH_SECONDS` | `300` | Refreshes market candles every five minutes |
| `MODEL_RETRAIN_SECONDS` | `21600` | Retrains the ensemble every six hours |
| `LOG_LEVEL` | `INFO` | Production logging level |
| `PORT` | assigned by host | Web server port |

The host must permit outbound HTTPS access to Yahoo Finance and Coinbase. Use one web worker: each worker would otherwise train a separate in-memory ensemble.

## Option 1: Vercel web frontend + persistent ML backend

Vercel serverless functions cannot keep training threads alive between requests. The included `vercel.json` therefore deploys a lightweight Flask web function with only Flask installed from `uv.lock`. Live training must run on Render, Railway, Docker, or a VPS.

1. Deploy this repository's backend with Option 2 or Option 3 below.
2. Import the same repository into Vercel.
3. In Vercel project settings, add `TEAM_A_BACKEND_URL` with the public backend origin, for example `https://team-a-btc.onrender.com`.
4. Redeploy. Requests to `/api/status`, `/api/signal`, and `/api/retrain` are securely relayed by the Vercel function to the persistent service.

The repository pins Python `3.12` in `.python-version` and declares `requires-python = ">=3.12"` in `pyproject.toml`. `uv.lock` contains the matching lock metadata. Do not enable `BTC_ORACLE_MODE=ml` directly on Vercel; the persistent backend owns live data refresh and model training.

Without `TEAM_A_BACKEND_URL`, Vercel still deploys successfully in clearly labelled demo mode.

## Option 2: Render

1. Push this repository to GitHub.
2. In Render, select **New > Blueprint** and choose the repository.
3. Render reads `render.yaml`, installs dependencies, creates an admin token, and starts the web service.
4. Wait for `/health/ready` to return HTTP 200 after initial model training.

`/health` is the liveness endpoint and remains available while models train.

## Option 3: Docker

```bash
docker build -t team-a-btc .
docker run -d --name team-a-btc \
  -p 5000:5000 \
  -e TEAM_A_ADMIN_TOKEN="$(openssl rand -hex 32)" \
  --restart unless-stopped \
  team-a-btc
```

Or use Compose:

```bash
export TEAM_A_ADMIN_TOKEN="$(openssl rand -hex 32)"
docker compose up -d --build
```

Open `http://localhost:5000`. The container runs as a non-root user and has a built-in health check.

## Option 4: Generic Python host

Upload the deployment ZIP, set the required environment variables, and use:

```bash
pip install -r requirements.txt
gunicorn --workers 1 --threads 4 --timeout 600 --graceful-timeout 30 \
  --access-logfile - --error-logfile - --bind 0.0.0.0:$PORT app:app
```

Platforms that support a `Procfile` detect this command automatically.

## Production verification

```bash
curl -fsS https://YOUR-DOMAIN/health
curl -fsS https://YOUR-DOMAIN/health/ready
curl -fsS https://YOUR-DOMAIN/api/status
curl -fsS https://YOUR-DOMAIN/api/signal
```

The signal response should report `mode: live`, a recent `dataAgeSeconds`, an active `dataSource`, and timestamps for market updates, model training, model data, and prediction generation.

## Manual retraining

The website prompts for the admin token when **Retrain models** is selected. The API equivalent is:

```bash
curl -X POST https://YOUR-DOMAIN/api/retrain \
  -H "X-Admin-Token: $TEAM_A_ADMIN_TOKEN"
```

The token is never embedded in the application or returned by the API.
