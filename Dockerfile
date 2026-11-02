FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BTC_ORACLE_MODE=ml \
    MARKET_REFRESH_SECONDS=300 \
    MODEL_RETRAIN_SECONDS=21600 \
    PORT=5000

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app
COPY --chown=app:app . .
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '5000') + '/health', timeout=3)"

CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 4 --timeout 600 --graceful-timeout 30 --keep-alive 5 --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - --bind 0.0.0.0:${PORT} app:app"]
