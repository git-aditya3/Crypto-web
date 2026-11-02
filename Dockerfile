FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BTC_ORACLE_MODE=ml \
    PORT=5000

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:${PORT} app:app"]
