FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_TRAINING_ENABLED=false

# LightGBM a besoin d'OpenMP (absent de l'image slim)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
# Render fournit $PORT ; un seul worker (état en mémoire)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]