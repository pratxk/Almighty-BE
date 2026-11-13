FROM python:3.12-slim

WORKDIR /app

# Build deps for sentence-transformers / numpy wheels if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Presidio needs a spaCy NER model for real PII detection; without it
# the guardrails silently fall back to regex-only (that is what the
# live Render deploy reports as pii_engine "regex"). Download the
# small English model into the image so Presidio runs in production.
RUN python -m spacy download en_core_web_sm

COPY app ./app
COPY .env.example ./.env.example

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
