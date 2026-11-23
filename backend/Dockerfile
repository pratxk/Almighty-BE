FROM python:3.12-slim

WORKDIR /app

# Build deps for sentence-transformers / numpy wheels if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: do NOT install the spaCy model (en_core_web_sm) here. It lets
# Presidio run, but loading it on Render's 512 MB free tier OOM-kills
# the container on the first chat request (crashed the live deploy
# 2026-11-14 ~03:35 IST). Without the model the guardrails run on the
# regex PII fallback, which is the correct setting for this tier.
# Presidio needs a bigger instance (2 GB+) to run safely.

COPY app ./app
COPY start.sh ./start.sh
COPY .env.example ./.env.example

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "start.sh"]
