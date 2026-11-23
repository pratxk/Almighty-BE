#!/bin/sh
# Run the web server and the RQ ingest worker in this one container.
#
# Why together: Render's free tier gives 750 instance-hours a month and
# the two web services already use ~660 of them — a second worker
# service would not fit. The worker only embeds and upserts (network
# waits, tiny memory), so sharing the 512 MB container is fine.
#
# The worker reads the same env as the web server (DATABASE_URL,
# REDIS_URL, GEMINI_API_KEY, QDRANT_URL, ...). If Redis is down the
# worker exits and the web server still serves chat — jobs stay queued
# in Postgres until a worker is back.
set -e
cd "$(dirname "$0")"

rq worker --url "$REDIS_URL" --path . ingest &
# exec makes uvicorn PID 1, so Render's stop signal reaches it; the
# worker dies with the container, and any queued job is picked up
# again on the next boot (its row and bytes are in Postgres).
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
