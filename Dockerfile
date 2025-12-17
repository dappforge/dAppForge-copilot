# syntax=docker/dockerfile:1.7-labs

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Redis
RUN apt-get update && apt-get install -y --no-install-recommends redis-server \
    && rm -rf /var/lib/apt/lists/*
RUN sed -i 's/^supervised .*/supervised no/' /etc/redis/redis.conf
RUN mkdir -p /var/run/redis && chown redis:redis /var/run/redis

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements for layer caching
COPY requirements.api.txt ./

RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.api.txt

ENV PATH="/opt/venv/bin:$PATH"

# Copy only the necessary folders and files for the application
COPY api/ ./api/
COPY caching/ ./caching/
COPY feedback/ ./feedback/
COPY knowledge_graph_core/ ./knowledge_graph_core/
COPY utils/ ./utils/
COPY vector_data/ ./vector_data/

# Redis default envs (override in compose/infra)
ENV REDIS_HOST=localhost \
    REDIS_PORT=6379

# Create startup script (run uvicorn with multiple workers for better concurrency)
RUN echo '#!/bin/bash\nredis-server --daemonize yes --pidfile /var/run/redis/redis-server.pid\nuvicorn api.app:app --host 0.0.0.0 --port 8081 --workers 2 --http httptools --loop uvloop --timeout-keep-alive 120' > /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 8081

CMD ["/app/start.sh"]