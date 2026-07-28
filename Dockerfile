FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

EXPOSE 8001

# Runs a benchmark against QDRANT_URL then serves Prometheus metrics on :8001.
CMD ["python", "-m", "tenantq.app"]
