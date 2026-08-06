---
title: tenantq demo
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# tenantq demo

A small, self-contained web demo of [**tenantq**](https://github.com/AgentPostmortem/tenantq),
a multi-tenant hybrid-search reference deployment on Qdrant.

Several tenants' documents live in a single Qdrant collection, but every query is
strictly tenant-isolated: an agent for tenant A can never see tenant B's data. The
demo seeds tenantq's synthetic multi-tenant dataset into an embedded on-disk Qdrant
store (qdrant-client local mode) at startup, using the real fastembed dense
(`sentence-transformers/all-MiniLM-L6-v2`) + sparse (`Qdrant/bm25`) backend so
dense, sparse, and hybrid (RRF fusion) retrieval all carry real signal.

## Run locally

```bash
pip install -r requirements.txt
python app.py            # or: uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open http://localhost:7860.

## Docker

```bash
docker build -t tenantq-demo .
docker run -p 7860:7860 tenantq-demo
```

## Endpoints

- `GET /` — single-page UI (tenant dropdown, query box, dense/sparse/hybrid modes).
- `POST /search` — `{"tenant","query","mode","k"}` → `{"results":[{"content","score","category"}],"latency_ms","mode"}`.
- `GET /health` — readiness probe.

Source: [github.com/AgentPostmortem/tenantq](https://github.com/AgentPostmortem/tenantq)
