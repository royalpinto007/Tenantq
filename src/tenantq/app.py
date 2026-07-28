"""Container entrypoint.

Waits for the configured Qdrant server, runs a benchmark against it (populating
the Prometheus gauges), then serves /metrics indefinitely so Prometheus can
scrape latency, ingestion and recall metrics.
"""

from __future__ import annotations

import os
import time

from .benchmark import run_benchmark
from .client import make_client
from .collection import recreate_collection
from .config import Settings
from .data import make_dataset
from .embeddings import build_embedder
from .ingest import ingest_documents
from .metrics import serve_metrics


def _wait_for_server(settings: Settings, timeout: float = 60.0) -> None:
    if not settings.qdrant_url:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            make_client(settings).get_collections()
            return
        except Exception:  # pragma: no cover - startup race
            time.sleep(2.0)
    raise RuntimeError(f"Qdrant not reachable at {settings.qdrant_url}")


def main() -> None:
    port = int(os.getenv("TENANTQ_METRICS_PORT", "8001"))
    embedder_kind = os.getenv("TENANTQ_EMBEDDER", "fastembed")
    settings = Settings.from_env()

    serve_metrics(port)
    print(f"[tenantq] metrics on :{port}/metrics", flush=True)

    _wait_for_server(settings)
    client = make_client(settings)
    emb = build_embedder(embedder_kind, settings.dense_model, settings.sparse_model, settings.dense_dim)
    dataset = make_dataset()

    recreate_collection(client, settings)
    report = ingest_documents(client, settings, emb, dataset.documents)
    print(f"[tenantq] ingested {report.points} points at {report.throughput:.0f} pts/s", flush=True)

    result = run_benchmark(client, settings, emb, dataset)
    for m in result.modes:
        print(
            f"[tenantq] {m.mode}: recall@10={m.recall_at_10:.3f} p95={m.p95_ms:.2f}ms qps={m.qps:.0f}",
            flush=True,
        )

    print("[tenantq] serving metrics; press Ctrl-C to exit", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
