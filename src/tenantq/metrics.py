"""Prometheus metrics.

Exposes a query-latency histogram, an ingestion counter and a recall gauge, plus
a helper to start the ``/metrics`` HTTP endpoint.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

QUERY_LATENCY = Histogram(
    "tenantq_query_latency_seconds",
    "Search query latency in seconds",
    labelnames=("mode",),
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)

INGESTED_POINTS = Counter(
    "tenantq_ingested_points_total",
    "Total number of points upserted",
    labelnames=("tenant",),
)

RECALL_GAUGE = Gauge(
    "tenantq_recall",
    "Most recent measured recall@k",
    labelnames=("mode", "k"),
)

INGEST_THROUGHPUT = Gauge(
    "tenantq_ingest_throughput_points_per_second",
    "Throughput of the most recent ingestion run",
)


def serve_metrics(port: int = 8001) -> None:
    """Start a background HTTP server exposing /metrics."""
    start_http_server(port)
