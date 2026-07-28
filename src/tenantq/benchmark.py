"""Benchmark harness: Recall@K, p50/p95/p99 latency, and exact-index recall.

Two notions of correctness are measured:

* **Recall@K vs ground truth** -- the synthetic dataset defines, for every query,
  the set of truly relevant documents (same tenant + same topic). This measures
  end-to-end retrieval *quality* and lets us compare dense / sparse / hybrid.
* **Index recall vs exact** -- an exact brute-force cosine top-K over the tenant's
  vectors is the reference; comparing the ANN dense result against it measures how
  much recall the HNSW index itself sacrifices for speed.

Latency percentiles are computed from per-query wall-clock timings.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from qdrant_client import QdrantClient

from .config import Settings
from .data import Dataset, Document
from .embeddings import Embedder
from .metrics import RECALL_GAUGE
from .search import Mode, search


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def recall_at_k(retrieved: Sequence[int], relevant: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hit = len(top & set(relevant))
    return hit / min(k, len(relevant))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def exact_topk(
    query_vec: Sequence[float],
    docs: Sequence[Document],
    doc_vecs: Dict[int, List[float]],
    tenant_id: str,
    k: int,
) -> List[int]:
    """Exact brute-force cosine top-K within a tenant (reference ranking)."""
    scored = [
        (_cosine(query_vec, doc_vecs[d.id]), d.id)
        for d in docs
        if d.tenant_id == tenant_id
    ]
    scored.sort(reverse=True)
    return [doc_id for _, doc_id in scored[:k]]


@dataclass
class ModeResult:
    mode: str
    recall_at_5: float
    recall_at_10: float
    index_recall_at_10: float  # ANN dense vs exact; NaN for non-dense modes
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    n_queries: int


@dataclass
class BenchmarkResult:
    settings_summary: Dict[str, object]
    modes: List[ModeResult] = field(default_factory=list)


def run_benchmark(
    client: QdrantClient,
    settings: Settings,
    embedder: Embedder,
    dataset: Dataset,
    modes: Sequence[Mode] = ("dense", "sparse", "hybrid"),
    k_values: Sequence[int] = (5, 10),
    limit: int = 10,
) -> BenchmarkResult:
    kmax = max(k_values)
    # Precompute exact dense reference rankings once (per query).
    doc_vecs = {
        d.id: v
        for d, v in zip(
            dataset.documents,
            embedder.embed_dense([d.text for d in dataset.documents]),
        )
    }
    query_vecs = embedder.embed_dense([q.text for q in dataset.queries])
    exact_ref: List[List[int]] = [
        exact_topk(qv, dataset.documents, doc_vecs, q.tenant_id, kmax)
        for q, qv in zip(dataset.queries, query_vecs)
    ]

    result = BenchmarkResult(
        settings_summary={
            "collection": settings.collection,
            "hnsw_m": settings.hnsw.m,
            "ef_construct": settings.hnsw.ef_construct,
            "hnsw_ef": settings.hnsw.hnsw_ef,
            "multitenant": settings.hnsw.multitenant,
            "payload_m": settings.hnsw.payload_m,
            "n_docs": len(dataset.documents),
            "n_queries": len(dataset.queries),
        }
    )

    for mode in modes:
        latencies: List[float] = []
        r5: List[float] = []
        r10: List[float] = []
        idx_recall: List[float] = []
        for q, ref in zip(dataset.queries, exact_ref):
            t0 = time.perf_counter()
            hits = search(
                client,
                settings,
                embedder,
                q.text,
                tenant_id=q.tenant_id,
                mode=mode,
                limit=max(limit, kmax),
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            retrieved = [h.id for h in hits]
            r5.append(recall_at_k(retrieved, q.relevant_ids, 5))
            r10.append(recall_at_k(retrieved, q.relevant_ids, 10))
            if mode == "dense":
                idx_recall.append(recall_at_k(retrieved, ref, 10))

        total_time = sum(latencies) / 1000.0
        avg_r5 = sum(r5) / len(r5)
        avg_r10 = sum(r10) / len(r10)
        RECALL_GAUGE.labels(mode=mode, k="5").set(avg_r5)
        RECALL_GAUGE.labels(mode=mode, k="10").set(avg_r10)
        result.modes.append(
            ModeResult(
                mode=mode,
                recall_at_5=avg_r5,
                recall_at_10=avg_r10,
                index_recall_at_10=(sum(idx_recall) / len(idx_recall)) if idx_recall else float("nan"),
                p50_ms=percentile(latencies, 0.50),
                p95_ms=percentile(latencies, 0.95),
                p99_ms=percentile(latencies, 0.99),
                qps=(len(latencies) / total_time) if total_time > 0 else 0.0,
                n_queries=len(latencies),
            )
        )
    return result
