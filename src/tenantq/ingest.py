"""Batch ingestion.

Embeds documents (dense + sparse), builds Qdrant points and upserts them in
configurable batches with optional thread parallelism. Reports throughput and
increments the Prometheus ingestion counter.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Sequence

from qdrant_client import QdrantClient, models

from .config import (
    CATEGORY_FIELD,
    CREATED_AT_FIELD,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    TENANT_FIELD,
    TEXT_FIELD,
    Settings,
)
from .data import Document
from .embeddings import Embedder, SparseVec, batched
from .metrics import INGEST_THROUGHPUT, INGESTED_POINTS


@dataclass
class IngestReport:
    points: int
    seconds: float
    throughput: float


def _build_points(
    docs: Sequence[Document],
    dense: List[List[float]],
    sparse: List[SparseVec],
) -> List[models.PointStruct]:
    points: List[models.PointStruct] = []
    for doc, dv, sv in zip(docs, dense, sparse):
        points.append(
            models.PointStruct(
                id=doc.id,
                vector={
                    DENSE_VECTOR_NAME: dv,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sv.indices, values=sv.values
                    ),
                },
                payload={
                    TENANT_FIELD: doc.tenant_id,
                    CATEGORY_FIELD: doc.category,
                    CREATED_AT_FIELD: doc.created_at,
                    TEXT_FIELD: doc.text,
                },
            )
        )
    return points


def ingest_documents(
    client: QdrantClient,
    settings: Settings,
    embedder: Embedder,
    documents: Sequence[Document],
    batch_size: int = 256,
    parallelism: int = 4,
) -> IngestReport:
    """Embed and upsert ``documents`` in parallel batches."""
    start = time.perf_counter()
    batches = list(batched(list(documents), batch_size))

    def _work(batch: Sequence[Document]) -> int:
        texts = [d.text for d in batch]
        dense = embedder.embed_dense(texts)
        sparse = embedder.embed_sparse(texts)
        points = _build_points(batch, dense, sparse)
        client.upsert(collection_name=settings.collection, points=points, wait=True)
        tenants: dict[str, int] = {}
        for d in batch:
            tenants[d.tenant_id] = tenants.get(d.tenant_id, 0) + 1
        for tenant, n in tenants.items():
            INGESTED_POINTS.labels(tenant=tenant).inc(n)
        return len(batch)

    total = 0
    if parallelism > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for n in pool.map(_work, batches):
                total += n
    else:
        for batch in batches:
            total += _work(batch)

    elapsed = time.perf_counter() - start
    throughput = total / elapsed if elapsed > 0 else float(total)
    INGEST_THROUGHPUT.set(throughput)
    return IngestReport(points=total, seconds=elapsed, throughput=throughput)
