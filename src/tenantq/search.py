"""Search: dense / sparse / hybrid, tenant-isolated, with metadata filtering.

Every query is scoped to a single tenant via a mandatory filter on ``tenant_id``.
Hybrid mode issues two prefetch branches (dense + sparse) and fuses them with
Qdrant's server-side Reciprocal Rank Fusion (``Fusion.RRF``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from qdrant_client import QdrantClient, models

from .scoped_client import TenantScopedClient

from .config import (
    CATEGORY_FIELD,
    CREATED_AT_FIELD,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    TENANT_FIELD,
    TEXT_FIELD,
    Settings,
)
from .embeddings import Embedder
from .metrics import QUERY_LATENCY

Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class SearchHit:
    id: int
    score: float
    text: str
    tenant_id: str
    category: str


def build_filter(
    tenant_id: str,
    category: Optional[str] = None,
    created_after: Optional[int] = None,
    created_before: Optional[int] = None,
) -> models.Filter:
    """Build a tenant-scoped filter, optionally narrowed by metadata."""
    must: List[models.FieldCondition] = [
        models.FieldCondition(key=TENANT_FIELD, match=models.MatchValue(value=tenant_id))
    ]
    if category is not None:
        must.append(
            models.FieldCondition(key=CATEGORY_FIELD, match=models.MatchValue(value=category))
        )
    if created_after is not None or created_before is not None:
        must.append(
            models.FieldCondition(
                key=CREATED_AT_FIELD,
                range=models.Range(gte=created_after, lte=created_before),
            )
        )
    return models.Filter(must=must)


def _to_hits(points: Sequence[models.ScoredPoint]) -> List[SearchHit]:
    hits: List[SearchHit] = []
    for p in points:
        payload = p.payload or {}
        hits.append(
            SearchHit(
                id=int(p.id),
                score=float(p.score),
                text=payload.get(TEXT_FIELD, ""),
                tenant_id=payload.get(TENANT_FIELD, ""),
                category=payload.get(CATEGORY_FIELD, ""),
            )
        )
    return hits


def search(
    client: QdrantClient,
    settings: Settings,
    embedder: Embedder,
    query: str,
    tenant_id: str,
    mode: Mode = "hybrid",
    limit: int = 10,
    category: Optional[str] = None,
    created_after: Optional[int] = None,
    created_before: Optional[int] = None,
    prefetch_limit: int = 50,
) -> List[SearchHit]:
    """Run a tenant-isolated search in the requested retrieval mode.

    All Qdrant reads go through :class:`TenantScopedClient`, which requires
    ``tenant_id`` and injects the tenant filter so a missing filter cannot
    silently cross tenants.
    """
    qfilter = build_filter(tenant_id, category, created_after, created_before)
    # build_filter already includes tenant_id; scoped client re-injects it as a
    # second hard guarantee (duplicate must-conditions are fine for MatchValue).
    scoped = TenantScopedClient(client)
    params = models.SearchParams(hnsw_ef=settings.hnsw.hnsw_ef)

    start = time.perf_counter()
    if mode == "dense":
        dense_vec = embedder.embed_dense([query])[0]
        res = scoped.query_points(
            tenant_id=tenant_id,
            collection_name=settings.collection,
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            query_filter=qfilter,
            search_params=params,
            limit=limit,
            with_payload=True,
        )
    elif mode == "sparse":
        sv = embedder.embed_sparse([query])[0]
        res = scoped.query_points(
            tenant_id=tenant_id,
            collection_name=settings.collection,
            query=models.SparseVector(indices=sv.indices, values=sv.values),
            using=SPARSE_VECTOR_NAME,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
    elif mode == "hybrid":
        dense_vec = embedder.embed_dense([query])[0]
        sv = embedder.embed_sparse([query])[0]
        res = scoped.query_points(
            tenant_id=tenant_id,
            collection_name=settings.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    filter=qfilter,
                    params=params,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sv.indices, values=sv.values),
                    using=SPARSE_VECTOR_NAME,
                    filter=qfilter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown mode: {mode!r}")

    QUERY_LATENCY.labels(mode=mode).observe(time.perf_counter() - start)
    return _to_hits(res.points)
