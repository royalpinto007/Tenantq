"""Collection lifecycle: creation, HNSW tuning and payload indexes.

Implements Qdrant's recommended multitenant setup:

* a single collection holding every tenant's data;
* a keyword payload index on ``tenant_id`` with ``is_tenant=True``;
* global HNSW ``m=0`` (no cross-tenant graph) plus per-tenant sub-graphs via
  ``payload_m`` on the tenant index.

Additional payload indexes (``category`` keyword, ``created_at`` integer) enable
fast metadata filtering.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from .config import (
    CATEGORY_FIELD,
    CREATED_AT_FIELD,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    TENANT_FIELD,
    HnswConfig,
    Settings,
)


def recreate_collection(client: QdrantClient, settings: Settings) -> None:
    """Create (or replace) the collection with tuned HNSW + payload indexes."""
    hnsw = settings.hnsw
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)

    client.create_collection(
        collection_name=settings.collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=settings.dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=hnsw.global_m,
            ef_construct=hnsw.ef_construct,
        ),
    )
    _create_payload_indexes(client, settings, hnsw)


def _create_payload_indexes(client: QdrantClient, settings: Settings, hnsw: HnswConfig) -> None:
    # Tenant index: is_tenant=True + payload_m builds isolated per-tenant graphs.
    client.create_payload_index(
        collection_name=settings.collection,
        field_name=TENANT_FIELD,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=hnsw.multitenant,
        ),
    )
    if hnsw.multitenant:
        client.update_collection(
            collection_name=settings.collection,
            hnsw_config=models.HnswConfigDiff(payload_m=hnsw.payload_m, m=0),
        )
    # Metadata indexes for fast filtering.
    client.create_payload_index(
        collection_name=settings.collection,
        field_name=CATEGORY_FIELD,
        field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    )
    client.create_payload_index(
        collection_name=settings.collection,
        field_name=CREATED_AT_FIELD,
        field_schema=models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            range=True,
        ),
    )
