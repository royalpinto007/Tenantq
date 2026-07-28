"""Configuration for collections, HNSW tuning and multitenancy.

All values are overridable from the environment so the same code path works for
an in-process local store (benchmarks/tests) and a real Qdrant server (docker).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Payload fields used throughout the project.
TENANT_FIELD = "tenant_id"
CATEGORY_FIELD = "category"
CREATED_AT_FIELD = "created_at"
TEXT_FIELD = "text"


@dataclass(frozen=True)
class HnswConfig:
    """HNSW build/query parameters.

    ``m`` and ``ef_construct`` affect index *build* quality and memory. ``hnsw_ef``
    is the *query*-time search breadth (higher -> better recall, slower).

    For multitenancy Qdrant recommends a global ``m=0`` (disables the global graph)
    combined with a per-tenant sub-graph built via ``payload_m`` on the tenant
    payload index. That keeps each tenant's vectors in their own small graph, which
    is both faster and strictly isolated.
    """

    m: int = 16
    ef_construct: int = 100
    hnsw_ef: int = 64
    # Multitenancy: 0 disables the global graph; payload_m builds per-tenant graphs.
    multitenant: bool = True
    payload_m: int = 16

    @property
    def global_m(self) -> int:
        return 0 if self.multitenant else self.m


@dataclass(frozen=True)
class Settings:
    collection: str = "tenantq"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"
    dense_dim: int = 384
    # Storage: an on-disk path, ":memory:", or a URL is resolved by client.py.
    qdrant_url: str | None = None
    qdrant_path: str | None = None
    hnsw: HnswConfig = field(default_factory=HnswConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        hnsw = HnswConfig(
            m=int(os.getenv("TENANTQ_HNSW_M", "16")),
            ef_construct=int(os.getenv("TENANTQ_HNSW_EF_CONSTRUCT", "100")),
            hnsw_ef=int(os.getenv("TENANTQ_HNSW_EF", "64")),
            multitenant=os.getenv("TENANTQ_MULTITENANT", "1") not in ("0", "false", "False"),
            payload_m=int(os.getenv("TENANTQ_PAYLOAD_M", "16")),
        )
        return cls(
            collection=os.getenv("TENANTQ_COLLECTION", "tenantq"),
            dense_model=os.getenv("TENANTQ_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            sparse_model=os.getenv("TENANTQ_SPARSE_MODEL", "Qdrant/bm25"),
            dense_dim=int(os.getenv("TENANTQ_DENSE_DIM", "384")),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_path=os.getenv("TENANTQ_QDRANT_PATH"),
            hnsw=hnsw,
        )
