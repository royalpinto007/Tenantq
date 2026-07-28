"""Qdrant client factory.

Resolves, in priority order:

1. ``url`` argument or ``QDRANT_URL`` env  -> real server (docker-compose path).
2. ``path`` argument or ``TENANTQ_QDRANT_PATH`` env -> on-disk local store.
3. otherwise -> in-memory local store (``:memory:``), used by tests/benchmarks.

The same code runs against all three; local mode supports sparse vectors and
the Query API fusion used for hybrid search.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from .config import Settings


def make_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or Settings.from_env()
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(":memory:")
