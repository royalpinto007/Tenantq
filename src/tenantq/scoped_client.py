"""Tenant-scoped Qdrant access.

Raw ``QdrantClient.query_points`` / ``scroll`` / ``retrieve`` accept an optional
filter. A caller that forgets the tenant condition silently returns other
tenants' documents. This wrapper is the only code path that may issue those
reads: it requires ``tenant_id`` and always injects the tenant filter.

``search()`` and any future query helpers must go through :class:`TenantScopedClient`.
A CI test greps the package for direct client reads outside this module.
"""

from __future__ import annotations

from typing import Any, Optional

from qdrant_client import QdrantClient, models

from .config import TENANT_FIELD


def _require_tenant_id(tenant_id: str) -> str:
    if tenant_id is None or not str(tenant_id).strip():
        raise ValueError("tenant_id is required and must be non-empty")
    return str(tenant_id)


def _with_tenant(tenant_id: str, query_filter: Optional[models.Filter]) -> models.Filter:
    """Merge a mandatory tenant match into an optional caller filter.

    The tenant condition is always present. Callers cannot pass a filter that
    drops it: even if they include their own tenant clause, we still add ours
    for the required ``tenant_id``.
    """
    tenant_id = _require_tenant_id(tenant_id)
    tenant_cond = models.FieldCondition(
        key=TENANT_FIELD, match=models.MatchValue(value=tenant_id)
    )
    if query_filter is None:
        return models.Filter(must=[tenant_cond])

    must = list(query_filter.must or [])
    # Put tenant first so the isolation guarantee is obvious in serialized filters.
    must.insert(0, tenant_cond)
    return models.Filter(
        must=must,
        should=query_filter.should,
        must_not=query_filter.must_not,
        min_should=query_filter.min_should,
    )


def _scope_prefetch(tenant_id: str, prefetch: Any) -> Any:
    """Ensure every Prefetch branch carries the tenant filter."""
    if prefetch is None:
        return None
    if isinstance(prefetch, list):
        return [_scope_prefetch(tenant_id, p) for p in prefetch]
    if isinstance(prefetch, models.Prefetch):
        merged = _with_tenant(tenant_id, prefetch.filter)
        nested = _scope_prefetch(tenant_id, prefetch.prefetch) if prefetch.prefetch else None
        # model_copy keeps forward-compat with new Prefetch fields
        if hasattr(prefetch, "model_copy"):
            return prefetch.model_copy(update={"filter": merged, "prefetch": nested})
        return models.Prefetch(
            query=prefetch.query,
            using=prefetch.using,
            filter=merged,
            params=prefetch.params,
            limit=prefetch.limit,
            prefetch=nested,
        )
    return prefetch


class TenantScopedClient:
    """Thin wrapper: the only surface allowed to call read APIs on Qdrant.

    Write/ingest paths still use the raw client (points already carry tenant_id
    in payload). Isolation at read time is what this enforces.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    @property
    def raw(self) -> QdrantClient:
        """Escape hatch for ingest, collection admin, and tests — not for search."""
        return self._client

    def query_points(
        self,
        *,
        tenant_id: str,
        collection_name: str,
        query_filter: Optional[models.Filter] = None,
        prefetch: Any = None,
        **kwargs: Any,
    ):
        qf = _with_tenant(tenant_id, query_filter)
        pf = _scope_prefetch(tenant_id, prefetch)
        return self._client.query_points(
            collection_name=collection_name,
            query_filter=qf,
            prefetch=pf,
            **kwargs,
        )

    def scroll(
        self,
        *,
        tenant_id: str,
        collection_name: str,
        scroll_filter: Optional[models.Filter] = None,
        **kwargs: Any,
    ):
        sf = _with_tenant(tenant_id, scroll_filter)
        return self._client.scroll(
            collection_name=collection_name,
            scroll_filter=sf,
            **kwargs,
        )

    def retrieve(
        self,
        *,
        tenant_id: str,
        collection_name: str,
        ids: list,
        **kwargs: Any,
    ):
        # retrieve has no filter param on all client versions; post-filter by tenant
        points = self._client.retrieve(collection_name=collection_name, ids=ids, **kwargs)
        tenant_id = _require_tenant_id(tenant_id)
        return [
            p
            for p in points
            if (p.payload or {}).get(TENANT_FIELD) == tenant_id
        ]
