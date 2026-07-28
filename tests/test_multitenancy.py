"""Tenant isolation must be absolute: a query never returns another tenant's data."""

from __future__ import annotations

from tenantq.search import search


def test_results_are_tenant_scoped(ingested, settings, embedder, dataset):
    tenants = sorted({d.tenant_id for d in dataset.documents})
    for tenant in tenants:
        for mode in ("dense", "sparse", "hybrid"):
            hits = search(
                ingested, settings, embedder, "database index query", tenant_id=tenant, mode=mode, limit=25
            )
            assert hits, f"expected hits for {tenant}/{mode}"
            assert all(h.tenant_id == tenant for h in hits)


def test_isolation_no_cross_tenant_leak(ingested, settings, embedder, dataset):
    # Take a document that exists only for tenant 'acme' and query its exact text
    # while scoped to a different tenant. Its id must never appear.
    acme_docs = [d for d in dataset.documents if d.tenant_id == "acme"]
    target = acme_docs[0]
    for other in ("globex", "initech"):
        hits = search(
            ingested, settings, embedder, target.text, tenant_id=other, mode="hybrid", limit=50
        )
        assert target.id not in {h.id for h in hits}
        assert all(h.tenant_id == other for h in hits)
