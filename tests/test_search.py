"""Search modes and metadata filtering."""

from __future__ import annotations

from tenantq.search import search


def test_all_modes_return_hits(ingested, settings, embedder):
    for mode in ("dense", "sparse", "hybrid"):
        hits = search(ingested, settings, embedder, "neural network embedding", tenant_id="acme", mode=mode, limit=10)
        assert len(hits) > 0
        # scores are sorted descending
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_category_filter_restricts_results(ingested, settings, embedder, dataset):
    category = "networking"
    hits = search(
        ingested, settings, embedder, "router packet latency", tenant_id="acme",
        mode="hybrid", limit=20, category=category,
    )
    assert hits
    assert all(h.category == category for h in hits)


def test_created_at_range_filter(ingested, settings, embedder, dataset):
    from tenantq.search import build_filter  # noqa: F401  (exercised indirectly)

    acme = sorted([d for d in dataset.documents if d.tenant_id == "acme"], key=lambda d: d.created_at)
    lo = acme[2].created_at
    hi = acme[6].created_at
    hits = search(
        ingested, settings, embedder, "database index query", tenant_id="acme",
        mode="dense", limit=50, category=None, created_after=lo, created_before=hi,
    )
    # every returned doc must fall inside the requested time window
    by_id = {d.id: d for d in dataset.documents}
    for h in hits:
        assert lo <= by_id[h.id].created_at <= hi
