"""Benchmark harness sanity + recall metric correctness."""

from __future__ import annotations

import math

from tenantq.benchmark import percentile, recall_at_k, run_benchmark


def test_percentile_and_recall_helpers():
    assert percentile([], 0.95) == 0.0
    assert percentile([10], 0.95) == 10
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert recall_at_k([1, 2, 3], [2, 3], 5) == 1.0
    assert recall_at_k([9, 8, 7], [1, 2], 3) == 0.0
    assert recall_at_k([1, 9, 9], [1, 2], 3) == 0.5


def test_benchmark_produces_numbers(ingested, settings, embedder, dataset):
    result = run_benchmark(ingested, settings, embedder, dataset)
    modes = {m.mode: m for m in result.modes}
    assert set(modes) == {"dense", "sparse", "hybrid"}
    for m in result.modes:
        assert 0.0 <= m.recall_at_5 <= 1.0
        assert 0.0 <= m.recall_at_10 <= 1.0
        assert m.p95_ms >= m.p50_ms >= 0.0
        assert m.p99_ms >= m.p95_ms
        assert m.qps > 0
        assert m.n_queries == len(dataset.queries)
    # dense mode reports index recall vs exact brute force
    assert not math.isnan(modes["dense"].index_recall_at_10)
    # hybrid should be at least as good as the weaker single mode on recall@10
    assert modes["hybrid"].recall_at_10 >= min(modes["dense"].recall_at_10, modes["sparse"].recall_at_10)
