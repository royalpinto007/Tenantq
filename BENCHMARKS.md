# Benchmarks

Real numbers from an actual run of the tenantq benchmark harness. Reproduce with:

```bash
tenantq benchmark --embedder fastembed --sweep
```

## Setup

| | |
| --- | --- |
| Qdrant | in-process local mode (`:memory:`), qdrant-client 1.18.0 |
| Embeddings | dense `sentence-transformers/all-MiniLM-L6-v2` (384-d, cosine) + sparse `Qdrant/bm25` (IDF), fastembed 0.8.0 |
| Corpus | 450 documents, 3 tenants x 5 topics x 30 docs (synthetic, labeled) |
| Queries | 30 short partial queries (3-4 topic terms + a common token) |
| Ground truth | same-tenant same-topic documents; plus exact brute-force cosine top-K as an index-recall reference |
| Fusion | Qdrant Query API prefetch (dense + sparse, tenant-filtered) fused with `Fusion.RRF` |
| Host | Linux x86_64, Python 3.12, 16 vCPU |
| Ingestion | batched parallel upsert, ~130-143 points/sec (includes dense+sparse embedding) |

Recall@K is measured against the labeled relevant set (retrieval *quality*).
`index_recall@10` compares the ANN dense result against the exact brute-force
ranking (how much recall the HNSW index itself gives up) and is only meaningful
for the dense mode.

## HNSW query-breadth sweep (`hnsw_ef` = 32 / 64 / 128)

Build parameters fixed at `m=16`, `ef_construct=100`, multitenant (`global m=0`,
`payload_m=16`).

### hnsw_ef = 32  (ingest 135 pts/s)

| mode | recall@5 | recall@10 | index_recall@10 | p50 ms | p95 ms | p99 ms | qps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense  | 0.953 | 0.917 | 1.000 | 13.04 | 14.04 | 14.74 | 76 |
| sparse | 0.913 | 0.860 | n/a   | 5.16  | 5.39  | 5.46  | 194 |
| hybrid | 0.953 | 0.893 | n/a   | 19.59 | 20.26 | 42.14 | 48 |

### hnsw_ef = 64  (ingest 143 pts/s)

| mode | recall@5 | recall@10 | index_recall@10 | p50 ms | p95 ms | p99 ms | qps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense  | 0.953 | 0.917 | 1.000 | 13.33 | 15.02 | 17.17 | 74 |
| sparse | 0.913 | 0.860 | n/a   | 4.89  | 5.26  | 5.38  | 206 |
| hybrid | 0.953 | 0.893 | n/a   | 19.77 | 20.98 | 22.45 | 50 |

### hnsw_ef = 128  (ingest 130 pts/s)

| mode | recall@5 | recall@10 | index_recall@10 | p50 ms | p95 ms | p99 ms | qps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense  | 0.953 | 0.917 | 1.000 | 13.91 | 14.63 | 14.92 | 72 |
| sparse | 0.913 | 0.860 | n/a   | 5.80  | 6.05  | 6.08  | 173 |
| hybrid | 0.953 | 0.893 | n/a   | 21.15 | 21.64 | 21.79 | 47 |

## Analysis

**Modes.** On this topical corpus the dense MiniLM model is the strongest single
retriever (recall@5 0.953, recall@10 0.917). Sparse BM25 trails
(0.913 / 0.860) but is **~3x faster** (p50 ~5 ms vs ~13 ms) because it touches an
inverted index rather than a dense graph. Hybrid RRF matches dense at recall@5
(0.953) and sits between the two at recall@10 (0.893): equal-weight RRF lets the
weaker sparse ranking pull a few strong dense hits down. The honest takeaway is
that **hybrid is not a free win** — it shines when queries carry exact terms
(codes, names, rare tokens) that dense embeddings blur, and on purely semantic
topical data a well-matched dense model can lead. tenantq exposes all three modes
precisely so you can measure this on *your* data instead of assuming.

**HNSW index recall.** Dense `index_recall@10` is **1.000** at every `hnsw_ef` —
the HNSW graph reproduces the exact brute-force top-10 perfectly at this corpus
size, so the recall gap versus ground truth is a property of the *embeddings*,
not the index. On larger corpora the `hnsw_ef` lever trades latency for index
recall; here the graph is already saturated, so raising `hnsw_ef` from 32 to 128
mostly just adds latency (dense p50 13.0 -> 13.9 ms) with no recall gain — a
useful reminder to tune against real scale.

**Latency.** All modes are single-digit-to-low-tens of milliseconds p50, with p95
close to p50 (occasional p99 spikes from GC/scheduling on a shared host, e.g. the
42 ms hybrid p99 at ef=32). Sparse is consistently the cheapest, hybrid the most
expensive (two prefetches + fusion).

**Ingestion.** End-to-end batched upsert including both dense and sparse embedding
runs at ~130-143 points/sec on CPU; embedding dominates, not the upsert.

> Note: local `:memory:` mode is used so the benchmark runs with zero
> infrastructure. Payload indexes are a no-op in local mode (filters are still
> applied correctly and enforce tenant isolation); their performance benefit
> appears against a real server via `QDRANT_URL` / docker-compose. All recall and
> latency numbers above are produced by real ingestion and real queries — none are
> hand-written.
