"""tenantq command-line interface."""

from __future__ import annotations

import json
from typing import List, Optional

import typer

from .benchmark import BenchmarkResult, run_benchmark
from .client import make_client
from .collection import recreate_collection
from .config import HnswConfig, Settings
from .data import make_dataset
from .embeddings import build_embedder
from .ingest import ingest_documents
from .search import search as run_search

app = typer.Typer(add_completion=False, help="Multi-tenant hybrid search on Qdrant.")


def _settings(
    collection: str,
    hnsw_m: int,
    ef_construct: int,
    hnsw_ef: int,
    payload_m: int,
    multitenant: bool,
) -> Settings:
    base = Settings.from_env()
    hnsw = HnswConfig(
        m=hnsw_m,
        ef_construct=ef_construct,
        hnsw_ef=hnsw_ef,
        payload_m=payload_m,
        multitenant=multitenant,
    )
    return Settings(
        collection=collection,
        dense_model=base.dense_model,
        sparse_model=base.sparse_model,
        dense_dim=base.dense_dim,
        qdrant_url=base.qdrant_url,
        qdrant_path=base.qdrant_path,
        hnsw=hnsw,
    )


def _render_markdown(result: BenchmarkResult) -> str:
    s = result.settings_summary
    lines = [
        "| mode | recall@5 | recall@10 | index_recall@10 | p50 ms | p95 ms | p99 ms | qps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in result.modes:
        ir = "n/a" if m.index_recall_at_10 != m.index_recall_at_10 else f"{m.index_recall_at_10:.3f}"
        lines.append(
            f"| {m.mode} | {m.recall_at_5:.3f} | {m.recall_at_10:.3f} | {ir} | "
            f"{m.p50_ms:.2f} | {m.p95_ms:.2f} | {m.p99_ms:.2f} | {m.qps:.0f} |"
        )
    header = (
        f"HNSW m={s['hnsw_m']} ef_construct={s['ef_construct']} hnsw_ef={s['hnsw_ef']} "
        f"multitenant={s['multitenant']} payload_m={s['payload_m']} "
        f"docs={s['n_docs']} queries={s['n_queries']}"
    )
    return header + "\n\n" + "\n".join(lines)


@app.command()
def benchmark(
    embedder: str = typer.Option("fastembed", help="'fastembed' (real) or 'hash' (offline)."),
    collection: str = typer.Option("tenantq_bench"),
    docs_per_topic: int = typer.Option(30),
    queries_per_tenant: int = typer.Option(10),
    hnsw_m: int = typer.Option(16),
    ef_construct: int = typer.Option(100),
    hnsw_ef: int = typer.Option(64),
    payload_m: int = typer.Option(16),
    multitenant: bool = typer.Option(True),
    batch_size: int = typer.Option(256),
    parallelism: int = typer.Option(4),
    sweep: bool = typer.Option(False, help="Sweep a couple of hnsw_ef settings."),
    markdown: bool = typer.Option(True, help="Emit a markdown report."),
) -> None:
    """Ingest a labeled dataset and report recall + latency across modes."""
    dataset = make_dataset(docs_per_topic=docs_per_topic, queries_per_tenant=queries_per_tenant)
    emb = build_embedder(embedder, "sentence-transformers/all-MiniLM-L6-v2", "Qdrant/bm25", 384)

    ef_settings = [hnsw_ef] if not sweep else sorted({32, hnsw_ef, 128})
    reports: List[str] = []
    for i, ef in enumerate(ef_settings):
        settings = _settings(f"{collection}_{ef}", hnsw_m, ef_construct, ef, payload_m, multitenant)
        client = make_client(settings)
        recreate_collection(client, settings)
        report = ingest_documents(
            client, settings, emb, dataset.documents, batch_size=batch_size, parallelism=parallelism
        )
        typer.echo(
            f"[ingest] {report.points} points in {report.seconds:.2f}s "
            f"({report.throughput:.0f} pts/s)"
        )
        result = run_benchmark(client, settings, emb, dataset)
        if markdown:
            reports.append(f"### ingest throughput: {report.throughput:.0f} pts/s\n\n" + _render_markdown(result))
        else:
            typer.echo(json.dumps([m.__dict__ for m in result.modes], indent=2))
    if markdown:
        typer.echo("\n\n".join(reports))


@app.command()
def search(
    query: str,
    tenant: str = typer.Option(..., help="Tenant id to scope the search."),
    mode: str = typer.Option("hybrid"),
    limit: int = typer.Option(5),
    category: Optional[str] = typer.Option(None),
    embedder: str = typer.Option("fastembed"),
    collection: str = typer.Option("tenantq"),
) -> None:
    """Run a one-off tenant-scoped search (requires an ingested collection)."""
    settings = _settings(collection, 16, 100, 64, 16, True)
    client = make_client(settings)
    emb = build_embedder(embedder, settings.dense_model, settings.sparse_model, settings.dense_dim)
    hits = run_search(client, settings, emb, query, tenant_id=tenant, mode=mode, limit=limit, category=category)
    for h in hits:
        typer.echo(f"{h.score:.4f}  [{h.category}]  {h.text[:70]}")


@app.command("serve-metrics")
def serve_metrics(port: int = typer.Option(8001)) -> None:
    """Expose Prometheus /metrics and block."""
    import time

    from .metrics import serve_metrics as _serve

    _serve(port)
    typer.echo(f"metrics on :{port}/metrics")
    while True:
        time.sleep(3600)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
