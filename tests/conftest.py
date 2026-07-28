"""Shared fixtures. Tests use the offline HashEmbedder + in-memory Qdrant."""

from __future__ import annotations

import pytest

from tenantq.client import make_client
from tenantq.collection import recreate_collection
from tenantq.config import HnswConfig, Settings
from tenantq.data import make_dataset
from tenantq.embeddings import build_embedder
from tenantq.ingest import ingest_documents


@pytest.fixture
def settings() -> Settings:
    return Settings(
        collection="tenantq_test",
        dense_dim=256,
        hnsw=HnswConfig(m=16, ef_construct=64, hnsw_ef=64, multitenant=True, payload_m=16),
    )


@pytest.fixture
def embedder():
    return build_embedder("hash", "", "", 256)


@pytest.fixture
def dataset():
    return make_dataset(docs_per_topic=8, queries_per_tenant=6, seed=7)


@pytest.fixture
def ingested(settings, embedder, dataset):
    client = make_client(settings)
    recreate_collection(client, settings)
    report = ingest_documents(client, settings, embedder, dataset.documents, batch_size=64, parallelism=2)
    assert report.points == len(dataset.documents)
    return client
