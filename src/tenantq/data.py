"""Synthetic labeled dataset with known ground truth.

We generate documents grouped into topics across several tenants. Each query is
built from a specific topic within a specific tenant, so the *relevant* set is
well defined: the documents of that topic belonging to that tenant. This gives a
deterministic, reproducible corpus for recall evaluation without any download.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

TOPICS = {
    "databases": [
        "index query transaction schema replication sharding postgres sql engine",
        "primary key foreign key normalization acid isolation vacuum autovacuum",
        "connection pool prepared statement explain analyze btree hash join",
    ],
    "networking": [
        "tcp udp packet router switch latency throughput congestion handshake",
        "dns resolver bgp routing table subnet cidr firewall nat gateway",
        "http header tls certificate socket port bandwidth proxy load balancer",
    ],
    "machinelearning": [
        "gradient descent neural network embedding transformer attention loss",
        "training validation overfitting regularization dropout batch epoch tensor",
        "vector similarity cosine retrieval ranking recall precision benchmark",
    ],
    "cooking": [
        "recipe onion garlic simmer saute knife chopping board skillet oven",
        "flour butter sugar dough bake pastry whisk oven temperature rise",
        "spice cumin paprika marinade grill roast season salt pepper broth",
    ],
    "finance": [
        "portfolio equity bond yield interest rate dividend valuation risk hedge",
        "cash flow balance sheet revenue margin liability asset depreciation audit",
        "market volatility option future derivative arbitrage liquidity spread index",
    ],
}
CATEGORIES = list(TOPICS.keys())

# Common "glue" tokens shared across every topic. They add lexical noise so that
# retrieval is non-trivial and dense/sparse/hybrid separate on quality.
COMMON = "system data value using based process result table field record set".split()


@dataclass
class Document:
    id: int
    tenant_id: str
    text: str
    category: str
    created_at: int


@dataclass
class Query:
    text: str
    tenant_id: str
    category: str
    # ground-truth relevant document ids (same tenant + same topic)
    relevant_ids: List[int] = field(default_factory=list)


@dataclass
class Dataset:
    documents: List[Document]
    queries: List[Query]


def _vocab_for(category: str) -> List[str]:
    return " ".join(TOPICS[category]).split()


def make_dataset(
    tenants: List[str] | None = None,
    docs_per_topic: int = 30,
    queries_per_tenant: int = 10,
    seed: int = 13,
) -> Dataset:
    tenants = tenants or ["acme", "globex", "initech"]
    rng = random.Random(seed)
    documents: List[Document] = []
    # map (tenant, category) -> list of doc ids for ground truth.
    by_group: dict[tuple[str, str], List[int]] = {}
    doc_id = 0
    base_ts = 1_700_000_000
    for tenant in tenants:
        for category in CATEGORIES:
            vocab = _vocab_for(category)
            for _ in range(docs_per_topic):
                # on-topic words diluted by common glue tokens and cross-topic
                # noise, so no single retrieval mode trivially wins.
                words = rng.choices(vocab, k=rng.randint(8, 14))
                words += rng.choices(COMMON, k=rng.randint(4, 8))
                for _ in range(rng.randint(1, 3)):
                    other = rng.choice(CATEGORIES)
                    words += rng.choices(_vocab_for(other), k=rng.randint(2, 4))
                rng.shuffle(words)
                documents.append(
                    Document(
                        id=doc_id,
                        tenant_id=tenant,
                        text=" ".join(words),
                        category=category,
                        created_at=base_ts + doc_id * 3600,
                    )
                )
                by_group.setdefault((tenant, category), []).append(doc_id)
                doc_id += 1

    queries: List[Query] = []
    for tenant in tenants:
        for _ in range(queries_per_tenant):
            category = rng.choice(CATEGORIES)
            vocab = _vocab_for(category)
            # short, partial queries (3-4 topic terms + a common token) make the
            # task realistic and force real ranking rather than exact recall.
            qwords = rng.sample(vocab, k=min(rng.randint(3, 4), len(vocab)))
            qwords += rng.choices(COMMON, k=1)
            queries.append(
                Query(
                    text=" ".join(qwords),
                    tenant_id=tenant,
                    category=category,
                    relevant_ids=list(by_group[(tenant, category)]),
                )
            )
    return Dataset(documents=documents, queries=queries)
