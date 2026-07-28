"""Dense + sparse embedding backends.

Two interchangeable implementations:

* :class:`FastEmbedEmbedder` -- production backend using ``fastembed`` (dense
  sentence-transformers + a sparse BM25/SPLADE model). Used by the CLI and the
  real benchmark.
* :class:`HashEmbedder` -- a deterministic, dependency-free, fully offline
  backend used by the test-suite and CI. It hashes tokens into a dense vector
  and a sparse bag-of-words so hybrid/dense/sparse retrieval all carry real
  lexical signal without downloading any model.

Both return the same simple value objects, so every downstream module (ingest,
search, benchmark) is agnostic to which backend is in use.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SparseVec:
    indices: List[int]
    values: List[float]


class Embedder(Protocol):
    dense_dim: int

    def embed_dense(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_sparse(self, texts: Sequence[str]) -> List[SparseVec]:
        ...


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Deterministic offline embedder.

    Dense: L2-normalised hashing vectorizer over tokens (feature hashing).
    Sparse: token-hash -> tf weight bag-of-words (BM25-like lexical signal).
    """

    def __init__(self, dense_dim: int = 384):
        self.dense_dim = dense_dim

    def _h(self, token: str, mod: int) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % mod

    def embed_dense(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dense_dim
            for tok in _tokenize(text):
                idx = self._h(tok, self.dense_dim)
                sign = 1.0 if self._h(tok + "#sign", 2) == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_sparse(self, texts: Sequence[str]) -> List[SparseVec]:
        out: List[SparseVec] = []
        for text in texts:
            counts: dict[int, float] = {}
            for tok in _tokenize(text):
                idx = self._h(tok, 2**31)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            # sublinear tf weighting keeps common tokens from dominating.
            items = sorted(counts.items())
            out.append(
                SparseVec(
                    indices=[i for i, _ in items],
                    values=[1.0 + math.log(v) for _, v in items],
                )
            )
        return out


class FastEmbedEmbedder:
    """Real dense+sparse backend built on fastembed."""

    def __init__(
        self,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        dense_dim: int = 384,
    ):
        from fastembed import SparseTextEmbedding, TextEmbedding  # lazy import

        self._dense = TextEmbedding(dense_model)
        self._sparse = SparseTextEmbedding(sparse_model)
        self.dense_dim = dense_dim

    def embed_dense(self, texts: Sequence[str]) -> List[List[float]]:
        return [v.tolist() for v in self._dense.embed(list(texts))]

    def embed_sparse(self, texts: Sequence[str]) -> List[SparseVec]:
        out: List[SparseVec] = []
        for s in self._sparse.embed(list(texts)):
            out.append(SparseVec(indices=s.indices.tolist(), values=s.values.tolist()))
        return out


def build_embedder(kind: str, dense_model: str, sparse_model: str, dense_dim: int) -> Embedder:
    kind = kind.lower()
    if kind == "hash":
        return HashEmbedder(dense_dim=dense_dim)
    if kind == "fastembed":
        return FastEmbedEmbedder(dense_model, sparse_model, dense_dim)
    raise ValueError(f"unknown embedder kind: {kind!r} (use 'fastembed' or 'hash')")


def batched(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
