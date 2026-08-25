"""
Retrieval implementations:
  - VectorIndex: dense embedding search (sentence-transformers, local, no API cost)
  - BM25Index: sparse keyword search (rank_bm25)
  - hybrid_search: reciprocal rank fusion of the two, optionally reranked

Kept separate from chunking so any chunk list (simple or semantic) can be
indexed by either retrieval strategy -- that's what lets the eval harness
produce a full 2x2 comparison (chunking strategy x retrieval strategy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.chunking import Chunk

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        model = _get_embed_model()
        texts = [c.text for c in chunks]
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        model = _get_embed_model()
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in top_idx]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(-scores)[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in top_idx]


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Combine ranked lists from different retrievers. RRF is rank-based
    (not raw-score-based) so it works even though BM25 and cosine-similarity
    scores live on totally different scales."""
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, sc in enumerate(results):
            chunk_lookup[sc.chunk.chunk_id] = sc.chunk
            fused_scores[sc.chunk.chunk_id] = fused_scores.get(sc.chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda x: -x[1])
    return [ScoredChunk(chunk_lookup[cid], score) for cid, score in ranked]


class HybridIndex:
    """Combines vector + BM25 via RRF. This is the retrieval strategy
    compared against vector-only in the eval."""

    def __init__(self, chunks: list[Chunk]):
        self.vector_index = VectorIndex(chunks)
        self.bm25_index = BM25Index(chunks)

    def search(self, query: str, k: int = 5, fusion_k: int = 20) -> list[ScoredChunk]:
        vector_results = self.vector_index.search(query, k=fusion_k)
        bm25_results = self.bm25_index.search(query, k=fusion_k)
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        return fused[:k]
