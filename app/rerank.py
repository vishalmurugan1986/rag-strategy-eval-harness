"""
Cross-encoder reranking. Retrieval (vector/BM25/hybrid) is fast but scores
query and document independently; a cross-encoder scores the (query, doc)
pair jointly, which is slower but more accurate -- so the pattern is:
retrieve a wider candidate set cheaply, then rerank the top N precisely.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.retrieval import ScoredChunk

_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_rerank_model: CrossEncoder | None = None


def _get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(_RERANK_MODEL_NAME)
    return _rerank_model


def rerank(query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
    if not candidates:
        return []
    model = _get_rerank_model()
    pairs = [(query, sc.chunk.text) for sc in candidates]
    scores = model.predict(pairs)
    reranked = sorted(
        (ScoredChunk(sc.chunk, float(score)) for sc, score in zip(candidates, scores)),
        key=lambda sc: -sc.score,
    )
    return reranked[:top_k]
