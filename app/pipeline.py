"""
Ties a (chunking strategy, retrieval strategy) pair into one RAG pipeline
that the eval harness can run identically across all 4 combinations.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking import CHUNKERS, Chunk
from app.llm import generate_answer
from app.rerank import rerank
from app.retrieval import HybridIndex, ScoredChunk, VectorIndex

CHUNKING_STRATEGIES = ["simple", "semantic"]
RETRIEVAL_STRATEGIES = ["vector_only", "hybrid_rerank"]


@dataclass
class RAGConfig:
    chunking: str        # "simple" | "semantic"
    retrieval: str       # "vector_only" | "hybrid_rerank"
    top_k: int = 3


class RAGPipeline:
    def __init__(self, docs: list[tuple[str, str, str]], config: RAGConfig):
        self.config = config
        chunker = CHUNKERS[config.chunking]

        self.chunks: list[Chunk] = []
        for doc_id, title, text in docs:
            self.chunks.extend(chunker(doc_id, title, text))

        if config.retrieval == "vector_only":
            self.index = VectorIndex(self.chunks)
        elif config.retrieval == "hybrid_rerank":
            self.index = HybridIndex(self.chunks)
        else:
            raise ValueError(f"Unknown retrieval strategy: {config.retrieval}")

    def retrieve(self, query: str) -> list[ScoredChunk]:
        if self.config.retrieval == "vector_only":
            return self.index.search(query, k=self.config.top_k)
        else:
            # widen the candidate pool, then rerank down to top_k
            candidates = self.index.search(query, k=self.config.top_k * 4)
            return rerank(query, candidates, top_k=self.config.top_k)

    def answer(self, query: str) -> tuple[str, list[ScoredChunk]]:
        retrieved = self.retrieve(query)
        context = [sc.chunk.text for sc in retrieved]
        answer_text = generate_answer(query, context)
        return answer_text, retrieved
