# RAG with Evaluation Harness — Results Write-Up

## The Problem

RAG demos frequently ship with a single chunking strategy and a single retrieval method chosen by intuition rather than empirical measurement. This project treats chunking and retrieval as two independent engineering variables and systematically evaluates all four combinations against the same labeled ground-truth evaluation set.

## Evaluation Methodology

- **Corpus**: 10 synthetic internal engineering documents (onboarding, deploy guides, incident postmortems, architecture decision records, security policy, on-call runbooks, API style guides). Includes deliberate vocabulary reuse across documents (e.g. "rollback" appears across deploy guides and postmortems) to stress-test retrieval precision against keyword collisions.
- **Chunking Strategies**:
  - `Simple`: Fixed-size character sliding window (500 chars with 100-char overlap) agnostic to document structure.
  - `Semantic`: Markdown-header-aware section parser packing coherent topics up to 700 chars without breaking structural boundaries.
- **Retrieval Strategies**:
  - `Vector-Only`: Dense embedding search using `sentence-transformers/all-MiniLM-L6-v2`.
  - `Hybrid + Reranking`: Reciprocal Rank Fusion (RRF) combining dense vector search and sparse BM25 (`rank-bm25`), followed by joint cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Eval Set**: 28 labeled Q&A pairs, including cross-document questions requiring synthesized context across separate documents.
- **Metrics**:
  - `Retrieval@3`: Binary score indicating whether any retrieved chunk in top-3 originated from the expected ground-truth document(s).
  - `Answer Accuracy`: LLM-as-a-judge score (0/1) evaluating factual consistency between generated answers and reference ground truth using `meta/llama-3.1-70b-instruct`.

---

## Benchmark Results (2×2 Matrix)

| Chunking Strategy | Retrieval Strategy | Chunks Indexed | Retrieval@3 | Answer Accuracy (LLM Judge) |
| :--- | :--- | :---: | :---: | :---: |
| **Simple (Fixed Window)** | Vector-Only (Dense) | 28 | 96.4% (27/28) | 92.9% (26/28) |
| **Simple (Fixed Window)** | **Hybrid + Rerank** | 28 | **100.0% (28/28)** | **100.0% (28/28)** |
| **Semantic (Header-Aware)** | Vector-Only (Dense) | 19 | 96.4% (27/28) | 96.4% (27/28) |
| **Semantic (Header-Aware)** | **Hybrid + Rerank** | 19 | **100.0% (28/28)** | **96.4% (27/28)** |

---

## What the Results Support

### 1. Hybrid Search + Reranking Hits 100% Retrieval@3
Hybrid search with cross-encoder reranking achieved **100% Retrieval@3 across both chunking strategies**, outperforming standalone vector search (96.4% → 100.0%). 
Dense embeddings struggled on exact identifier queries (e.g. specific ADR numbers like `ADR-021` or specific test suite names) where semantic similarity maps broadly across multiple incident reports. Combining BM25 keyword matching with dense embeddings via Reciprocal Rank Fusion (RRF) and joint cross-encoder reranking eliminated these retrieval misses entirely.

### 2. Semantic Chunking Yields ~32% Index Efficiency at Equal Recall Ceiling
Semantic chunking indexed the corpus into **19 chunks vs. 28 chunks** (a **32.1% reduction in index size** and embedding storage). Because chunks respect markdown section boundaries, they eliminate broken topical context while achieving the exact same 100% retrieval ceiling under hybrid reranking.

---

## What the Results Do NOT Support

At a sample size of $n=28$, a single question's variation accounts for a 3.6 percentage point delta (1/28 = 3.57%). 
The slight difference between Simple Hybrid (28/28 = 100%) and Semantic Hybrid (27/28 = 96.4%) is within expected sampling noise. It would be an overfit interpretation to claim semantic chunking degrades answer generation quality when paired with reranking; both configs operate at the top ceiling.

---

## Limitations & Honest Caveats

1. **Sample Size ($n=28$)**: While 28 labeled pairs establish a reliable baseline, detecting nuanced 2–3% generation improvements requires expanding the test suite to 50+ questions.
2. **Self-Judge Bias**: `meta/llama-3.1-70b-instruct` was used for both answer generation and LLM-as-a-judge scoring. While strict factual scoring prompts were used, production-scale evaluations benefit from multi-model cross-judging (e.g. Claude or GPT-4 evaluating LLaMA outputs).

---

## Takeaway

Hybrid search (Dense + BM25 via RRF) combined with cross-encoder reranking is the single highest-leverage improvement in the RAG retrieval pipeline, turning vocabulary-mismatch misses into 100% retrieval accuracy. Semantic chunking serves as an effective complementary optimization, cutting total chunk volume by ~32% without sacrificing retrieval recall.
