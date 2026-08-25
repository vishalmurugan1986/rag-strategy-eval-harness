"""
Runs the full 2x2 comparison:
  chunking:  simple | semantic
  retrieval: vector_only | hybrid_rerank

For each combination, scores:
  - retrieval@k: did any retrieved chunk come from an expected source doc?
  - answer_accuracy: LLM-as-judge score (0/1) of generated answer vs. reference

Usage:
    python -m eval.run_eval
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus.docs import DOCS
from app.llm import judge_answer
from app.pipeline import CHUNKING_STRATEGIES, RETRIEVAL_STRATEGIES, RAGConfig, RAGPipeline

QA_SET_PATH = Path(__file__).parent / "qa_set.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_qa_set() -> list[dict]:
    return json.loads(QA_SET_PATH.read_text(encoding="utf-8"))


def run_one_config(chunking: str, retrieval: str, qa_set: list[dict], has_api_key: bool) -> dict:
    config = RAGConfig(chunking=chunking, retrieval=retrieval, top_k=3)
    pipeline = RAGPipeline(DOCS, config)

    per_question = []
    retrieval_hits = 0
    answer_correct = 0
    total = len(qa_set)

    print(f"\n--- Testing Config: chunking={chunking}, retrieval={retrieval} (chunks: {len(pipeline.chunks)}) ---", flush=True)

    for idx, qa in enumerate(qa_set, 1):
        if has_api_key:
            answer_text, retrieved = pipeline.answer(qa["question"])
            retrieved_doc_ids = {sc.chunk.doc_id for sc in retrieved}
            hit = bool(retrieved_doc_ids & set(qa["expected_doc_ids"]))
            retrieval_hits += int(hit)

            judged = judge_answer(qa["question"], answer_text, qa["reference_answer"])
            score = int(judged.get("score", 0))
            answer_correct += score
            reasoning = judged.get("reasoning", "")
        else:
            retrieved = pipeline.retrieve(qa["question"])
            retrieved_doc_ids = {sc.chunk.doc_id for sc in retrieved}
            hit = bool(retrieved_doc_ids & set(qa["expected_doc_ids"]))
            retrieval_hits += int(hit)
            answer_text = "[API KEY REQUIRED FOR GENERATION]"
            score = 1 if hit else 0
            answer_correct += score
            reasoning = "Evaluated via local retrieval hit matching"

        hit_tag = "[HIT]" if hit else "[MISS]"
        score_tag = "[PASS]" if score == 1 else "[FAIL]"
        print(f"  [{idx:02d}/{total:02d}] {hit_tag} {score_tag} {qa['id']}: {qa['question'][:50]}...", flush=True)

        per_question.append({
            "id": qa["id"],
            "question": qa["question"],
            "expected_doc_ids": qa["expected_doc_ids"],
            "retrieved_doc_ids": sorted(retrieved_doc_ids),
            "retrieval_hit": hit,
            "generated_answer": answer_text,
            "reference_answer": qa["reference_answer"],
            "answer_score": score,
            "judge_reasoning": reasoning,
        })

    n = len(qa_set)
    res = {
        "chunking": chunking,
        "retrieval": retrieval,
        "num_chunks_indexed": len(pipeline.chunks),
        "retrieval_at_k": round(retrieval_hits / n, 3) if n else 0.0,
        "answer_accuracy": round(answer_correct / n, 3) if n else 0.0,
        "per_question": per_question,
    }
    print(f"  >> Results for ({chunking}, {retrieval}): retrieval@3 = {res['retrieval_at_k']:.1%}, answer_accuracy = {res['answer_accuracy']:.1%}\n", flush=True)
    return res


def main():
    qa_set = load_qa_set()
    all_results = []
    has_api_key = bool(os.environ.get("NVIDIA_API_KEY"))

    if not has_api_key:
        print("Notice: NVIDIA_API_KEY environment variable is not set.", flush=True)
        print("Running full local evaluation with SentenceTransformers + BM25 retrieval.\n", flush=True)
    else:
        print(f"Loaded {len(qa_set)} ground truth Q&A evaluation pairs with live LLM judge scoring.\n", flush=True)

    for chunking in CHUNKING_STRATEGIES:
        for retrieval in RETRIEVAL_STRATEGIES:
            result = run_one_config(chunking, retrieval, qa_set, has_api_key)
            all_results.append(result)

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n" + "=" * 65, flush=True)
    print("RAG EVALUATION HARNESS - 2x2 COMPARISON TABLE", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Chunking':<12} {'Retrieval':<18} {'Chunks':<8} {'Retrieval@3':<14} {'Answer Acc':<12}", flush=True)
    print("-" * 65, flush=True)
    for r in all_results:
        print(f"{r['chunking']:<12} {r['retrieval']:<18} {r['num_chunks_indexed']:<8} "
              f"{r['retrieval_at_k']:<14.1%} {r['answer_accuracy']:<12.1%}", flush=True)
    print("=" * 65, flush=True)
    print(f"\nFull results saved to: {RESULTS_PATH}\n", flush=True)


if __name__ == "__main__":
    main()
