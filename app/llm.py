"""
LLM calls for two purposes:
  - generate_answer: produce an answer from retrieved context (RAG output)
  - judge_answer: LLM-as-judge scoring of generated answer vs. ground truth

Uses NVIDIA's OpenAI-compatible endpoint with strict 30 RPM rate-limiting and robust exponential backoff.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError

MODEL = "meta/llama-3.1-70b-instruct"

_last_call_time = 0.0
MIN_CALL_INTERVAL_SEC = 2.0  # Guarantees <= 30 RPM (under 40 RPM quota)


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


ANSWER_SYSTEM_PROMPT = """You are answering questions using ONLY the provided
context from an internal engineering knowledge base. If the context doesn't
contain the answer, say so explicitly rather than guessing. Be concise --
2-4 sentences."""

JUDGE_SYSTEM_PROMPT = """You are scoring whether a generated answer is
factually consistent with a reference (ground-truth) answer, for a RAG
evaluation. Score 1 if the generated answer captures the key facts of the
reference answer (wording can differ). Score 0 if it's missing a key fact,
contradicts the reference, or is a non-answer ("I don't know") when the
reference has a real answer.

Respond with ONLY a JSON object, no other text:
{"score": 0 or 1, "reasoning": "one sentence"}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse valid JSON from: {text!r}")


def _call_model(messages: list[dict], max_tokens: int = 300, temperature: float = 0.0, max_retries: int = 8) -> str:
    global _last_call_time
    client = get_client()

    for attempt in range(max_retries):
        elapsed = time.time() - _last_call_time
        if elapsed < MIN_CALL_INTERVAL_SEC:
            time.sleep(MIN_CALL_INTERVAL_SEC - elapsed)

        try:
            _last_call_time = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                stream=False,
            )
            return (response.choices[0].message.content or "").strip()
        except (RateLimitError, APIStatusError, Exception) as exc:
            err_msg = str(exc)
            is_rate_limit = isinstance(exc, RateLimitError) or "429" in err_msg or "Rate limit" in err_msg
            if attempt < max_retries - 1:
                backoff = (2 ** attempt) * 2.0 + random.uniform(1.0, 3.0)
                if is_rate_limit:
                    print(f"\n    [RateLimit 429] Pacing backoff for {backoff:.1f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(backoff)
                continue
            raise


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return _call_model(messages, max_tokens=300, temperature=0.0)


def judge_answer(question: str, generated: str, reference: str) -> dict:
    user_message = (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Generated answer: {generated}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        raw_text = _call_model(messages, max_tokens=200, temperature=0.0)
        return _extract_json(raw_text)
    except Exception as exc:
        return {"score": 0, "reasoning": f"judge extraction error: {str(exc)[:100]}"}
