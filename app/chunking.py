"""
Two chunking strategies to compare in the eval:

  simple    -- fixed-size sliding window over raw text, ignores structure.
               Fast, no assumptions about document format.
  semantic  -- splits on markdown headers/paragraphs first, then packs
               sections up to a max size. Preserves topical boundaries so a
               chunk doesn't straddle two unrelated sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str


def chunk_simple(doc_id: str, title: str, text: str,
                  chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Fixed-size character window with overlap. No awareness of
    sentence/section boundaries -- this is the naive baseline."""
    text = text.strip()
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::simple::{idx}",
                doc_id=doc_id,
                doc_title=title,
                text=chunk_text,
            ))
            idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_semantic(doc_id: str, title: str, text: str,
                    max_chunk_size: int = 700) -> list[Chunk]:
    """Split on markdown headers (## Section) into sections first, then
    pack consecutive sections into chunks up to max_chunk_size, never
    splitting a section itself unless it alone exceeds max_chunk_size."""
    text = text.strip()
    # Split on lines starting with '##' (keep the header with its section)
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[Chunk] = []
    idx = 0
    buffer = ""
    for section in sections:
        if len(section) > max_chunk_size:
            # Section itself too big -- flush buffer, emit section alone
            if buffer:
                chunks.append(Chunk(f"{doc_id}::semantic::{idx}", doc_id, title, buffer.strip()))
                idx += 1
                buffer = ""
            chunks.append(Chunk(f"{doc_id}::semantic::{idx}", doc_id, title, section.strip()))
            idx += 1
            continue

        if len(buffer) + len(section) <= max_chunk_size:
            buffer += ("\n\n" if buffer else "") + section
        else:
            if buffer:
                chunks.append(Chunk(f"{doc_id}::semantic::{idx}", doc_id, title, buffer.strip()))
                idx += 1
            buffer = section

    if buffer:
        chunks.append(Chunk(f"{doc_id}::semantic::{idx}", doc_id, title, buffer.strip()))

    return chunks


CHUNKERS = {
    "simple": chunk_simple,
    "semantic": chunk_semantic,
}
