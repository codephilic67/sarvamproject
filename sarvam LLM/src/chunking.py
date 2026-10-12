"""Configurable paragraph-aware chunking with stable IDs and duplicate detection."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Callable, Optional

from src.models import ChunkRecord, PageText


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4) if text else 0


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}|{chunk_index}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"chk_{digest}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _build_char_page_map(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Concatenate pages with double newlines and return (full_text, spans).
    Each span: (start, end, page_number) over the concatenated string.
    """
    spans: list[tuple[int, int, int]] = []
    pieces: list[str] = []
    cursor = 0
    for i, page in enumerate(pages):
        text = page.text or ""
        if i > 0:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(text)
        end = cursor + len(text)
        spans.append((start, end, page.page_number))
        cursor = end
    return "".join(pieces), spans


def pages_for_span(
    spans: list[tuple[int, int, int]], start: int, end: int
) -> list[int]:
    pages: list[int] = []
    for s, e, page_num in spans:
        if e <= start or s >= end:
            continue
        if page_num not in pages:
            pages.append(page_num)
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[tuple[int, int, str]]:
    """
    Paragraph-aware chunking.
    Returns list of (char_start, char_end, chunk_text).
    chunk_size / overlap are in characters.
    """
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    text = text.strip()
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # Map paragraph index -> (start, end) in original concatenated text.
    # Rebuild with same \n\n join used in _split after normalize.
    # Work on the full text positions via sequential search.
    para_spans: list[tuple[int, int, str]] = []
    search_from = 0
    for para in paragraphs:
        idx = text.find(para, search_from)
        if idx < 0:
            idx = search_from
        para_spans.append((idx, idx + len(para), para))
        search_from = idx + len(para)

    chunks: list[tuple[int, int, str]] = []
    current_parts: list[str] = []
    current_start: Optional[int] = None
    current_end = 0
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_start, current_end, current_len
        if not current_parts or current_start is None:
            return
        chunk_body = "\n\n".join(current_parts).strip()
        if chunk_body:
            chunks.append((current_start, current_end, chunk_body))
        current_parts = []
        current_start = None
        current_end = 0
        current_len = 0

    for start, end, para in para_spans:
        para_len = len(para)

        # Oversized single paragraph: hard-split with overlap
        if para_len > chunk_size:
            flush()
            step = max(1, chunk_size - overlap)
            pos = 0
            while pos < para_len:
                piece = para[pos : pos + chunk_size]
                abs_start = start + pos
                abs_end = start + pos + len(piece)
                chunks.append((abs_start, abs_end, piece.strip()))
                if pos + chunk_size >= para_len:
                    break
                pos += step
            continue

        if current_len + para_len + (2 if current_parts else 0) <= chunk_size:
            if current_start is None:
                current_start = start
            current_parts.append(para)
            current_end = end
            current_len = len("\n\n".join(current_parts))
        else:
            flush()
            # Overlap: pull trailing text from previous chunk if available
            if overlap > 0 and chunks:
                prev_start, prev_end, prev_text = chunks[-1]
                overlap_text = prev_text[-overlap:].strip()
                if overlap_text:
                    # Find overlap region start roughly near end of prev
                    ov_start = max(prev_start, prev_end - overlap)
                    current_parts = [overlap_text, para]
                    current_start = ov_start
                    current_end = end
                    current_len = len("\n\n".join(current_parts))
                    continue
            current_parts = [para]
            current_start = start
            current_end = end
            current_len = para_len

    flush()
    return [(s, e, t) for s, e, t in chunks if t]


def create_chunks(
    document_id: str,
    pages: list[PageText],
    chunk_size: int = 800,
    overlap: int = 100,
    hash_lookup: Optional[Callable[[str], Optional[ChunkRecord]]] = None,
) -> list[ChunkRecord]:
    """
    Create ChunkRecord list from page-aware document text.
    Exact duplicates are flagged (is_duplicate=True) but NEVER deleted.
    """
    full_text, page_spans = _build_char_page_map(pages)
    raw_chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    records: list[ChunkRecord] = []
    for idx, (start, end, text) in enumerate(raw_chunks):
        c_hash = content_hash(text)
        is_dup = False
        dup_of = None
        if hash_lookup:
            prior = hash_lookup(c_hash)
            if prior is not None:
                is_dup = True
                dup_of = prior.chunk_id

        # Also detect duplicates within this batch
        for prev in records:
            if prev.content_hash == c_hash and not prev.is_duplicate:
                is_dup = True
                dup_of = prev.chunk_id
                break

        records.append(
            ChunkRecord(
                chunk_id=stable_chunk_id(document_id, idx, text),
                document_id=document_id,
                chunk_index=idx,
                text=text,
                source_pages=pages_for_span(page_spans, start, end),
                char_start=start,
                char_end=end,
                token_estimate=estimate_tokens(text),
                content_hash=c_hash,
                is_duplicate=is_dup,
                duplicate_of=dup_of,
                metadata={
                    "chunk_size_config": chunk_size,
                    "overlap_config": overlap,
                },
            )
        )
    return records


def generate_ephemeral_id() -> str:
    return uuid.uuid4().hex[:12]
