"""Tests for chunking and provenance."""

from __future__ import annotations

from src.chunking import (
    chunk_text,
    content_hash,
    create_chunks,
    pages_for_span,
    stable_chunk_id,
)
from src.models import PageText


def test_chunk_text_basic():
    text = "Para one is a short block.\n\nPara two is here as well.\n\nPara three continues onward."
    chunks = chunk_text(text, chunk_size=50, overlap=5)
    assert len(chunks) >= 1
    assert all(t.strip() for _, _, t in chunks)


def test_chunk_overlap_constraint():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world " * 20, chunk_size=50, overlap=50)


def test_stable_chunk_id_deterministic():
    a = stable_chunk_id("doc1", 0, "same text")
    b = stable_chunk_id("doc1", 0, "same text")
    c = stable_chunk_id("doc1", 1, "same text")
    assert a == b
    assert a != c
    assert a.startswith("chk_")


def test_create_chunks_page_provenance(sample_pages):
    chunks = create_chunks("doc_x", sample_pages, chunk_size=120, overlap=20)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "doc_x"
        assert c.source_pages
        assert all(p in (1, 2) for p in c.source_pages)
        assert c.chunk_id.startswith("chk_")


def test_pages_for_span():
    spans = [(0, 10, 1), (12, 30, 2)]
    assert pages_for_span(spans, 0, 10) == [1]
    assert pages_for_span(spans, 8, 15) == [1, 2]
    assert pages_for_span(spans, 12, 20) == [2]


def test_duplicate_chunks_flagged_not_deleted():
    # Oversized identical paragraphs force separate hard-split chunks with same hash
    body = ("Repeated paragraph content that will hash identically. " * 8).strip()
    pages = [
        PageText(page_number=1, text=body),
        PageText(page_number=2, text=body),
    ]
    chunks = create_chunks("doc_dup", pages, chunk_size=80, overlap=0)
    assert len(chunks) >= 2
    hashes = [c.content_hash for c in chunks]
    assert len(set(hashes)) < len(hashes)
    assert any(c.is_duplicate for c in chunks)
    assert all(c.chunk_id for c in chunks)  # none deleted


def test_content_hash():
    assert content_hash("a") == content_hash("a")
