"""Tests for page provenance through chunking."""

from __future__ import annotations

from src.chunking import create_chunks
from src.models import PageText


def test_multi_page_provenance_attached():
    pages = [
        PageText(page_number=1, text="Alpha content unique to page one about rivers."),
        PageText(page_number=2, text="Beta content unique to page two about mountains."),
        PageText(page_number=3, text="Gamma content unique to page three about forests."),
    ]
    chunks = create_chunks("doc_prov", pages, chunk_size=80, overlap=10)
    assert chunks
    all_pages = set()
    for c in chunks:
        assert c.source_pages
        all_pages.update(c.source_pages)
    # Across the document we should see multiple pages represented
    assert all_pages & {1, 2, 3}


def test_chunk_indexes_sequential():
    pages = [PageText(page_number=1, text=("Sentence. " * 40) + "\n\n" + ("Other. " * 40))]
    chunks = create_chunks("doc_idx", pages, chunk_size=100, overlap=20)
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))
