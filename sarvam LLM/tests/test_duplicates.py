"""Tests for document/chunk duplicate detection without silent deletion."""

from __future__ import annotations

from src.chunking import create_chunks
from src.models import DocumentRecord, PageText
from src.parsing import parse_document


def test_document_duplicate_kept_in_db(tmp_db):
    data = b"Identical payload for duplicate testing."
    first = parse_document("one.txt", data)
    tmp_db.insert_document(
        first.document,
        pages=[{"page_number": p.page_number, "text": p.text} for p in first.pages],
    )

    second = parse_document(
        "two.txt",
        data,
        existing_hash_lookup=tmp_db.find_document_by_hash,
    )
    tmp_db.insert_document(
        second.document,
        pages=[{"page_number": p.page_number, "text": p.text} for p in second.pages],
    )

    docs = tmp_db.list_documents()
    assert len(docs) == 2
    dup = next(d for d in docs if d.document_id == second.document.document_id)
    assert dup.is_duplicate is True
    assert dup.duplicate_of == first.document.document_id


def test_chunk_duplicates_flagged_and_persisted(tmp_db):
    text = "Exact same chunk body for hashing."
    pages = [PageText(page_number=1, text=text)]
    # Insert document first for FK
    doc = DocumentRecord(
        document_id="doc_a",
        filename="a.txt",
        file_type="txt",
        file_size=10,
        content_hash="h1",
        page_count=1,
        char_count=len(text),
        status="extracted",
    )
    tmp_db.insert_document(doc, pages=[{"page_number": 1, "text": text}])

    chunks1 = create_chunks("doc_a", pages, chunk_size=500, overlap=0)
    tmp_db.insert_chunks(chunks1)

    doc_b = DocumentRecord(
        document_id="doc_b",
        filename="b.txt",
        file_type="txt",
        file_size=10,
        content_hash="h2",
        page_count=1,
        char_count=len(text),
        status="extracted",
    )
    tmp_db.insert_document(doc_b, pages=[{"page_number": 1, "text": text}])
    chunks2 = create_chunks(
        "doc_b",
        pages,
        chunk_size=500,
        overlap=0,
        hash_lookup=tmp_db.find_chunk_by_hash,
    )
    tmp_db.insert_chunks(chunks2)

    all_chunks = tmp_db.get_chunks()
    assert len(all_chunks) == len(chunks1) + len(chunks2)
    assert any(c.is_duplicate for c in all_chunks if c.document_id == "doc_b")
