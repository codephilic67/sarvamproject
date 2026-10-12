"""Tests for document validation and extraction."""

from __future__ import annotations

import pytest

from src.parsing import (
    ParseError,
    content_hash,
    extract_plain_text,
    parse_document,
    validate_filename,
    validate_size,
)


def test_validate_filename_accepts_allowed():
    assert validate_filename("notes.PDF") == "notes.PDF"
    assert validate_filename("a/b/doc.md") == "doc.md"


def test_validate_filename_rejects_unknown():
    with pytest.raises(ParseError, match="Unsupported"):
        validate_filename("image.png")


def test_validate_size_empty():
    with pytest.raises(ParseError, match="empty"):
        validate_size(0)


def test_validate_size_too_large():
    with pytest.raises(ParseError, match="exceeds"):
        validate_size(50 * 1024 * 1024)


def test_extract_plain_text():
    pages = extract_plain_text(b"Hello world.\n\nSecond paragraph.")
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Hello world" in pages[0].text


def test_extract_plain_text_empty():
    with pytest.raises(ParseError, match="empty"):
        extract_plain_text(b"   \n\n  ")


def test_parse_document_txt_success():
    data = b"# Title\n\nBody text for parsing."
    parsed = parse_document("guide.txt", data)
    assert parsed.document.status == "extracted"
    assert parsed.document.char_count > 0
    assert parsed.document.page_count == 1
    assert parsed.pages[0].text.startswith("# Title")


def test_parse_document_duplicate_flag():
    data = b"Same content forever."
    first = parse_document("a.txt", data)

    def lookup(h: str):
        return first.document if h == first.document.content_hash else None

    second = parse_document("b.txt", data, existing_hash_lookup=lookup)
    assert second.document.is_duplicate is True
    assert second.document.duplicate_of == first.document.document_id
    # Record still created (not deleted)
    assert second.document.document_id != first.document.document_id


def test_content_hash_stable():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


def test_parse_unsupported_extension_via_validate():
    with pytest.raises(ParseError):
        parse_document("x.exe", b"MZ")
