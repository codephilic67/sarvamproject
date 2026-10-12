"""Tests for Sarvam response JSON extraction (no live API)."""

from __future__ import annotations

import pytest

from src.generation import _parse_examples_payload
from src.models import ChunkRecord
from src.sarvam import SarvamResponseError, extract_json_object, get_api_key


def test_extract_json_object_raw():
    data = extract_json_object('{"examples": []}')
    assert data == {"examples": []}


def test_extract_json_object_fenced():
    data = extract_json_object('Here:\n```json\n{"a": 1}\n```')
    assert data == {"a": 1}


def test_extract_json_object_invalid():
    with pytest.raises(SarvamResponseError):
        extract_json_object("not json at all")


def test_parse_examples_maps_chunk():
    chunk = ChunkRecord(
        chunk_id="chk_abc",
        document_id="doc1",
        chunk_index=0,
        text="The capital of France is Paris and it is large.",
        source_pages=[3],
        content_hash="x",
    )
    raw = """
    {
      "examples": [
        {
          "question": "What is the capital of France?",
          "answer": "Paris",
          "evidence_quote": "The capital of France is Paris",
          "question_type": "factual",
          "source_chunk_id": "chk_abc"
        }
      ]
    }
    """
    examples = _parse_examples_payload(
        raw, {"chk_abc": chunk}, fallback_chunk=chunk, model="sarvam-105b"
    )
    assert len(examples) == 1
    assert examples[0].source_pages == [3]
    assert examples[0].source_chunk_ids == ["chk_abc"]
    assert examples[0].generation_model == "sarvam-105b"


def test_get_api_key_none_by_default(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert get_api_key() is None
