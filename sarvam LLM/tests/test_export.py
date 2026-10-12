"""Tests for JSONL/CSV export."""

from __future__ import annotations

import json

from src.export import export_bundle, filter_examples, to_csv, to_jsonl
from src.models import QAExample, QuestionType, ValidationStatus


def _ex(eid: str, status: ValidationStatus, demo: bool = False) -> QAExample:
    return QAExample(
        example_id=eid,
        question=f"Q {eid}?",
        answer=f"A {eid}",
        source_document="doc1",
        source_chunk_ids=["chk1"],
        source_pages=[1, 2],
        evidence_quote="evidence text here",
        question_type=QuestionType.FACTUAL,
        generation_model="m",
        validation_status=status,
        quality_flags=[],
        is_demo=demo,
    )


def test_filter_excludes_rejected():
    examples = [
        _ex("1", ValidationStatus.PASSED),
        _ex("2", ValidationStatus.REJECTED),
        _ex("3", ValidationStatus.FLAGGED),
    ]
    filtered = filter_examples(examples, exclude_rejected=True)
    ids = {e.example_id for e in filtered}
    assert ids == {"1", "3"}


def test_jsonl_contains_provenance():
    examples = [_ex("1", ValidationStatus.PASSED)]
    raw = to_jsonl(examples)
    row = json.loads(raw.strip())
    assert row["example_id"] == "1"
    assert row["source_chunk_ids"] == ["chk1"]
    assert row["source_pages"] == [1, 2]
    assert "evidence_quote" in row


def test_csv_header_and_row():
    examples = [_ex("1", ValidationStatus.PASSED)]
    csv_text = to_csv(examples)
    lines = csv_text.strip().splitlines()
    assert "example_id" in lines[0]
    assert "source_document" in lines[0]
    assert "1" in lines[1]


def test_export_bundle_preview():
    examples = [
        _ex("1", ValidationStatus.PASSED),
        _ex("d", ValidationStatus.DEMO, demo=True),
    ]
    bundle = export_bundle(examples, exclude_rejected=True, include_demo=True)
    assert bundle["count"] == 2
    assert bundle["jsonl"]
    assert bundle["csv"]
    assert len(bundle["preview"]) == 2
