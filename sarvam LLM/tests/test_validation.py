"""Tests for quality validation and evidence checks."""

from __future__ import annotations

from src.models import QAExample, QuestionType, ValidationStatus
from src.validation import (
    build_quality_report,
    evidence_in_source,
    validate_example,
    validate_examples,
)


def test_evidence_in_source_positive(sample_chunk):
    assert evidence_in_source("exact substring of the cited chunk", [sample_chunk.text])


def test_evidence_in_source_negative(sample_chunk):
    assert not evidence_in_source(
        "this quote was invented and does not appear", [sample_chunk.text]
    )


def test_validate_passes_good_example(sample_example, sample_chunk):
    result = validate_example(
        sample_example, {sample_chunk.chunk_id: sample_chunk}
    )
    assert result.validation_status == ValidationStatus.PASSED
    assert result.quality_flags == []


def test_validate_flags_missing_evidence(sample_example, sample_chunk):
    bad = sample_example.model_copy(update={"evidence_quote": ""})
    result = validate_example(bad, {sample_chunk.chunk_id: sample_chunk})
    assert "empty_evidence" in result.quality_flags
    assert result.validation_status == ValidationStatus.FLAGGED


def test_validate_flags_unsupported_evidence(sample_example, sample_chunk):
    bad = sample_example.model_copy(
        update={"evidence_quote": "completely fabricated citation text here"}
    )
    result = validate_example(bad, {sample_chunk.chunk_id: sample_chunk})
    assert "evidence_not_in_source" in result.quality_flags


def test_validate_empty_question_rejected(sample_example, sample_chunk):
    bad = sample_example.model_copy(update={"question": ""})
    result = validate_example(bad, {sample_chunk.chunk_id: sample_chunk})
    assert result.validation_status == ValidationStatus.REJECTED


def test_duplicate_question_detection(sample_example, sample_chunk):
    chunks = {sample_chunk.chunk_id: sample_chunk}
    first = validate_example(sample_example, chunks, set(), set())
    second = validate_example(
        sample_example.model_copy(update={"example_id": "ex2"}),
        chunks,
        {sample_example.question.lower()},
        set(),
    )
    assert first.validation_status == ValidationStatus.PASSED
    assert "duplicate_question" in second.quality_flags


def test_build_quality_report(sample_example, sample_chunk):
    examples = validate_examples([sample_example], [sample_chunk])
    report = build_quality_report(examples)
    assert report.total_examples == 1
    assert report.passed == 1
    assert "schema / required fields" in report.deterministic_checks
