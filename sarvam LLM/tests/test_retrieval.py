"""Tests for TF-IDF retrieval metrics."""

from __future__ import annotations

from src.models import ChunkRecord, QAExample, QuestionType, ValidationStatus
from src.retrieval import (
    evaluate_retrieval,
    recall_at_k,
    reciprocal_rank,
    retrieve_top_k,
    build_tfidf_index,
)


def _chunk(cid: str, text: str, doc: str = "d1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid,
        document_id=doc,
        chunk_index=0,
        text=text,
        source_pages=[1],
        content_hash=cid,
    )


def _example(qid: str, question: str, chunk_ids: list[str]) -> QAExample:
    return QAExample(
        example_id=qid,
        question=question,
        answer="ans",
        source_document="d1",
        source_chunk_ids=chunk_ids,
        source_pages=[1],
        evidence_quote="x" * 20,
        question_type=QuestionType.FACTUAL,
        generation_model="test",
        validation_status=ValidationStatus.PASSED,
    )


def test_recall_and_mrr_helpers():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_retrieve_prefers_relevant_chunk():
    chunks = [
        _chunk("c1", "Solar battery storage systems use lithium ion cells."),
        _chunk("c2", "Cats prefer warm sunny windowsills in the afternoon."),
        _chunk("c3", "Baking bread requires flour water yeast and salt."),
    ]
    vectorizer, matrix, ids = build_tfidf_index(chunks)
    ranked = retrieve_top_k(
        "How do solar battery storage systems work?",
        vectorizer,
        matrix,
        ids,
        k=3,
    )
    assert ranked[0][0] == "c1"


def test_evaluate_retrieval_metrics():
    chunks = [
        _chunk("c1", "Retrieval augmented generation uses a retriever and a generator."),
        _chunk("c2", "Mean reciprocal rank averages reciprocal ranks of first hits."),
        _chunk("c3", "Unrelated gardening tips for tomatoes and basil plants."),
    ]
    examples = [
        _example("e1", "What is retrieval augmented generation?", ["c1"]),
        _example("e2", "What does mean reciprocal rank average?", ["c2"]),
    ]
    metrics = evaluate_retrieval(examples, chunks, k=3)
    assert metrics.num_queries == 2
    assert metrics.recall_at_3 >= 0.5
    assert 0.0 <= metrics.mrr <= 1.0
    assert "baseline" in metrics.caveat.lower() or "Baseline" in metrics.caveat
