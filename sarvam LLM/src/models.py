"""Pydantic schemas for documents, chunks, examples, and evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    REASONING = "reasoning"
    OTHER = "other"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FLAGGED = "flagged"
    REJECTED = "rejected"
    DEMO = "demo"


class QualityFlag(str, Enum):
    EMPTY_QUESTION = "empty_question"
    EMPTY_ANSWER = "empty_answer"
    EMPTY_EVIDENCE = "empty_evidence"
    MISSING_METADATA = "missing_metadata"
    INVALID_SCHEMA = "invalid_schema"
    DUPLICATE_QUESTION = "duplicate_question"
    DUPLICATE_PAIR = "duplicate_pair"
    EVIDENCE_NOT_IN_SOURCE = "evidence_not_in_source"
    MISSING_CITATION = "missing_citation"
    REPETITIVE = "repetitive"
    UNSUPPORTED = "unsupported"


class PageText(BaseModel):
    page_number: int
    text: str


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    content_hash: str
    page_count: int = 0
    char_count: int = 0
    status: str = "uploaded"
    error_message: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_pages: list[int] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    content_hash: str = ""
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAExample(BaseModel):
    example_id: str
    question: str
    answer: str
    source_document: str
    source_chunk_ids: list[str]
    source_pages: list[int] = Field(default_factory=list)
    evidence_quote: str
    question_type: QuestionType = QuestionType.OTHER
    generation_model: str
    validation_status: ValidationStatus = ValidationStatus.PENDING
    quality_flags: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_demo: bool = False

    @field_validator("question", "answer", "evidence_quote")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return (v or "").strip()


class GeneratedPairRaw(BaseModel):
    """Schema expected from Sarvam JSON output for one QA pair."""

    question: str
    answer: str
    evidence_quote: str
    question_type: str = "other"
    source_chunk_id: Optional[str] = None


class QualityReport(BaseModel):
    total_examples: int = 0
    passed: int = 0
    flagged: int = 0
    rejected: int = 0
    demo: int = 0
    flag_counts: dict[str, int] = Field(default_factory=dict)
    deterministic_checks: list[str] = Field(default_factory=list)
    semantic_note: str = (
        "Deterministic checks verify schema, duplicates, citations, and "
        "exact evidence-quote presence in source chunks. Exact quote matching "
        "does NOT prove the entire answer is factually correct."
    )


class RetrievalMetrics(BaseModel):
    recall_at_3: float
    mrr: float
    num_queries: int
    definition_recall: str = (
        "Recall@3: fraction of queries where at least one relevant chunk "
        "appears in the top-3 retrieved results."
    )
    definition_mrr: str = (
        "MRR: mean reciprocal rank of the first relevant chunk across queries."
    )
    caveat: str = (
        "Baseline lexical (TF-IDF) retrieval metrics only. Not proof of "
        "overall answer quality or end-to-end RAG performance."
    )


class PipelineStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    examples: int = 0
    passed_examples: int = 0
    flagged_examples: int = 0
    rejected_examples: int = 0
    duplicate_documents: int = 0
    duplicate_chunks: int = 0
