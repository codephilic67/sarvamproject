"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import Database
from src.models import ChunkRecord, PageText, QAExample, QuestionType, ValidationStatus


@pytest.fixture
def sample_pages() -> list[PageText]:
    return [
        PageText(
            page_number=1,
            text=(
                "Retrieval-Augmented Generation combines a retriever with a language model.\n\n"
                "The retriever finds relevant passages from a knowledge base."
            ),
        ),
        PageText(
            page_number=2,
            text=(
                "Chunking should preserve semantic boundaries.\n\n"
                "Typical chunk sizes range from 500 to 1000 characters with overlap."
            ),
        ),
    ]


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def sample_chunk() -> ChunkRecord:
    text = (
        "Every generated question-answer pair must cite source chunk IDs. "
        "An evidence quote must be an exact substring of the cited chunk."
    )
    return ChunkRecord(
        chunk_id="chk_test001",
        document_id="doc_test001",
        chunk_index=0,
        text=text,
        source_pages=[1],
        char_start=0,
        char_end=len(text),
        token_estimate=len(text) // 4,
        content_hash="abc",
    )


@pytest.fixture
def sample_example(sample_chunk: ChunkRecord) -> QAExample:
    return QAExample(
        example_id="ex_test001",
        question="What must every QA pair cite?",
        answer="Source chunk IDs.",
        source_document=sample_chunk.document_id,
        source_chunk_ids=[sample_chunk.chunk_id],
        source_pages=[1],
        evidence_quote="must cite source chunk IDs",
        question_type=QuestionType.FACTUAL,
        generation_model="test-model",
        validation_status=ValidationStatus.PENDING,
    )
