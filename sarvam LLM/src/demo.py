"""Demo mode fixtures — clearly labeled, no API key required."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from src.models import (
    ChunkRecord,
    DocumentRecord,
    PageText,
    QAExample,
    QuestionType,
    ValidationStatus,
)

DEMO_MODEL = "demo-fixture (not Sarvam)"
DEMO_FILENAME = "demo_knowledge_base.txt"

DEMO_TEXT = """# LLM Knowledge Pipeline — Demo Document

## What is RAG?
Retrieval-Augmented Generation (RAG) combines a retriever with a language model.
The retriever finds relevant passages from a knowledge base. The language model
then answers using those passages as context. This reduces hallucinations when
answers must stay grounded in source documents.

## Chunking best practices
Documents should be split into overlapping chunks so that related sentences stay
together. Typical chunk sizes range from 500 to 1000 characters with 50 to 150
characters of overlap. Paragraph-aware splitting preserves semantic boundaries
better than naive fixed windows.

## Provenance and citations
Every generated question-answer pair must cite source chunk IDs and page numbers.
An evidence quote must be an exact substring of the cited chunk. Exact quote
matching verifies citation integrity but does not prove that the full answer is
factually correct.

## Evaluation metrics
Recall@K measures whether at least one relevant chunk appears in the top-K
results. Mean Reciprocal Rank (MRR) averages the reciprocal rank of the first
relevant result. A TF-IDF lexical baseline is a useful starting point before
testing denser embedding retrievers.
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_demo_document() -> tuple[DocumentRecord, list[PageText]]:
    pages = [PageText(page_number=1, text=DEMO_TEXT.strip())]
    doc = DocumentRecord(
        document_id="doc_demo000001",
        filename=DEMO_FILENAME,
        file_type="txt",
        file_size=len(DEMO_TEXT.encode("utf-8")),
        content_hash=_hash(DEMO_TEXT),
        page_count=1,
        char_count=len(DEMO_TEXT.strip()),
        status="extracted",
        created_at=datetime.now(timezone.utc).isoformat(),
        is_duplicate=False,
    )
    return doc, pages


def create_demo_chunks(document_id: str = "doc_demo000001") -> list[ChunkRecord]:
    # Manually curated chunks aligned to sections for stable demo provenance
    sections = [
        (
            "Retrieval-Augmented Generation (RAG) combines a retriever with a language model.\n"
            "The retriever finds relevant passages from a knowledge base. The language model\n"
            "then answers using those passages as context. This reduces hallucinations when\n"
            "answers must stay grounded in source documents.",
            [1],
        ),
        (
            "Documents should be split into overlapping chunks so that related sentences stay\n"
            "together. Typical chunk sizes range from 500 to 1000 characters with 50 to 150\n"
            "characters of overlap. Paragraph-aware splitting preserves semantic boundaries\n"
            "better than naive fixed windows.",
            [1],
        ),
        (
            "Every generated question-answer pair must cite source chunk IDs and page numbers.\n"
            "An evidence quote must be an exact substring of the cited chunk. Exact quote\n"
            "matching verifies citation integrity but does not prove that the full answer is\n"
            "factually correct.",
            [1],
        ),
        (
            "Recall@K measures whether at least one relevant chunk appears in the top-K\n"
            "results. Mean Reciprocal Rank (MRR) averages the reciprocal rank of the first\n"
            "relevant result. A TF-IDF lexical baseline is a useful starting point before\n"
            "testing denser embedding retrievers.",
            [1],
        ),
    ]
    chunks: list[ChunkRecord] = []
    cursor = 0
    for i, (text, pages) in enumerate(sections):
        text = text.strip()
        c_hash = _hash(text)
        chunks.append(
            ChunkRecord(
                chunk_id=f"chk_demo{i:04d}{c_hash[:8]}",
                document_id=document_id,
                chunk_index=i,
                text=text,
                source_pages=pages,
                char_start=cursor,
                char_end=cursor + len(text),
                token_estimate=max(1, len(text) // 4),
                content_hash=c_hash,
                is_duplicate=False,
                metadata={"demo": True},
            )
        )
        cursor += len(text) + 2
    return chunks


def create_demo_examples(chunks: list[ChunkRecord]) -> list[QAExample]:
    """Fixture QA pairs with real evidence quotes from demo chunks."""
    assert len(chunks) >= 4
    c0, c1, c2, c3 = chunks[0], chunks[1], chunks[2], chunks[3]
    fixtures = [
        (
            c0,
            "What is Retrieval-Augmented Generation (RAG)?",
            "RAG combines a retriever with a language model so answers can be grounded in retrieved passages.",
            "Retrieval-Augmented Generation (RAG) combines a retriever with a language model.",
            QuestionType.DEFINITION,
        ),
        (
            c0,
            "How does RAG help reduce hallucinations?",
            "By answering with passages from a knowledge base as context, answers stay grounded in source documents.",
            "This reduces hallucinations when\nanswers must stay grounded in source documents.",
            QuestionType.REASONING,
        ),
        (
            c1,
            "What typical chunk size range is recommended?",
            "Typical chunk sizes range from 500 to 1000 characters with 50 to 150 characters of overlap.",
            "Typical chunk sizes range from 500 to 1000 characters with 50 to 150\ncharacters of overlap.",
            QuestionType.FACTUAL,
        ),
        (
            c1,
            "Why use paragraph-aware splitting?",
            "Paragraph-aware splitting preserves semantic boundaries better than naive fixed windows.",
            "Paragraph-aware splitting preserves semantic boundaries\nbetter than naive fixed windows.",
            QuestionType.COMPARISON,
        ),
        (
            c2,
            "What must every generated QA pair cite?",
            "Source chunk IDs and page numbers, with an evidence quote that is an exact substring of the cited chunk.",
            "Every generated question-answer pair must cite source chunk IDs and page numbers.",
            QuestionType.FACTUAL,
        ),
        (
            c2,
            "Does exact quote matching prove the answer is correct?",
            "No. Exact quote matching verifies citation integrity but does not prove the full answer is factually correct.",
            "Exact quote\nmatching verifies citation integrity but does not prove that the full answer is\nfactually correct.",
            QuestionType.REASONING,
        ),
        (
            c3,
            "What does Recall@K measure?",
            "Whether at least one relevant chunk appears in the top-K results.",
            "Recall@K measures whether at least one relevant chunk appears in the top-K\nresults.",
            QuestionType.DEFINITION,
        ),
        (
            c3,
            "What is Mean Reciprocal Rank (MRR)?",
            "MRR averages the reciprocal rank of the first relevant result across queries.",
            "Mean Reciprocal Rank (MRR) averages the reciprocal rank of the first\nrelevant result.",
            QuestionType.DEFINITION,
        ),
    ]
    examples: list[QAExample] = []
    for i, (chunk, q, a, evidence, qtype) in enumerate(fixtures):
        examples.append(
            QAExample(
                example_id=f"ex_demo{i:04d}",
                question=q,
                answer=a,
                source_document=chunk.document_id,
                source_chunk_ids=[chunk.chunk_id],
                source_pages=list(chunk.source_pages),
                evidence_quote=evidence,
                question_type=qtype,
                generation_model=DEMO_MODEL,
                validation_status=ValidationStatus.DEMO,
                quality_flags=[],
                is_demo=True,
            )
        )
    return examples


def load_demo_bundle() -> dict:
    """Return document, pages, chunks, and examples for demo mode."""
    doc, pages = create_demo_document()
    chunks = create_demo_chunks(doc.document_id)
    examples = create_demo_examples(chunks)
    return {
        "document": doc,
        "pages": pages,
        "chunks": chunks,
        "examples": examples,
    }
