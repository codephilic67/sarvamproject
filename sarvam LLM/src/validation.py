"""Quality evaluation: deterministic schema, citation, duplicate, and evidence checks."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from src.models import (
    ChunkRecord,
    QAExample,
    QualityFlag,
    QualityReport,
    ValidationStatus,
)


def normalize_for_compare(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def evidence_in_source(evidence: str, chunk_texts: list[str]) -> bool:
    """Programmatic check: evidence quote must appear in cited source chunks."""
    ev = normalize_for_compare(evidence)
    if not ev or len(ev) < 8:
        return False
    for text in chunk_texts:
        if ev in normalize_for_compare(text):
            return True
    # Soft fallback: allow minor whitespace differences via collapsed compare
    ev_compact = re.sub(r"[^a-z0-9]", "", ev)
    if len(ev_compact) < 12:
        return False
    for text in chunk_texts:
        src = re.sub(r"[^a-z0-9]", "", normalize_for_compare(text))
        if ev_compact in src:
            return True
    return False


def is_repetitive(question: str, answer: str) -> bool:
    q = normalize_for_compare(question)
    a = normalize_for_compare(answer)
    if not q or not a:
        return False
    if q == a:
        return True
    if len(a) > 20 and a in q:
        return True
    # High word overlap
    qw = set(q.split())
    aw = set(a.split())
    if qw and aw:
        overlap = len(qw & aw) / max(len(qw | aw), 1)
        if overlap > 0.92 and len(aw) <= len(qw) + 2:
            return True
    return False


def validate_example(
    example: QAExample,
    chunks_by_id: dict[str, ChunkRecord],
    seen_questions: Optional[set[str]] = None,
    seen_pairs: Optional[set[str]] = None,
) -> QAExample:
    """
    Run deterministic quality checks and set validation_status / quality_flags.
    Does not claim semantic answer correctness.
    """
    flags: list[str] = []
    seen_questions = seen_questions if seen_questions is not None else set()
    seen_pairs = seen_pairs if seen_pairs is not None else set()

    if not example.question:
        flags.append(QualityFlag.EMPTY_QUESTION.value)
    if not example.answer:
        flags.append(QualityFlag.EMPTY_ANSWER.value)
    if not example.evidence_quote:
        flags.append(QualityFlag.EMPTY_EVIDENCE.value)

    if (
        not example.source_document
        or not example.source_chunk_ids
        or not example.generation_model
    ):
        flags.append(QualityFlag.MISSING_METADATA.value)

    if not example.source_chunk_ids:
        flags.append(QualityFlag.MISSING_CITATION.value)

    q_norm = normalize_for_compare(example.question)
    pair_norm = f"{q_norm}||{normalize_for_compare(example.answer)}"
    if q_norm and q_norm in seen_questions:
        flags.append(QualityFlag.DUPLICATE_QUESTION.value)
    if pair_norm.strip("|") and pair_norm in seen_pairs:
        flags.append(QualityFlag.DUPLICATE_PAIR.value)
    if q_norm:
        seen_questions.add(q_norm)
    if pair_norm.strip("|"):
        seen_pairs.add(pair_norm)

    chunk_texts: list[str] = []
    missing_chunk = False
    for cid in example.source_chunk_ids:
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            missing_chunk = True
        else:
            chunk_texts.append(chunk.text)
    if missing_chunk:
        flags.append(QualityFlag.MISSING_CITATION.value)

    if example.evidence_quote and chunk_texts:
        if not evidence_in_source(example.evidence_quote, chunk_texts):
            flags.append(QualityFlag.EVIDENCE_NOT_IN_SOURCE.value)
            flags.append(QualityFlag.UNSUPPORTED.value)
    elif example.evidence_quote and not chunk_texts:
        flags.append(QualityFlag.EVIDENCE_NOT_IN_SOURCE.value)

    if is_repetitive(example.question, example.answer):
        flags.append(QualityFlag.REPETITIVE.value)

    # Deduplicate flags while preserving order
    flags = list(dict.fromkeys(flags))

    reject_triggers = {
        QualityFlag.EMPTY_QUESTION.value,
        QualityFlag.EMPTY_ANSWER.value,
        QualityFlag.INVALID_SCHEMA.value,
    }
    hard_flags = {
        QualityFlag.EVIDENCE_NOT_IN_SOURCE.value,
        QualityFlag.EMPTY_EVIDENCE.value,
        QualityFlag.MISSING_CITATION.value,
        QualityFlag.UNSUPPORTED.value,
    }

    if example.is_demo:
        status = ValidationStatus.DEMO
    elif flags and (set(flags) & reject_triggers):
        status = ValidationStatus.REJECTED
    elif flags and (set(flags) & hard_flags):
        status = ValidationStatus.FLAGGED
    elif flags:
        status = ValidationStatus.FLAGGED
    else:
        status = ValidationStatus.PASSED

    return example.model_copy(
        update={"quality_flags": flags, "validation_status": status}
    )


def validate_examples(
    examples: list[QAExample],
    chunks: list[ChunkRecord],
) -> list[QAExample]:
    chunks_by_id = {c.chunk_id: c for c in chunks}
    seen_q: set[str] = set()
    seen_p: set[str] = set()
    return [
        validate_example(ex, chunks_by_id, seen_q, seen_p) for ex in examples
    ]


def build_quality_report(examples: list[QAExample]) -> QualityReport:
    flag_counter: Counter[str] = Counter()
    passed = flagged = rejected = demo = 0
    for ex in examples:
        status = (
            ex.validation_status.value
            if hasattr(ex.validation_status, "value")
            else str(ex.validation_status)
        )
        if status == ValidationStatus.PASSED.value:
            passed += 1
        elif status == ValidationStatus.FLAGGED.value:
            flagged += 1
        elif status == ValidationStatus.REJECTED.value:
            rejected += 1
        elif status == ValidationStatus.DEMO.value:
            demo += 1
        for f in ex.quality_flags:
            flag_counter[f] += 1

    return QualityReport(
        total_examples=len(examples),
        passed=passed,
        flagged=flagged,
        rejected=rejected,
        demo=demo,
        flag_counts=dict(flag_counter),
        deterministic_checks=[
            "schema / required fields",
            "empty question / answer / evidence",
            "missing metadata and citations",
            "exact duplicate questions and QA pairs",
            "evidence quote substring presence in cited chunks",
            "repetitive question≈answer heuristic",
        ],
    )
