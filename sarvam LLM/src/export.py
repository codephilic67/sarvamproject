"""Export validated QA examples as JSONL and CSV with provenance."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Optional

from src.models import QAExample, ValidationStatus


EXPORT_FIELDS = [
    "example_id",
    "question",
    "answer",
    "source_document",
    "source_chunk_ids",
    "source_pages",
    "evidence_quote",
    "question_type",
    "generation_model",
    "validation_status",
    "quality_flags",
    "is_demo",
]


def _status_value(ex: QAExample) -> str:
    s = ex.validation_status
    return s.value if hasattr(s, "value") else str(s)


def _qtype_value(ex: QAExample) -> str:
    t = ex.question_type
    return t.value if hasattr(t, "value") else str(t)


def filter_examples(
    examples: list[QAExample],
    exclude_rejected: bool = True,
    include_demo: bool = True,
) -> list[QAExample]:
    out: list[QAExample] = []
    for ex in examples:
        status = _status_value(ex)
        if exclude_rejected and status == ValidationStatus.REJECTED.value:
            continue
        if not include_demo and ex.is_demo:
            continue
        out.append(ex)
    return out


def example_to_dict(ex: QAExample) -> dict[str, Any]:
    return {
        "example_id": ex.example_id,
        "question": ex.question,
        "answer": ex.answer,
        "source_document": ex.source_document,
        "source_chunk_ids": list(ex.source_chunk_ids),
        "source_pages": list(ex.source_pages),
        "evidence_quote": ex.evidence_quote,
        "question_type": _qtype_value(ex),
        "generation_model": ex.generation_model,
        "validation_status": _status_value(ex),
        "quality_flags": list(ex.quality_flags),
        "is_demo": ex.is_demo,
    }


def to_jsonl(examples: list[QAExample]) -> str:
    lines = [json.dumps(example_to_dict(ex), ensure_ascii=False) for ex in examples]
    return "\n".join(lines) + ("\n" if lines else "")


def to_csv(examples: list[QAExample]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for ex in examples:
        row = example_to_dict(ex)
        row["source_chunk_ids"] = json.dumps(row["source_chunk_ids"])
        row["source_pages"] = json.dumps(row["source_pages"])
        row["quality_flags"] = json.dumps(row["quality_flags"])
        writer.writerow(row)
    return buf.getvalue()


def preview_rows(
    examples: list[QAExample], limit: int = 20
) -> list[dict[str, Any]]:
    return [example_to_dict(ex) for ex in examples[:limit]]


def export_bundle(
    examples: list[QAExample],
    exclude_rejected: bool = True,
    include_demo: bool = True,
) -> dict[str, Any]:
    filtered = filter_examples(
        examples, exclude_rejected=exclude_rejected, include_demo=include_demo
    )
    return {
        "count": len(filtered),
        "jsonl": to_jsonl(filtered),
        "csv": to_csv(filtered),
        "preview": preview_rows(filtered),
    }
