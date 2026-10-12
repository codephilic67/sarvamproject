"""Live demo of the LLM Knowledge Pipeline (no API key required)."""

from __future__ import annotations

from pathlib import Path

from src.chunking import create_chunks
from src.db import Database
from src.demo import load_demo_bundle
from src.export import export_bundle
from src.models import ValidationStatus
from src.parsing import parse_document
from src.retrieval import evaluate_retrieval
from src.validation import build_quality_report, validate_examples


def main() -> None:
    print("=" * 60)
    print(" LLM KNOWLEDGE PIPELINE — LIVE DEMO")
    print("=" * 60)

    db = Database(Path("data") / "demo_session.db")
    db.clear_all()

    bundle = load_demo_bundle()
    pages = [{"page_number": p.page_number, "text": p.text} for p in bundle["pages"]]
    db.insert_document(bundle["document"], pages=pages)
    db.insert_chunks(bundle["chunks"])
    examples = validate_examples(bundle["examples"], bundle["chunks"])
    for ex in examples:
        if ex.is_demo:
            ex.validation_status = ValidationStatus.DEMO
    db.insert_examples(examples)

    doc = bundle["document"]
    print("\n[1] DEMO MODE LOADED (no API key)")
    print(f"    Document: {doc.filename}")
    print(f"    Chunks:   {len(bundle['chunks'])}")
    print(f"    Examples: {len(examples)}  (labeled DEMO FIXTURE)")

    print("\n[2] SAMPLE Q&A WITH EVIDENCE")
    for ex in examples[:3]:
        print(f"    Q: {ex.question}")
        print(f"    A: {ex.answer}")
        ev = ex.evidence_quote.replace("\n", " ")[:70]
        print(f"    Evidence: {ev}...")
        print(f"    Chunks: {ex.source_chunk_ids} | Pages: {ex.source_pages}")
        print()

    report = build_quality_report(examples)
    print("[3] QUALITY REPORT")
    print(
        f"    Total={report.total_examples} Demo={report.demo} "
        f"Passed={report.passed} Flagged={report.flagged}"
    )
    print(f"    Deterministic checks: {len(report.deterministic_checks)}")

    metrics = evaluate_retrieval(examples, bundle["chunks"])
    print("\n[4] TF-IDF RETRIEVAL BENCHMARK")
    print(f"    Queries:  {metrics.num_queries}")
    print(f"    Recall@3: {metrics.recall_at_3:.3f}")
    print(f"    MRR:      {metrics.mrr:.3f}")
    print("    Note: baseline lexical retrieval only")

    out = export_bundle(examples, exclude_rejected=True)
    print("\n[5] EXPORT")
    print(f"    Examples exported: {out['count']}")
    print("    First JSONL line:")
    print("   ", out["jsonl"].splitlines()[0][:120] + "...")

    data = Path("sample_data/sample_knowledge.txt").read_bytes()
    parsed = parse_document("sample_knowledge.txt", data)
    chunks = create_chunks(
        parsed.document.document_id, parsed.pages, chunk_size=400, overlap=50
    )
    db.insert_document(
        parsed.document,
        pages=[{"page_number": p.page_number, "text": p.text} for p in parsed.pages],
    )
    db.insert_chunks(chunks)
    print("\n[6] REAL DOCUMENT UPLOAD PATH")
    print(f"    File: {parsed.document.filename} ({parsed.document.status})")
    print(f"    Chars: {parsed.document.char_count} | Chunks: {len(chunks)}")
    print(f"    First chunk pages: {chunks[0].source_pages}")
    preview = chunks[0].text[:100].replace("\n", " ")
    print(f"    Preview: {preview}...")

    stats = db.get_stats()
    print("\n[7] PIPELINE STATS")
    print(
        f"    Documents={stats.documents} Chunks={stats.chunks} "
        f"Examples={stats.examples}"
    )
    print("\n" + "=" * 60)
    print(" DEMO COMPLETE — this is a working prototype")
    print("=" * 60)
    print("\nOpen the UI with:  streamlit run app.py")


if __name__ == "__main__":
    main()
