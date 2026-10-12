"""
LLM Knowledge Pipeline — Streamlit dashboard.
Upload → Chunk → Generate → Evaluate → Benchmark → Export
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on path when launched via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.chunking import create_chunks
from src.db import Database
from src.demo import load_demo_bundle
from src.export import export_bundle
from src.generation import generate_for_chunks
from src.models import ValidationStatus
from src.parsing import ParseError, parse_document
from src.retrieval import evaluate_retrieval
from src.sarvam import has_api_key
from src.validation import build_quality_report, validate_examples

st.set_page_config(
    page_title="LLM Knowledge Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg, #f7f9fc 0%, #eef2f8 100%);
  border: 1px solid #d8e0ec;
  border-radius: 12px;
  padding: 0.75rem 1rem;
}
.hero {
  background: linear-gradient(120deg, #0f2744 0%, #1a4a6e 55%, #2a6f8f 100%);
  color: #f4f8fc;
  padding: 1.4rem 1.6rem;
  border-radius: 14px;
  margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
.hero p { margin: 0.35rem 0 0; opacity: 0.9; font-size: 0.98rem; }
.badge {
  display: inline-block; padding: 0.15rem 0.55rem; border-radius: 6px;
  font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
}
.badge-demo { background: #fff3cd; color: #856404; }
.badge-pass { background: #d4edda; color: #155724; }
.badge-flag { background: #fff3cd; color: #856404; }
.badge-rej { background: #f8d7da; color: #721c24; }
.muted { color: #5a6a7a; font-size: 0.9rem; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }
</style>
""",
    unsafe_allow_html=True,
)


def get_db() -> Database:
    if "db" not in st.session_state:
        st.session_state.db = Database(ROOT / "data" / "pipeline.db")
    return st.session_state.db


def resolve_secrets():
    try:
        return st.secrets
    except Exception:
        return None


def api_configured() -> bool:
    return has_api_key(streamlit_secrets=resolve_secrets())


def status_badge(status: str) -> str:
    s = str(status)
    if s == ValidationStatus.DEMO.value:
        return f'<span class="badge badge-demo">DEMO FIXTURE</span>'
    if s == ValidationStatus.PASSED.value:
        return f'<span class="badge badge-pass">PASSED</span>'
    if s == ValidationStatus.FLAGGED.value:
        return f'<span class="badge badge-flag">FLAGGED</span>'
    if s == ValidationStatus.REJECTED.value:
        return f'<span class="badge badge-rej">REJECTED</span>'
    return f'<span class="badge">{s}</span>'


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Pipeline Controls")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Upload & Process",
            "Chunk Explorer",
            "Generate Dataset",
            "Dataset Explorer",
            "Quality Report",
            "Retrieval Benchmark",
            "Export",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Settings")
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 800, 50)
    overlap = st.slider("Overlap (chars)", 0, 400, 100, 10)
    pairs_per_chunk = st.slider("QA pairs per chunk", 3, 5, 3)
    batch_size = st.slider("API batch size", 1, 3, 2)
    st.divider()
    key_ok = api_configured()
    if key_ok:
        st.success("Sarvam API key detected")
    else:
        st.warning("No SARVAM_API_KEY — use Demo Mode or set the key")
    st.caption("Key is read from env / secrets only. Never shown in the UI.")
    if st.button("Reset database", type="secondary"):
        get_db().clear_all()
        st.session_state.pop("last_retrieval", None)
        st.session_state.pop("last_quality", None)
        st.success("Database cleared.")
        st.rerun()


# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>LLM Knowledge Pipeline</h1>
  <p>Convert documents into traceable RAG datasets — extract, chunk, generate, evaluate, benchmark, export.</p>
</div>
""",
    unsafe_allow_html=True,
)

db = get_db()
stats = db.get_stats()

# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", stats.documents)
    c2.metric("Chunks", stats.chunks)
    c3.metric("Examples", stats.examples)
    c4.metric("Passed", stats.passed_examples)
    c5.metric("Flagged / Rejected", f"{stats.flagged_examples} / {stats.rejected_examples}")

    st.markdown("#### How it works")
    st.markdown(
        """
1. **Upload** PDF, DOCX, TXT, or Markdown  
2. **Chunk** with paragraph-aware splitting and page provenance  
3. **Generate** QA pairs via Sarvam `sarvam-105b` (or load Demo fixtures)  
4. **Evaluate** with deterministic quality checks (no LLM judge by default)  
5. **Benchmark** TF-IDF Recall@3 and MRR  
6. **Export** JSONL / CSV with IDs and provenance  
"""
    )
    if stats.duplicate_documents or stats.duplicate_chunks:
        st.info(
            f"Duplicates flagged (records kept): "
            f"{stats.duplicate_documents} documents, {stats.duplicate_chunks} chunks."
        )

    docs = db.list_documents()
    if docs:
        st.markdown("#### Documents")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": d.document_id,
                        "File": d.filename,
                        "Type": d.file_type,
                        "Pages": d.page_count,
                        "Chars": d.char_count,
                        "Status": d.status,
                        "Duplicate": d.is_duplicate,
                    }
                    for d in docs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown(
            '<p class="muted">No documents yet. Go to <b>Upload &amp; Process</b> or load Demo Mode.</p>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
# UPLOAD & PROCESS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Upload & Process":
    st.subheader("Document Upload")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown (max 20 MB)",
        type=["pdf", "docx", "txt", "md", "markdown"],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        process_btn = st.button("Extract & Chunk", type="primary", disabled=uploaded is None)
    with col_b:
        demo_btn = st.button("Load Demo Mode (no API key)")

    if demo_btn:
        with st.spinner("Loading demo fixtures…"):
            bundle = load_demo_bundle()
            # Avoid re-inserting identical demo doc if already present
            existing = db.get_document(bundle["document"].document_id)
            if existing:
                st.info("Demo document already loaded. Showing existing records.")
            else:
                pages_payload = [
                    {"page_number": p.page_number, "text": p.text}
                    for p in bundle["pages"]
                ]
                db.insert_document(bundle["document"], pages=pages_payload)
                db.insert_chunks(bundle["chunks"])
                validated = validate_examples(bundle["examples"], bundle["chunks"])
                # Keep DEMO status for demo fixtures
                for ex in validated:
                    if ex.is_demo:
                        ex.validation_status = ValidationStatus.DEMO
                db.insert_examples(validated)
                st.success(
                    f"Demo loaded: 1 document, {len(bundle['chunks'])} chunks, "
                    f"{len(validated)} fixture examples (labeled DEMO)."
                )
                st.caption(
                    "These are demonstration fixtures — not outputs from Sarvam AI."
                )
        st.rerun()

    if process_btn and uploaded is not None:
        data = uploaded.getvalue()
        with st.spinner("Extracting text…"):
            try:
                parsed = parse_document(
                    uploaded.name,
                    data,
                    existing_hash_lookup=db.find_document_by_hash,
                )
            except ParseError as exc:
                st.error(f"Validation failed: {exc}")
                st.stop()

            doc = parsed.document
            pages_payload = [
                {"page_number": p.page_number, "text": p.text} for p in parsed.pages
            ]
            db.insert_document(doc, pages=pages_payload)

            if doc.status == "failed":
                st.error(f"Extraction failed: {doc.error_message}")
                st.stop()

            if doc.is_duplicate:
                st.warning(
                    f"Exact duplicate of document `{doc.duplicate_of}`. "
                    "Record kept and flagged — not deleted."
                )

            chunks = create_chunks(
                doc.document_id,
                parsed.pages,
                chunk_size=chunk_size,
                overlap=overlap,
                hash_lookup=db.find_chunk_by_hash,
            )
            db.insert_chunks(chunks)
            db.update_document_status(doc.document_id, "chunked")

            dup_chunks = sum(1 for c in chunks if c.is_duplicate)
            st.success(
                f"Processed **{doc.filename}**: {doc.page_count} pages, "
                f"{doc.char_count:,} chars → {len(chunks)} chunks"
                + (f" ({dup_chunks} duplicate chunks flagged)." if dup_chunks else ".")
            )
            with st.expander("Preview first chunk"):
                if chunks:
                    st.code(chunks[0].text[:800], language=None)
                    st.caption(
                        f"ID: {chunks[0].chunk_id} | Pages: {chunks[0].source_pages}"
                    )

# ═══════════════════════════════════════════════════════════════════════════
# CHUNK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Chunk Explorer":
    st.subheader("Chunk Explorer")
    docs = db.list_documents()
    if not docs:
        st.info("No documents yet. Upload a file or load Demo Mode.")
    else:
        doc_options = {f"{d.filename} ({d.document_id})": d.document_id for d in docs}
        selected = st.selectbox("Document", list(doc_options.keys()))
        doc_id = doc_options[selected]
        show_dups = st.checkbox("Show duplicate chunks", value=True)
        chunks = db.get_chunks(document_id=doc_id, include_duplicates=show_dups)
        st.caption(f"{len(chunks)} chunks")
        if not chunks:
            st.warning("No chunks for this document.")
        else:
            rows = [
                {
                    "Index": c.chunk_index,
                    "Chunk ID": c.chunk_id,
                    "Pages": ", ".join(map(str, c.source_pages)),
                    "Chars": len(c.text),
                    "Tokens~": c.token_estimate,
                    "Duplicate": c.is_duplicate,
                    "Preview": c.text[:120].replace("\n", " "),
                }
                for c in chunks
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            pick = st.selectbox(
                "Inspect chunk",
                [f"[{c.chunk_index}] {c.chunk_id}" for c in chunks],
            )
            idx = int(pick.split("]")[0].strip("["))
            chunk = next(c for c in chunks if c.chunk_index == idx)
            st.markdown(f"**Source pages:** {chunk.source_pages}")
            if chunk.is_duplicate:
                st.warning(f"Duplicate of `{chunk.duplicate_of}` (record retained).")
            st.text_area("Source text", chunk.text, height=240)

# ═══════════════════════════════════════════════════════════════════════════
# GENERATE DATASET
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Generate Dataset":
    st.subheader("Dataset Generation")
    docs = [d for d in db.list_documents() if d.status in ("chunked", "extracted", "generated")]
    if not docs:
        st.info("Chunk a document first (Upload & Process).")
    else:
        doc_options = {f"{d.filename} ({d.document_id})": d.document_id for d in docs}
        selected = st.selectbox("Document to generate from", list(doc_options.keys()))
        doc_id = doc_options[selected]
        chunks = db.get_chunks(document_id=doc_id, include_duplicates=False)
        if not chunks:
            chunks = db.get_chunks(document_id=doc_id, include_duplicates=True)
        st.caption(f"{len(chunks)} chunks available for generation")

        max_chunks = st.number_input(
            "Max chunks to process (cost control)",
            min_value=1,
            max_value=max(1, len(chunks)),
            value=min(5, len(chunks)),
        )
        selected_chunks = chunks[: int(max_chunks)]

        if not api_configured():
            st.warning(
                "No API key configured. Generation requires `SARVAM_API_KEY`. "
                "Use **Load Demo Mode** on the Upload page for fixtures."
            )

        if st.button(
            "Generate QA pairs with Sarvam",
            type="primary",
            disabled=not api_configured() or not selected_chunks,
        ):
            progress = st.empty()
            status_box = st.empty()

            def on_progress(msg: str) -> None:
                progress.info(msg)

            with st.spinner("Calling Sarvam API…"):
                examples, errors = generate_for_chunks(
                    selected_chunks,
                    pairs_per_chunk=pairs_per_chunk,
                    batch_size=batch_size,
                    progress_callback=on_progress,
                )
                if examples:
                    validated = validate_examples(examples, db.get_chunks())
                    db.insert_examples(validated)
                    db.update_document_status(doc_id, "generated")
                    report = build_quality_report(validated)
                    st.session_state.last_quality = report.model_dump()
                    status_box.success(
                        f"Generated {len(validated)} examples "
                        f"(passed={report.passed}, flagged={report.flagged}, "
                        f"rejected={report.rejected})."
                    )
                else:
                    status_box.error("No examples generated.")
                if errors:
                    with st.expander("Errors / warnings", expanded=True):
                        for err in errors:
                            st.write(f"- {err}")

# ═══════════════════════════════════════════════════════════════════════════
# DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    examples = db.get_examples()
    if not examples:
        st.info("No examples yet. Generate with Sarvam or load Demo Mode.")
    else:
        status_filter = st.multiselect(
            "Filter by status",
            options=[s.value for s in ValidationStatus],
            default=[s.value for s in ValidationStatus],
        )
        filtered = [
            e
            for e in examples
            if (
                e.validation_status.value
                if hasattr(e.validation_status, "value")
                else str(e.validation_status)
            )
            in status_filter
        ]
        st.caption(f"Showing {len(filtered)} / {len(examples)} examples")
        table = []
        for e in filtered:
            status = (
                e.validation_status.value
                if hasattr(e.validation_status, "value")
                else str(e.validation_status)
            )
            table.append(
                {
                    "ID": e.example_id,
                    "Question": e.question[:80],
                    "Type": e.question_type.value
                    if hasattr(e.question_type, "value")
                    else e.question_type,
                    "Status": status,
                    "Flags": ", ".join(e.quality_flags) or "—",
                    "Demo": e.is_demo,
                    "Pages": ", ".join(map(str, e.source_pages)),
                    "Model": e.generation_model,
                }
            )
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        if filtered:
            pick = st.selectbox(
                "Inspect example",
                [f"{e.example_id}: {e.question[:60]}" for e in filtered],
            )
            ex = next(e for e in filtered if e.example_id == pick.split(":")[0])
            status = (
                ex.validation_status.value
                if hasattr(ex.validation_status, "value")
                else str(ex.validation_status)
            )
            st.markdown(status_badge(status), unsafe_allow_html=True)
            if ex.is_demo:
                st.caption("Demonstration fixture — not generated by Sarvam AI.")
            st.markdown(f"**Question:** {ex.question}")
            st.markdown(f"**Answer:** {ex.answer}")
            st.markdown(f"**Evidence quote:**")
            st.code(ex.evidence_quote, language=None)
            st.markdown(
                f"**Chunks:** `{', '.join(ex.source_chunk_ids)}` · "
                f"**Pages:** {ex.source_pages} · **Flags:** {ex.quality_flags or 'none'}"
            )
            src_chunks = db.get_chunks_by_ids(ex.source_chunk_ids)
            if src_chunks:
                with st.expander("Cited source chunk text"):
                    for c in src_chunks:
                        st.markdown(f"`{c.chunk_id}` pages {c.source_pages}")
                        st.text(c.text)

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Quality Report":
    st.subheader("Quality Report")
    examples = db.get_examples()
    if not examples:
        st.info("No examples to evaluate.")
    else:
        if st.button("Re-run quality checks", type="primary"):
            chunks = db.get_chunks()
            updated = validate_examples(examples, chunks)
            for ex in updated:
                if ex.is_demo:
                    ex.validation_status = ValidationStatus.DEMO
            db.insert_examples(updated)
            report = build_quality_report(updated)
            st.session_state.last_quality = report.model_dump()
            run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            db.save_evaluation_run(
                run_id,
                datetime.now(timezone.utc).isoformat(),
                quality_report=report,
            )
            st.success("Quality checks updated.")
            examples = updated

        report = build_quality_report(examples)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total", report.total_examples)
        m2.metric("Passed", report.passed)
        m3.metric("Flagged", report.flagged)
        m4.metric("Rejected", report.rejected)
        m5.metric("Demo", report.demo)

        st.markdown("#### Deterministic checks")
        for check in report.deterministic_checks:
            st.write(f"- {check}")
        st.info(report.semantic_note)

        if report.flag_counts:
            st.markdown("#### Flag counts")
            st.bar_chart(pd.Series(report.flag_counts, name="count"))
            st.dataframe(
                pd.DataFrame(
                    [{"Flag": k, "Count": v} for k, v in report.flag_counts.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No quality flags raised.")

        flagged = [
            e
            for e in examples
            if (
                e.validation_status.value
                if hasattr(e.validation_status, "value")
                else str(e.validation_status)
            )
            in (ValidationStatus.FLAGGED.value, ValidationStatus.REJECTED.value)
        ]
        if flagged:
            st.markdown("#### Review queue")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": e.example_id,
                            "Status": e.validation_status.value
                            if hasattr(e.validation_status, "value")
                            else e.validation_status,
                            "Flags": ", ".join(e.quality_flags),
                            "Question": e.question[:70],
                        }
                        for e in flagged
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Retrieval Benchmark":
    st.subheader("Retrieval Benchmark (TF-IDF baseline)")
    st.markdown(
        """
This runs a **lexical TF-IDF** retriever over your chunks. For each generated
question, the originating `source_chunk_ids` are treated as relevance labels.
Metrics are **baseline retrieval results**, not proof of answer quality.
"""
    )
    examples = db.get_examples(exclude_rejected=True)
    chunks = db.get_chunks(include_duplicates=False) or db.get_chunks()
    if not examples or not chunks:
        st.info("Need both chunks and examples to run the benchmark.")
    else:
        st.caption(f"{len(examples)} queries · {len(chunks)} chunks in index")
        if st.button("Run TF-IDF benchmark", type="primary"):
            with st.spinner("Computing Recall@3 and MRR…"):
                metrics = evaluate_retrieval(examples, chunks, k=3)
                st.session_state.last_retrieval = metrics.model_dump()
                run_id = f"ret_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                db.save_evaluation_run(
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    retrieval_metrics=metrics,
                )
            st.success("Benchmark complete.")

        metrics_data = st.session_state.get("last_retrieval")
        if metrics_data:
            r1, r2, r3 = st.columns(3)
            r1.metric("Recall@3", f"{metrics_data['recall_at_3']:.3f}")
            r2.metric("MRR", f"{metrics_data['mrr']:.3f}")
            r3.metric("Queries", metrics_data["num_queries"])
            st.markdown("#### Metric definitions")
            st.write(f"- **Recall@3:** {metrics_data.get('definition_recall', '')}")
            st.write(f"- **MRR:** {metrics_data.get('definition_mrr', '')}")
            st.warning(metrics_data.get("caveat", ""))
        else:
            st.markdown(
                '<p class="muted">No benchmark run yet in this session.</p>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Export":
    st.subheader("Export Dataset")
    examples = db.get_examples()
    if not examples:
        st.info("Nothing to export.")
    else:
        exclude_rejected = st.checkbox("Exclude rejected examples", value=True)
        include_demo = st.checkbox("Include demo fixtures", value=True)
        bundle = export_bundle(
            examples,
            exclude_rejected=exclude_rejected,
            include_demo=include_demo,
        )
        st.metric("Examples in export", bundle["count"])
        if bundle["count"] == 0:
            st.warning("Export is empty with current filters.")
        else:
            st.markdown("#### Preview (first rows)")
            st.dataframe(
                pd.DataFrame(bundle["preview"]),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Inspect JSONL (first 3 lines)"):
                preview_lines = "\n".join(bundle["jsonl"].splitlines()[:3])
                st.code(preview_lines or "(empty)", language="json")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download JSONL",
                data=bundle["jsonl"].encode("utf-8"),
                file_name=f"rag_dataset_{ts}.jsonl",
                mime="application/jsonl",
                type="primary",
            )
            st.download_button(
                "Download CSV",
                data=bundle["csv"].encode("utf-8"),
                file_name=f"rag_dataset_{ts}.csv",
                mime="text/csv",
            )
