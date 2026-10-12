# LLM Knowledge Pipeline

AI-powered MVP that converts documents into **traceable RAG datasets**, runs deterministic quality checks, and reports a TF-IDF retrieval baseline.

## Features

1. **Upload** — PDF, DOCX, TXT, Markdown (size/type validation, page provenance for PDFs)
2. **Chunk** — configurable size/overlap, paragraph-aware splitting, stable IDs, duplicate flagging (records kept)
3. **Generate** — Sarvam AI `sarvam-105b` via `/v1/chat/completions` (or **Demo Mode** without a key)
4. **Quality** — schema, duplicates, citations, evidence-quote presence (no LLM judge by default)
5. **Benchmark** — TF-IDF Recall@3 and MRR with originating chunks as labels
6. **Export** — JSONL and CSV with IDs and provenance

## Quick start

```bash
cd "c:\Users\chatt\Downloads\sarvam LLM"
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python sample_data/create_sample.py

# Optional: live generation
copy .env.example .env
# Edit .env and set SARVAM_API_KEY=...

# Or Streamlit secrets: .streamlit/secrets.toml
# SARVAM_API_KEY = "..."

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Demo Mode (no API key)

In the app sidebar go to **Upload & Process** → click **Load Demo Mode**. Fixture examples are labeled `DEMO` / `is_demo=true` and are **not** Sarvam outputs.

## Tests

```bash
pytest -q
```

## Project layout

```
app.py                 # Streamlit UI
src/
  parsing.py           # Document extraction
  chunking.py          # Chunking + duplicate detection
  sarvam.py            # Sarvam API client
  generation.py        # QA dataset generation
  validation.py        # Quality checks
  retrieval.py         # TF-IDF benchmark
  export.py            # JSONL / CSV
  demo.py              # Demo fixtures
  db.py / models.py    # SQLite + Pydantic
tests/                 # pytest suite
sample_data/           # Sample document + creator script
```

## API key safety

- Read only from `SARVAM_API_KEY` env var or Streamlit secrets
- Never rendered in the browser, logs, repo, or exports
- Generation fails gracefully if the key is missing

## Notes

- Evidence quote matching verifies citation integrity; it does **not** prove full answer correctness
- Retrieval metrics are a **lexical baseline**, not end-to-end RAG quality
- Live Sarvam calls require a valid key and network access
