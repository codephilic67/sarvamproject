"""SQLite persistence for documents, chunks, examples, and evaluation results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from src.models import (
    ChunkRecord,
    DocumentRecord,
    PipelineStats,
    QAExample,
    QualityReport,
    RetrievalMetrics,
    ValidationStatus,
)

DEFAULT_DB_PATH = Path("data") / "pipeline.db"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(raw: Optional[str], default: Any = None) -> Any:
    if raw is None or raw == "":
        return default if default is not None else []
    return json.loads(raw)


class Database:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    page_count INTEGER DEFAULT 0,
                    char_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'uploaded',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    is_duplicate INTEGER DEFAULT 0,
                    duplicate_of TEXT,
                    pages_json TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_pages TEXT DEFAULT '[]',
                    char_start INTEGER DEFAULT 0,
                    char_end INTEGER DEFAULT 0,
                    token_estimate INTEGER DEFAULT 0,
                    content_hash TEXT DEFAULT '',
                    is_duplicate INTEGER DEFAULT 0,
                    duplicate_of TEXT,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS examples (
                    example_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_document TEXT NOT NULL,
                    source_chunk_ids TEXT NOT NULL,
                    source_pages TEXT DEFAULT '[]',
                    evidence_quote TEXT NOT NULL,
                    question_type TEXT DEFAULT 'other',
                    generation_model TEXT NOT NULL,
                    validation_status TEXT DEFAULT 'pending',
                    quality_flags TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    is_demo INTEGER DEFAULT 0,
                    FOREIGN KEY (source_document) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    quality_report TEXT,
                    retrieval_metrics TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_examples_doc ON examples(source_document);
                CREATE INDEX IF NOT EXISTS idx_examples_status ON examples(validation_status);
                CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
                """
            )

    # ── Documents ──────────────────────────────────────────────────────────

    def insert_document(
        self, doc: DocumentRecord, pages: list[dict[str, Any]] | None = None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_type, file_size, content_hash,
                    page_count, char_count, status, error_message, created_at,
                    is_duplicate, duplicate_of, pages_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.document_id,
                    doc.filename,
                    doc.file_type,
                    doc.file_size,
                    doc.content_hash,
                    doc.page_count,
                    doc.char_count,
                    doc.status,
                    doc.error_message,
                    doc.created_at,
                    int(doc.is_duplicate),
                    doc.duplicate_of,
                    _json_dumps(pages or []),
                ),
            )

    def update_document_status(
        self, document_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ? WHERE document_id = ?",
                (status, error_message, document_id),
            )

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def get_document_pages(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT pages_json FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        if not row:
            return []
        return _json_loads(row["pages_json"], [])

    def list_documents(self) -> list[DocumentRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def find_document_by_hash(self, content_hash: str) -> Optional[DocumentRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND is_duplicate = 0 "
                "ORDER BY created_at ASC LIMIT 1",
                (content_hash,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            content_hash=row["content_hash"],
            page_count=row["page_count"],
            char_count=row["char_count"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            is_duplicate=bool(row["is_duplicate"]),
            duplicate_of=row["duplicate_of"],
        )

    # ── Chunks ─────────────────────────────────────────────────────────────

    def insert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, text, source_pages,
                    char_start, char_end, token_estimate, content_hash,
                    is_duplicate, duplicate_of, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.text,
                        _json_dumps(c.source_pages),
                        c.char_start,
                        c.char_end,
                        c.token_estimate,
                        c.content_hash,
                        int(c.is_duplicate),
                        c.duplicate_of,
                        _json_dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )

    def get_chunks(
        self, document_id: Optional[str] = None, include_duplicates: bool = True
    ) -> list[ChunkRecord]:
        query = "SELECT * FROM chunks"
        params: list[Any] = []
        clauses: list[str] = []
        if document_id:
            clauses.append("document_id = ?")
            params.append(document_id)
        if not include_duplicates:
            clauses.append("is_duplicate = 0")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY document_id, chunk_index"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def find_chunk_by_hash(self, content_hash: str) -> Optional[ChunkRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE content_hash = ? AND is_duplicate = 0 "
                "ORDER BY chunk_index ASC LIMIT 1",
                (content_hash,),
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            source_pages=_json_loads(row["source_pages"], []),
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_estimate=row["token_estimate"],
            content_hash=row["content_hash"],
            is_duplicate=bool(row["is_duplicate"]),
            duplicate_of=row["duplicate_of"],
            metadata=_json_loads(row["metadata"], {}),
        )

    # ── Examples ───────────────────────────────────────────────────────────

    def insert_examples(self, examples: list[QAExample]) -> None:
        if not examples:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO examples (
                    example_id, question, answer, source_document, source_chunk_ids,
                    source_pages, evidence_quote, question_type, generation_model,
                    validation_status, quality_flags, created_at, is_demo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.example_id,
                        e.question,
                        e.answer,
                        e.source_document,
                        _json_dumps(e.source_chunk_ids),
                        _json_dumps(e.source_pages),
                        e.evidence_quote,
                        e.question_type.value
                        if hasattr(e.question_type, "value")
                        else str(e.question_type),
                        e.generation_model,
                        e.validation_status.value
                        if hasattr(e.validation_status, "value")
                        else str(e.validation_status),
                        _json_dumps(e.quality_flags),
                        e.created_at,
                        int(e.is_demo),
                    )
                    for e in examples
                ],
            )

    def update_example(self, example: QAExample) -> None:
        self.insert_examples([example])

    def get_examples(
        self,
        document_id: Optional[str] = None,
        status: Optional[str] = None,
        exclude_rejected: bool = False,
    ) -> list[QAExample]:
        query = "SELECT * FROM examples"
        params: list[Any] = []
        clauses: list[str] = []
        if document_id:
            clauses.append("source_document = ?")
            params.append(document_id)
        if status:
            clauses.append("validation_status = ?")
            params.append(status)
        if exclude_rejected:
            clauses.append("validation_status != ?")
            params.append(ValidationStatus.REJECTED.value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_example(r) for r in rows]

    def _row_to_example(self, row: sqlite3.Row) -> QAExample:
        return QAExample(
            example_id=row["example_id"],
            question=row["question"],
            answer=row["answer"],
            source_document=row["source_document"],
            source_chunk_ids=_json_loads(row["source_chunk_ids"], []),
            source_pages=_json_loads(row["source_pages"], []),
            evidence_quote=row["evidence_quote"],
            question_type=row["question_type"],
            generation_model=row["generation_model"],
            validation_status=row["validation_status"],
            quality_flags=_json_loads(row["quality_flags"], []),
            created_at=row["created_at"],
            is_demo=bool(row["is_demo"]),
        )

    # ── Evaluation runs ────────────────────────────────────────────────────

    def save_evaluation_run(
        self,
        run_id: str,
        created_at: str,
        quality_report: Optional[QualityReport] = None,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs
                (run_id, created_at, quality_report, retrieval_metrics)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    _json_dumps(quality_report.model_dump()) if quality_report else None,
                    _json_dumps(retrieval_metrics.model_dump())
                    if retrieval_metrics
                    else None,
                ),
            )

    def get_latest_evaluation(self) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "quality_report": _json_loads(row["quality_report"], None),
            "retrieval_metrics": _json_loads(row["retrieval_metrics"], None),
        }

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_stats(self) -> PipelineStats:
        with self.connect() as conn:
            docs = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
            dup_docs = conn.execute(
                "SELECT COUNT(*) AS c FROM documents WHERE is_duplicate = 1"
            ).fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
            dup_chunks = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE is_duplicate = 1"
            ).fetchone()["c"]
            examples = conn.execute("SELECT COUNT(*) AS c FROM examples").fetchone()[
                "c"
            ]
            passed = conn.execute(
                "SELECT COUNT(*) AS c FROM examples WHERE validation_status = ?",
                (ValidationStatus.PASSED.value,),
            ).fetchone()["c"]
            flagged = conn.execute(
                "SELECT COUNT(*) AS c FROM examples WHERE validation_status = ?",
                (ValidationStatus.FLAGGED.value,),
            ).fetchone()["c"]
            rejected = conn.execute(
                "SELECT COUNT(*) AS c FROM examples WHERE validation_status = ?",
                (ValidationStatus.REJECTED.value,),
            ).fetchone()["c"]
        return PipelineStats(
            documents=docs,
            chunks=chunks,
            examples=examples,
            passed_examples=passed,
            flagged_examples=flagged,
            rejected_examples=rejected,
            duplicate_documents=dup_docs,
            duplicate_chunks=dup_chunks,
        )

    def clear_all(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                DELETE FROM examples;
                DELETE FROM chunks;
                DELETE FROM evaluation_runs;
                DELETE FROM documents;
                """
            )
