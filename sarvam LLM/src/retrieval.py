"""Lexical TF-IDF retrieval baseline with Recall@3 and MRR."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import ChunkRecord, QAExample, RetrievalMetrics


def build_tfidf_index(
    chunks: list[ChunkRecord],
) -> tuple[TfidfVectorizer, np.ndarray, list[str]]:
    """Fit TF-IDF on chunk texts. Returns vectorizer, matrix, chunk_id list."""
    if not chunks:
        raise ValueError("Cannot build retrieval index with zero chunks.")
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, ids


def retrieve_top_k(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    chunk_ids: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()
    if len(scores) == 0:
        return []
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 3,
) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return 1.0 if top & relevant_ids else 0.0


def reciprocal_rank(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    examples: list[QAExample],
    chunks: list[ChunkRecord],
    k: int = 3,
) -> RetrievalMetrics:
    """
    For each generated question, originating source_chunk_ids are relevance labels.
    Computes Recall@k and MRR on the same query set.
    """
    usable = [e for e in examples if e.question.strip() and e.source_chunk_ids]
    if not usable or not chunks:
        return RetrievalMetrics(recall_at_3=0.0, mrr=0.0, num_queries=0)

    vectorizer, matrix, chunk_ids = build_tfidf_index(chunks)
    recalls: list[float] = []
    rrs: list[float] = []

    for ex in usable:
        ranked = retrieve_top_k(
            ex.question, vectorizer, matrix, chunk_ids, k=max(k, 10)
        )
        ranked_ids = [cid for cid, _ in ranked]
        relevant = set(ex.source_chunk_ids)
        recalls.append(recall_at_k(ranked_ids, relevant, k=k))
        rrs.append(reciprocal_rank(ranked_ids, relevant))

    n = len(usable)
    return RetrievalMetrics(
        recall_at_3=float(sum(recalls) / n),
        mrr=float(sum(rrs) / n),
        num_queries=n,
    )


def format_metrics_summary(metrics: RetrievalMetrics) -> str:
    return (
        f"Queries evaluated: {metrics.num_queries}\n"
        f"Recall@3: {metrics.recall_at_3:.3f}\n"
        f"MRR: {metrics.mrr:.3f}\n"
        f"Note: {metrics.caveat}"
    )
