"""
Hybrid retrieval: vector similarity (sqlite-vec) + keyword search (FTS5),
merged with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import os
import struct

import httpx
from dotenv import load_dotenv

from db.connection import get_connection

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    os.getenv("AZURE_DEPLOYMENT", "text-embedding-ada-002"),
)
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

RRF_K = 60


def embed_query(text: str) -> list[float]:
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise RuntimeError("Embedding configuration is missing.")

    url = (
        f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_DEPLOYMENT}/embeddings?api-version={AZURE_API_VERSION}"
    )

    response = httpx.post(
        url,
        headers={"Content-Type": "application/json", "api-key": AZURE_API_KEY},
        json={"input": [text]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def _vector_search(blob: bytes, course: str | None, fetch: int) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if course:
            cur.execute(
                """
                SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course,
                       ce.distance AS distance
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE ce.embedding MATCH ? AND k = ?
                  AND lower(d.course) = lower(?)
                ORDER BY ce.distance
                """,
                (blob, fetch, course),
            )
        else:
            cur.execute(
                """
                SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course,
                       ce.distance AS distance
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE ce.embedding MATCH ? AND k = ?
                ORDER BY ce.distance
                """,
                (blob, fetch),
            )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _fts_search(question: str, course: str | None, fetch: int) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if course:
            cur.execute(
                """
                SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ? AND lower(d.course) = lower(?)
                ORDER BY rank
                LIMIT ?
                """,
                (question, course, fetch),
            )
        else:
            cur.execute(
                """
                SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (question, fetch),
            )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _rrf(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    scores: dict[int, float] = {}
    rows: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, row in enumerate(ranked):
            chunk_id = row["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            rows[chunk_id] = row

    merged = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    result = []
    for chunk_id, score in merged:
        entry = dict(rows[chunk_id])
        entry["score"] = score
        result.append(entry)
    return result


def query(question: str, course: str | None = None, top_k: int = 5) -> list[dict]:
    """
    Return the top-k relevant chunks for a natural language query.
    """
    fetch = max(top_k * 4, top_k)
    embedding = embed_query(question)
    blob = struct.pack(f"{len(embedding)}f", *embedding)

    vector_results = _vector_search(blob, course, fetch)

    try:
        fts_results = _fts_search(question, course, fetch)
    except Exception:
        # FTS can fail on reserved tokens; retrieval should still work via vectors.
        fts_results = []

    return _rrf([vector_results, fts_results])[:top_k]


if __name__ == "__main__":
    import sys

    user_question = sys.argv[1] if len(sys.argv) > 1 else "What is the main topic?"
    course_filter = sys.argv[2] if len(sys.argv) > 2 else None

    for index, result in enumerate(query(user_question, course=course_filter), 1):
        print(f"\n--- Result {index} (score: {result['score']:.4f}) ---")
        print(
            f"Source: {result['document_title']}, page {result['page_number']}, "
            f"course: {result['course']}"
        )
        print(result["text"])
