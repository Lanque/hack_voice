"""
Chunk-based summary generation for the web API.
"""

from __future__ import annotations

from db.connection import get_connection
from llm import call_llm
from query import query as retrieve


def _format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk['document_title']}, page {chunk['page_number']}\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def get_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []

    placeholders = ",".join("?" for _ in chunk_ids)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        )
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    order = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
    return sorted(rows, key=lambda row: order.get(row["id"], len(order)))


def summarize_chunks(chunks: list[dict], topic: str | None = None) -> dict:
    """
    Create an Estonian study summary from retrieved chunks.
    Returns {summary, sources}.
    """
    if not chunks:
        return {"summary": "Kokkuvotet ei saanud koostada, sest allikmaterjali ei leitud.", "sources": []}

    sources_text = _format_sources(chunks)
    topic_hint = f"Teema: {topic}\n\n" if topic else ""

    system = (
        "Sa oled eesti keeles oppematerjalide kokkuvotja. "
        "Kasuta AINULT etteantud allikaid. "
        "Kirjuta selge, kompaktne oppija jaoks sobiv kokkuvote eesti keeles. "
        "Kui allikates on vastuolusid voi infot ei ole piisavalt, utle seda selgelt. "
        "Lisa lause lopus viited kujul [1], [2]."
    )

    prompt = (
        f"{topic_hint}"
        f"Allikad:\n{sources_text}\n\n"
        "Koosta 1-2 luhikest loiku, mis votavad peamise sisu kokku. "
        "Too valja peamised moisted, definitsioonid ja olulised seosed. "
        "Ara lisa midagi, mida allikates ei ole."
    )

    return {
        "summary": call_llm(prompt, system=system),
        "sources": chunks,
    }


def build_summary(topic: str | None = None, course: str | None = None, top_k: int = 5,
                  chunk_ids: list[int] | None = None) -> dict:
    if chunk_ids:
        chunks = get_chunks_by_ids(chunk_ids)
    elif topic:
        chunks = retrieve(topic, course=course, top_k=top_k)
    else:
        chunks = []

    return summarize_chunks(chunks, topic=topic)
