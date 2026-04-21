"""Estonian summary generation from retrieved chunks."""
from __future__ import annotations

from db.connection import get_connection
from llm import call_llm
from query import query as retrieve


def get_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []
    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
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
    order = {cid: idx for idx, cid in enumerate(chunk_ids)}
    return sorted(rows, key=lambda r: order.get(r["id"], len(order)))


def _format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['document_title']}, lk {chunk['page_number']}\n{chunk['text']}")
    return "\n\n".join(parts)


def summarize_chunks(chunks: list[dict], topic: str | None = None) -> dict:
    if not chunks:
        return {
            "summary": "Kokkuvõtet ei saanud koostada, sest allikmaterjali ei leitud.",
            "sources": [],
        }

    system = (
        "Sa oled eesti keeles õppematerjalide kokkuvõtja. "
        "Kasuta AINULT etteantud allikaid. "
        "Kirjuta selge, kompaktne õppija jaoks sobiv kokkuvõte eesti keeles. "
        "Kui allikates on vastuolusid või infot ei ole piisavalt, ütle seda selgelt. "
        "Lisa lause lõpus viited kujul [1], [2]."
    )

    topic_hint = f"Teema: {topic}\n\n" if topic else ""
    prompt = (
        f"{topic_hint}"
        f"Allikad:\n{_format_sources(chunks)}\n\n"
        "Koosta 1-2 lühikest lõiku, mis võtavad peamise sisu kokku. "
        "Too välja peamised mõisted, definitsioonid ja olulised seosed. "
        "Ära lisa midagi, mida allikates ei ole."
    )

    return {
        "summary": call_llm(prompt, system=system),
        "sources": chunks,
    }


def build_summary(
    topic: str | None = None,
    course: str | None = None,
    top_k: int = 5,
    chunk_ids: list[int] | None = None,
) -> dict:
    if chunk_ids:
        chunks = get_chunks_by_ids(chunk_ids)
    elif topic:
        chunks = retrieve(topic, course=course, top_k=top_k)
    else:
        chunks = []
    return summarize_chunks(chunks, topic=topic)
