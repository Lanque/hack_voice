"""Helpers for course/topic views used by the frontend."""

from __future__ import annotations

import os
import re

from db.connection import get_connection

TOPIC_BLACKLIST = {
    "demo course",
    "perf course",
    "smoke test",
    "taltech",
    "current",
    "voltage",
    "electrical machine",
    "veateated",
    "soovitused",
    "töökoht",
    "pädevusala",
    "kasutajaliides",
}
HIDDEN_COURSES = {
    item.strip()
    for item in os.getenv("HIDDEN_COURSES", "Demo Course,Perf Course").split(",")
    if item.strip()
}


def _build_file_url(document_id: int, page: int | None = None) -> str:
    base = f"/api/documents/{document_id}/file"
    if not page:
        return base
    return f"{base}?page={page}#page={page}"


def _clean_excerpt_text(text: str | None) -> str:
    if not text:
        return ""

    cleaned = text.replace("\u00a0", " ")
    cleaned = re.sub(r"[.·•]{4,}\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d+(?:\.\d+)+\s+[A-ZÕÄÖÜ][^\n]{0,120}[.·•]{4,}\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"[.·•]{4,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -:;,.")


def _truncate(text: str | None, limit: int = 240) -> str:
    if not text:
        return ""
    compact = _clean_excerpt_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _fetch_topic_row(topic_id: int) -> dict | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, course
            FROM concepts
            WHERE id = ?
            """,
            (topic_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _is_hidden_course(course: str | None) -> bool:
    return bool(course and course in HIDDEN_COURSES)


def _is_usable_topic(topic_name: str, material_count: int) -> bool:
    if material_count <= 0:
        return False

    normalized = topic_name.strip().lower()
    if normalized in TOPIC_BLACKLIST:
        return False
    if len(normalized) < 4:
        return False
    return True


def list_course_names() -> list[str]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT course
            FROM documents
            WHERE course IS NOT NULL AND trim(course) <> ''
            ORDER BY lower(course)
            """
        )
        return [row["course"] for row in cur.fetchall() if not _is_hidden_course(row["course"])]
    finally:
        conn.close()


def get_stats() -> dict:
    visible_courses = list_course_names()

    conn = get_connection()
    try:
        cur = conn.cursor()
        if visible_courses:
            placeholders = ",".join("?" for _ in visible_courses)
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS sources,
                    COALESCE(SUM(chunk_count), 0) AS chunks
                FROM (
                    SELECT d.id, COUNT(c.id) AS chunk_count
                    FROM documents d
                    LEFT JOIN chunks c ON c.document_id = d.id
                    WHERE d.course IN ({placeholders})
                    GROUP BY d.id
                )
                """,
                visible_courses,
            )
            row = cur.fetchone()
        else:
            row = {"sources": 0, "chunks": 0}
        return {
            "courses": len(visible_courses),
            "sources": row["sources"] or 0,
            "chunks": row["chunks"] or 0,
        }
    finally:
        conn.close()


def get_topic_materials(topic_id: int, limit: int = 8) -> list[dict]:
    topic = _fetch_topic_row(topic_id)
    if not topic or _is_hidden_course(topic.get("course")):
        return []

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.id AS chunk_id, c.text, c.page_number, d.title AS document_title, d.source_file
                 , d.id AS document_id
            FROM chunk_concepts cc
            JOIN chunks c ON c.id = cc.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE cc.concept_id = ?
            ORDER BY d.title, c.page_number, c.id
            LIMIT ?
            """,
            (topic_id, limit),
        )
        rows = [dict(row) for row in cur.fetchall()]

        materials = []
        for rank, row in enumerate(rows, 1):
            cur.execute(
                """
                SELECT co.name
                FROM chunk_concepts cc
                JOIN concepts co ON co.id = cc.concept_id
                WHERE cc.chunk_id = ? AND co.id != ?
                ORDER BY lower(co.name)
                LIMIT 3
                """,
                (row["chunk_id"], topic_id),
            )
            neighbors = [neighbor["name"] for neighbor in cur.fetchall()]
            page = row["page_number"]
            page_label = f"lk {page}" if page else "ilma leheta"
            materials.append(
                {
                    "id": f"{topic_id}:{row['chunk_id']}",
                    "rank": rank,
                    "title": row["document_title"],
                    "source": row["document_title"],
                    "documentId": row["document_id"],
                    "page": page,
                    "quote": _truncate(row["text"], 280),
                    "whyRelevant": (
                        f"See lõik on seotud teemaga \"{topic['name']}\" ja aitab leida "
                        f"algallika õige koha ({page_label})."
                    ),
                    "concepts": [topic["name"], *neighbors],
                    "link": row["source_file"],
                    "fileUrl": _build_file_url(row["document_id"], page),
                    "assets": (
                        [{"title": row["document_title"], "link": _build_file_url(row["document_id"], page)}]
                        if row["source_file"]
                        else []
                    ),
                }
            )
        return materials
    finally:
        conn.close()


def search_topic_materials(topic_id: int, query_text: str, limit: int = 5) -> list[dict]:
    topic = _fetch_topic_row(topic_id)
    if not topic or _is_hidden_course(topic.get("course")):
        return []

    tokens = [token.strip().lower() for token in query_text.split() if token.strip()]
    if not tokens:
        return []

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.id AS chunk_id,
                   c.text,
                   c.page_number,
                   d.id AS document_id,
                   d.title AS document_title
            FROM chunk_concepts cc
            JOIN chunks c ON c.id = cc.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE cc.concept_id = ?
            ORDER BY c.id
            """,
            (topic_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    ranked = []
    lower_query = query_text.lower()
    for row in rows:
        text = row["text"] or ""
        lower_text = _clean_excerpt_text(text).lower()
        if not lower_text:
            continue
        exact_hits = lower_text.count(lower_query)
        token_hits = sum(lower_text.count(token) for token in tokens)
        score = exact_hits * 5 + token_hits
        if score <= 0:
            continue
        ranked.append((score, row))

    ranked.sort(key=lambda item: (-item[0], item[1]["page_number"] or 0, item[1]["chunk_id"]))

    matches = []
    for score, row in ranked[:limit]:
        matches.append({
            "id": f"{topic_id}:{row['chunk_id']}:search",
            "score": score,
            "title": row["document_title"],
            "documentId": row["document_id"],
            "page": row["page_number"],
            "quote": _truncate(row["text"], 320),
            "fileUrl": _build_file_url(row["document_id"], row["page_number"]),
            "where": f"{row['document_title']} | lk {row['page_number'] or '-'}",
        })
    return matches


def _get_topic_document_stats(topic_id: int) -> dict:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(DISTINCT d.id) AS document_count,
                   MIN(c.page_number) AS first_page,
                   MAX(c.page_number) AS last_page
            FROM chunk_concepts cc
            JOIN chunks c ON c.id = cc.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE cc.concept_id = ?
            """,
            (topic_id,),
        )
        row = cur.fetchone()
        return {
            "document_count": row["document_count"] if row else 0,
            "first_page": row["first_page"] if row else None,
            "last_page": row["last_page"] if row else None,
        }
    finally:
        conn.close()


def _get_topic_metrics(topic_id: int) -> dict:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS material_count
            FROM chunk_concepts
            WHERE concept_id = ?
            """,
            (topic_id,),
        )
        counts_row = cur.fetchone()

        cur.execute(
            """
            SELECT c.text
            FROM chunk_concepts cc
            JOIN chunks c ON c.id = cc.chunk_id
            WHERE cc.concept_id = ?
            ORDER BY c.id
            LIMIT 1
            """,
            (topic_id,),
        )
        preview_row = cur.fetchone()

        return {
            "material_count": counts_row["material_count"] if counts_row else 0,
            "preview": _truncate(preview_row["text"], 240) if preview_row else "",
        }
    finally:
        conn.close()


def _get_related_topics(topic_id: int, limit: int = 5) -> list[dict]:
    topic = _fetch_topic_row(topic_id)
    if not topic:
        return []

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT other.id, other.name, COUNT(*) AS overlap_count
            FROM document_concepts current_dc
            JOIN document_concepts other_dc
              ON other_dc.document_id = current_dc.document_id
             AND other_dc.concept_id != current_dc.concept_id
            JOIN concepts other ON other.id = other_dc.concept_id
            WHERE current_dc.concept_id = ?
            GROUP BY other.id, other.name
            ORDER BY overlap_count DESC, lower(other.name)
            LIMIT ?
            """,
            (topic_id, limit),
        )
        return [
            {
                "id": row["id"],
                "title": row["name"],
                "overlap": [f"{row['overlap_count']} ühine allikas"],
            }
            for row in cur.fetchall()
            if row["name"].lower() not in TOPIC_BLACKLIST
        ]
    finally:
        conn.close()


def _build_mode_guides(topic_name: str) -> dict:
    return {
        "learn": {
            "title": f"Õpi teemat \"{topic_name}\"",
            "description": "Alusta viidetest, liigu seejärel mõistete ja näidete juurde.",
            "actions": [
                "Ava esmased viited ja märgi definitsioonid või valemid.",
                "Seleta teema oma sõnadega enne lisa küsimuste küsimist.",
                "Kontrolli arusaamist ühe näite või arvutusülesandega.",
            ],
        },
        "recall": {
            "title": "Recall lecture",
            "description": "Korda teemat etappide kaupa ja too kõrvale seotud lisamaterjalid.",
            "actions": [
                "Võta ette üks alateema korraga.",
                "Püüa meenutada põhiseoseid enne allika uuesti avamist.",
                "Kasuta seotud teemasid, kui mõni mõiste jääb segaseks.",
            ],
        },
        "quiz": {
            "title": "Quiz topic",
            "description": "Harjuta lühikeste avatud vastustega küsimustega ja vaata kohe tagasisidet.",
            "actions": [
                "Vasta esmalt ilma materjale avamata.",
                "Loe tagasiside järel viidatud lõik uuesti läbi.",
                "Märgi nõrgad kohad järgmise õppetsükli jaoks.",
            ],
        },
        "plan": {
            "title": "Study plan",
            "description": "Planeeri kordamine nii, et teema tuleks tagasi õigel hetkel.",
            "actions": [
                "Pane paika sihthinne ja ajapiirang.",
                "Tõsta nõrgemad teemad ettepoole.",
                "Jäta viimaseks päevaks ainult kordamine ja testimine.",
            ],
        },
    }


def build_topic_payload(topic_id: int, include_materials: bool = False) -> dict | None:
    topic = _fetch_topic_row(topic_id)
    if not topic or _is_hidden_course(topic.get("course")):
        return None

    metrics = _get_topic_metrics(topic_id)
    if not _is_usable_topic(topic["name"], metrics["material_count"]):
        return None

    materials = get_topic_materials(topic_id) if include_materials else []
    doc_stats = _get_topic_document_stats(topic_id)
    related_topics = _get_related_topics(topic_id)
    subtopics = [item["title"] for item in related_topics[:4]]

    reading_list = [
        {
            "source": material["source"],
            "pages": material["page"] or "-",
            "reason": material["whyRelevant"],
        }
        for material in materials[:4]
    ]

    document_count = doc_stats["document_count"] or 1
    first_page = doc_stats["first_page"]
    last_page = doc_stats["last_page"]
    page_span = ""
    if first_page and last_page:
        page_span = f" Leheküljed {first_page}-{last_page} sisaldavad enim seotud kohti."
    elif first_page:
        page_span = f" Esimene seotud viide on lehel {first_page}."

    summary = (
        f"Teema \"{topic['name']}\" koondab {metrics['material_count']} viidet "
        f"{document_count} allikast kursusel {topic['course']}." + page_span
    )

    return {
        "id": str(topic["id"]),
        "title": topic["name"],
        "name": topic["name"],
        "courseId": topic["course"] or "General",
        "content": summary,
        "summary": summary,
        "whyItMatters": (
            f"Teema \"{topic['name']}\" aitab siduda kursuse materjalid, kordamisküsimused "
            f"ja harjutused üheks navigeeritavaks teadmiste sõlmeks. Fookus on algallikatele jõudmisel, "
            f"mitte automaatsel kokkuvõttel."
        ),
        "subtopics": subtopics,
        "materials": materials,
        "matches": [],
        "readingList": reading_list,
        "recallPrompts": [
            f"Mis on teema \"{topic['name']}\" põhisisu ilma materjali vaatamata?",
            f"Millise teise kursuse või alateemaga \"{topic['name']}\" kõige rohkem haakub?",
            f"Milline näide või ülesandetüüp kontrollib kõige paremini selle teema mõistmist?",
        ],
        "relatedTopics": related_topics,
        "modeGuides": _build_mode_guides(topic["name"]),
        "materialCount": metrics["material_count"],
    }


def list_topics(course: str | None = None) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if course:
            cur.execute(
                """
                SELECT id
                FROM concepts
                WHERE lower(course) = lower(?)
                ORDER BY lower(name)
                """,
                (course,),
            )
        else:
            cur.execute("SELECT id FROM concepts ORDER BY lower(name)")
        topic_ids = [row["id"] for row in cur.fetchall()]
    finally:
        conn.close()

    topics = []
    for topic_id in topic_ids:
        payload = build_topic_payload(topic_id, include_materials=False)
        if payload:
            topics.append(payload)

    topics.sort(key=lambda item: (item["courseId"].lower(), -item["materialCount"], item["title"].lower()))
    return topics
