"""
Concept extraction and management.

Phase 1 extracts a bounded concept list for the whole document.
Phase 2 assigns chunk-to-concept links, using a fast heuristic by default.
"""

from __future__ import annotations

import json
import os
import re
import struct

from db.connection import get_connection
from llm import call_llm

MAX_CONCEPTS_PER_CHUNK = 3
MAX_DOCUMENT_CONCEPTS = 15
MIN_DOCUMENT_CONCEPTS = 5
CHUNK_ASSIGNMENT_MODE = os.getenv("CHUNK_CONCEPT_ASSIGNMENT_MODE", "heuristic").lower()


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9_-]+", _normalize(text))


def get_existing_concepts(course: str) -> list[dict]:
    """Return all concepts for this course: {id, name, embedding}."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, embedding FROM concepts WHERE course = ? ORDER BY id",
            (course,),
        )
        rows = []
        for row in cur.fetchall():
            rows.append({
                "id": row["id"],
                "name": row["name"],
                "embedding": _blob_to_vec(row["embedding"]) if row["embedding"] else None,
            })
        return rows
    finally:
        conn.close()


def _insert_concept(name: str, course: str, embedding: list[float]) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO concepts (name, course, embedding) VALUES (?, ?, ?)",
            (name, course, _vec_to_blob(embedding)),
        )
        conn.commit()
        cur.execute("SELECT id FROM concepts WHERE name = ?", (name,))
        return cur.fetchone()["id"]
    finally:
        conn.close()


def _link_document_concept(document_id: int, concept_id: int):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO document_concepts (document_id, concept_id) VALUES (?, ?)",
            (document_id, concept_id),
        )
        conn.commit()
    finally:
        conn.close()


def extract_document_concepts(full_text: str, course: str, embed_fn, document_id: int) -> list[str]:
    """
    Send the full document text to the LLM once and extract a bounded master concept list.
    Inserts concepts into the DB and returns their names.
    """
    trimmed = full_text[:12000]

    system = (
        "You are an academic concept extractor for a study tool. "
        "Identify the key concepts covered in this document. "
        "Respond ONLY with a JSON array of short concept names (1-3 words each), lowercase. "
        "No explanation, no markdown - only the JSON array."
    )

    prompt = (
        f"Document text:\n{trimmed}\n\n"
        f"Extract {MIN_DOCUMENT_CONCEPTS}-{MAX_DOCUMENT_CONCEPTS} key concepts that cover "
        f"the main topics of this document. Be broad - these will be used to categorise "
        f"all sections of the document. Use the same language as the document."
    )

    response = call_llm(prompt, system=system)

    try:
        clean = response.strip().removeprefix("```json").removesuffix("```").strip()
        names = json.loads(clean)
        if isinstance(names, list):
            names = [str(name).lower().strip() for name in names if name][:MAX_DOCUMENT_CONCEPTS]
    except (json.JSONDecodeError, ValueError):
        names = []

    if not names:
        print("  Warning: LLM returned no concepts for document.")
        return []

    print(f"  Master concepts: {names}")

    for name in names:
        embedding = embed_fn(name)
        concept_id = _insert_concept(name, course, embedding)
        _link_document_concept(document_id, concept_id)

    return names


def _assign_via_llm(chunk_text: str, existing_names: list[str]) -> list[str]:
    existing_str = ", ".join(f'"{name}"' for name in existing_names)

    system = (
        "You are a concept tagging assistant for a study tool. "
        "Respond ONLY with a JSON array of strings from the provided list. "
        "No explanation, no new concepts, no markdown."
    )

    prompt = (
        f"Available concepts: [{existing_str}]\n\n"
        f"Text:\n{chunk_text}\n\n"
        f"Select 1-{MAX_CONCEPTS_PER_CHUNK} concepts from the list above that best describe "
        f"what this text is about. Only use concepts from the provided list."
    )

    response = call_llm(prompt, system=system)

    try:
        clean = response.strip().removeprefix("```json").removesuffix("```").strip()
        chosen = json.loads(clean)
        if isinstance(chosen, list):
            valid = [
                str(name).lower().strip()
                for name in chosen
                if str(name).lower().strip() in existing_names
            ]
            return valid[:MAX_CONCEPTS_PER_CHUNK]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _assign_via_heuristic(chunk_text: str, existing: list[dict]) -> list[str]:
    normalized_text = _normalize(chunk_text)
    text_tokens = set(_tokenize(chunk_text))
    ranked: list[tuple[int, int, str]] = []

    for item in existing:
        concept_name = item["name"]
        concept_tokens = _tokenize(concept_name)
        exact_phrase = concept_name in normalized_text
        overlap = len(text_tokens.intersection(concept_tokens))
        if not exact_phrase and overlap == 0:
            continue

        ranked.append((
            2 if exact_phrase else 1,
            overlap,
            concept_name,
        ))

    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [name for _, _, name in ranked[:MAX_CONCEPTS_PER_CHUNK]]


def assign_concepts_for_chunks(chunk_rows: list[dict], course: str) -> dict[int, list[str]]:
    """
    Return {chunk_id: [concept_name, ...]} for many chunks at once.
    Uses heuristics by default because it is dramatically faster on large PDFs.
    """
    existing = get_existing_concepts(course)
    if not existing:
        return {}

    existing_names = [item["name"] for item in existing]
    chosen_by_chunk: dict[int, list[str]] = {}

    for row in chunk_rows:
        chunk_id = row["id"]
        chunk_text = row["text"]
        if CHUNK_ASSIGNMENT_MODE == "llm":
            chosen = _assign_via_llm(chunk_text, existing_names)
        else:
            chosen = _assign_via_heuristic(chunk_text, existing)
        chosen_by_chunk[chunk_id] = chosen

    return chosen_by_chunk
