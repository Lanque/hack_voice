"""
/quiz-topic mode

Flow:
  1. Find chunks linked to the requested concept
  2. Generate N multiple-choice or open questions from those chunks
  3. Ask questions one by one in a CLI loop
  4. Evaluate each answer with the LLM, citing the source
  5. Print a score summary at the end
"""

import json
from db.connection import get_connection
from llm import call_llm


# ---------------------------------------------------------------------------
# 1. Fetch chunks for a concept
# ---------------------------------------------------------------------------

def get_chunks_for_concept(concept_name: str, course: str = None) -> list[dict]:
    """Return chunks linked to a concept, optionally filtered by course.
    Matches if the stored concept name contains the query or vice versa."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        pattern = f"%{concept_name.lower()}%"
        if course:
            cur.execute(
                """
                SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course
                FROM chunks c
                JOIN chunk_concepts cc ON cc.chunk_id = c.id
                JOIN concepts co ON co.id = cc.concept_id
                JOIN documents d ON d.id = c.document_id
                WHERE (lower(co.name) LIKE ? OR lower(?) LIKE '%' || lower(co.name) || '%')
                  AND lower(d.course) = lower(?)
                """,
                (pattern, concept_name.lower(), course),
            )
        else:
            cur.execute(
                """
                SELECT c.id, c.text, c.page_number, d.title AS document_title, d.course
                FROM chunks c
                JOIN chunk_concepts cc ON cc.chunk_id = c.id
                JOIN concepts co ON co.id = cc.concept_id
                JOIN documents d ON d.id = c.document_id
                WHERE lower(co.name) LIKE ? OR lower(?) LIKE '%' || lower(co.name) || '%'
                """,
                (pattern, concept_name.lower()),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def list_concepts(course: str = None) -> list[str]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if course:
            cur.execute(
                "SELECT name FROM concepts WHERE lower(course) = lower(?) ORDER BY name",
                (course,),
            )
        else:
            cur.execute("SELECT name FROM concepts ORDER BY name")
        return [r["name"] for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. Generate questions
# ---------------------------------------------------------------------------

def generate_questions(chunks: list[dict], n: int = 5, concept: str = "") -> list[dict]:
    """
    Ask LLM to generate N questions from the provided chunks.
    Returns list of {question, source_text, page_number, document_title, concept}
    """
    sources = "\n\n".join(
        f"[Source {i+1} — {c['document_title']}, page {c['page_number']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system = (
        "You are a study quiz generator. Generate questions strictly based on the provided sources. "
        "Respond ONLY with a JSON array. Each item must have: "
        "\"question\" (string), \"source_index\" (1-based integer referencing the source). "
        "No answers, no explanations — only the JSON array."
    )

    prompt = (
        f"Sources:\n{sources}\n\n"
        f"Generate exactly {n} questions that test understanding of these sources. "
        f"Mix factual recall and conceptual questions. "
        f"Each question must be answerable from the sources."
    )

    response = call_llm(prompt, system=system)

    try:
        clean = response.strip().strip("```json").strip("```").strip()
        items = json.loads(clean)
        if not isinstance(items, list):
            return _fallback_questions(chunks, n=n, concept=concept)
    except (json.JSONDecodeError, ValueError):
        return _fallback_questions(chunks, n=n, concept=concept)

    questions = []
    for item in items[:n]:
        idx = item.get("source_index", 1) - 1
        source_chunk = chunks[idx] if 0 <= idx < len(chunks) else chunks[0]
        questions.append({
            "question": item.get("question", ""),
            "source_text": source_chunk["text"],
            "page_number": source_chunk["page_number"],
            "document_title": source_chunk["document_title"],
            "concept": concept,
        })
    return questions or _fallback_questions(chunks, n=n, concept=concept)


def _fallback_questions(chunks: list[dict], n: int = 5, concept: str = "") -> list[dict]:
    questions = []
    templates = [
        "Selgita oma sõnadega, mida allikas ütleb teema \"{concept}\" kohta.",
        "Millised on selle lõigu järgi teema \"{concept}\" peamised omadused või seosed?",
        "Too selle allika põhjal välja üks oluline definitsioon või väide teema \"{concept}\" kohta.",
    ]

    for index, source_chunk in enumerate(chunks[:n]):
        template = templates[index % len(templates)]
        questions.append({
            "question": template.format(concept=concept or "see teema"),
            "source_text": source_chunk["text"],
            "page_number": source_chunk["page_number"],
            "document_title": source_chunk["document_title"],
            "concept": concept,
        })
    return questions


# ---------------------------------------------------------------------------
# 3. Evaluate answer
# ---------------------------------------------------------------------------

def evaluate_answer(question: str, student_answer: str, source_text: str) -> dict:
    """
    LLM evaluates the student's answer against the source.
    Returns {correct: bool, feedback: str}
    """
    system = (
        "You are a study tutor evaluating a student's answer. "
        "Base your evaluation strictly on the provided source text. "
        "Respond ONLY with a JSON object: {\"correct\": true/false, \"feedback\": \"...\"}. "
        "Feedback should be 1-2 sentences: confirm what was right, correct what was wrong, "
        "and cite the key fact from the source."
    )

    prompt = (
        f"Source text:\n{source_text}\n\n"
        f"Question: {question}\n"
        f"Student answer: {student_answer}\n\n"
        f"Is the answer correct or substantially correct based on the source? "
        f"Provide brief feedback."
    )

    response = call_llm(prompt, system=system)

    try:
        clean = response.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        return {
            "correct": bool(result.get("correct", False)),
            "feedback": str(result.get("feedback", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return {"correct": False, "feedback": response.strip()}


def evaluate_answers_batch(items: list[dict]) -> list[dict]:
    """
    Evaluate many answers in one LLM call.
    Each item must contain: question, student_answer, source_text.
    Returns list of {correct, feedback}.
    """
    if not items:
        return []

    system = (
        "You are a study tutor evaluating student answers. "
        "Base every evaluation strictly on the provided source text. "
        "Respond ONLY with a JSON array. Each item must be an object with "
        "\"correct\" (boolean) and \"feedback\" (string). "
        "Keep feedback to 1-2 short sentences and mention the key fact from the source."
    )

    blocks = []
    for index, item in enumerate(items, 1):
        blocks.append(
            f"[Item {index}]\n"
            f"Source text:\n{item['source_text']}\n\n"
            f"Question: {item['question']}\n"
            f"Student answer: {item['student_answer']}\n"
        )

    prompt = (
        "Evaluate all items below in order.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn exactly one JSON array item per input item, in the same order."
    )

    response = call_llm(prompt, system=system)

    try:
        clean = response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list) and len(parsed) == len(items):
            return [
                {
                    "correct": bool(result.get("correct", False)),
                    "feedback": str(result.get("feedback", "")),
                }
                for result in parsed
            ]
    except (json.JSONDecodeError, ValueError):
        pass

    return [
        evaluate_answer(item["question"], item["student_answer"], item["source_text"])
        for item in items
    ]


# ---------------------------------------------------------------------------
# 5. Save result
# ---------------------------------------------------------------------------

def save_quiz_result(concept: str, course: str, score: int, total: int):
    """Save quiz result and flag if it's a new personal best."""
    if total == 0:
        return

    pct = round(score / total * 100, 1)

    conn = get_connection()
    try:
        cur = conn.cursor()

        # Find previous best for this concept
        cur.execute(
            "SELECT MAX(percentage) FROM quiz_results WHERE concept = ? AND (course = ? OR course IS NULL)",
            (concept, course),
        )
        row = cur.fetchone()
        prev_best = row[0] if row and row[0] is not None else -1
        is_best = 1 if pct >= prev_best else 0

        # If new best, clear old best flags
        if is_best:
            cur.execute(
                "UPDATE quiz_results SET best = 0 WHERE concept = ? AND (course = ? OR course IS NULL)",
                (concept, course),
            )

        cur.execute(
            "INSERT INTO quiz_results (concept, course, score, total, percentage, best) VALUES (?, ?, ?, ?, ?, ?)",
            (concept, course, score, total, pct, is_best),
        )
        conn.commit()
        return pct, is_best
    finally:
        conn.close()


def get_weak_concepts(course: str = None, limit: int = 5) -> list[dict]:
    """Return concepts with the lowest best percentage, for study recommendations."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if course:
            cur.execute(
                """
                SELECT concept, MAX(percentage) AS best_pct, COUNT(*) AS attempts
                FROM quiz_results
                WHERE course = ?
                GROUP BY concept
                ORDER BY best_pct ASC
                LIMIT ?
                """,
                (course, limit),
            )
        else:
            cur.execute(
                """
                SELECT concept, MAX(percentage) AS best_pct, COUNT(*) AS attempts
                FROM quiz_results
                GROUP BY concept
                ORDER BY best_pct ASC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6. Quiz loop
# ---------------------------------------------------------------------------

def run_quiz(concept: str, course: str = None, n_questions: int = 5):
    print(f"\n=== Quiz: {concept} ===\n")

    chunks = get_chunks_for_concept(concept, course)
    if not chunks:
        available = list_concepts(course)
        print(f"No material found for concept '{concept}'.")
        if available:
            print(f"Available concepts: {', '.join(available)}")
        return

    print(f"Found {len(chunks)} relevant chunks. Generating {n_questions} questions...\n")
    questions = generate_questions(chunks, n=n_questions, concept=concept)

    if not questions:
        print("Could not generate questions from this material.")
        return

    score = 0
    missed_concepts: list[str] = []

    for i, q in enumerate(questions, 1):
        print(f"Q{i}/{len(questions)}: {q['question']}")
        print(f"  [Source: {q['document_title']}, page {q['page_number']}]")

        try:
            answer = input("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nQuiz cancelled.")
            break

        if not answer:
            print("  Skipped.\n")
            missed_concepts.append(q.get("concept", concept))
            continue

        result = evaluate_answer(q["question"], answer, q["source_text"])

        if result["correct"]:
            score += 1
            print(f"  CORRECT. {result['feedback']}\n")
        else:
            print(f"  INCORRECT. {result['feedback']}\n")
            missed_concepts.append(q.get("concept", concept))

    total = len(questions)
    print(f"=== Result: {score}/{total} correct ===\n")

    pct, is_best = save_quiz_result(concept, course or "", score, total)
    if is_best:
        print(f"New personal best: {pct}%!\n")
    else:
        print(f"Score: {pct}%\n")

    if missed_concepts:
        # Deduplicate while preserving order
        seen = set()
        unique_missed = [c for c in missed_concepts if not (c in seen or seen.add(c))]

        pct = int(score / total * 100) if total else 0
        if score == total:
            print("Perfect score! Great work.")
        elif pct >= 60:
            print(f"Good effort ({pct}%)! A few areas to review:")
        else:
            print(f"Keep at it ({pct}%)! Focus on these topics:")

        for c in unique_missed:
            print(f"  - {c}")

        print(
            f"\nRun the quiz again on any of these, or use RAG to review:\n"
            f"  .venv\\Scripts\\python.exe quiz.py \"<concept>\" \"{course or ''}\""
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quiz.py <concept> [course] [n_questions]")
        print("\nAvailable concepts:")
        for c in list_concepts():
            print(f"  - {c}")
        sys.exit(0)

    concept_arg = sys.argv[1]
    course_arg = sys.argv[2] if len(sys.argv) > 2 else None
    n_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    run_quiz(concept_arg, course=course_arg, n_questions=n_arg)
