"""
Easels API — FastAPI wrapper around the RAG study companion.

Endpoints:
  POST /ingest              — upload a PDF and ingest it
  GET  /concepts            — list all concepts, optionally filtered by course
  GET  /courses             — list all courses
  POST /rag                 — RAG query: retrieve + generate grounded answer
  POST /quiz/generate       — generate quiz questions for a concept
  POST /quiz/evaluate       — evaluate a single student answer
  POST /quiz/result         — save a completed quiz result
  GET  /quiz/weak           — get weakest concepts by quiz history
"""

import json
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ingest import ingest
from rag import rag, rag_stream
from summary import build_summary
from speech import synthesize
from plan import generate_plan
from homework_helper import homework_help
from sm2 import update_sm2, get_next_concept
from llm import call_llm
from quiz import list_concepts as _list_concepts
from quiz import (
    get_chunks_for_concept,
    generate_questions,
    evaluate_answer,
    evaluate_answers_batch,
    save_quiz_result,
    get_weak_concepts,
)
from catalog import (
    build_topic_payload,
    get_stats,
    get_topic_materials,
    list_course_names,
    list_topics,
    search_topic_materials,
)

app = FastAPI(title="Easels API", version="0.1.0")
MEDIA_DIR = os.getenv("MEDIA_DIR", "media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RAGRequest(BaseModel):
    question: str
    course: str | None = None
    top_k: int = 5
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]


class QuizGenerateRequest(BaseModel):
    concept: str
    course: str | None = None
    n_questions: int = 5


class QuizEvaluateRequest(BaseModel):
    question: str
    student_answer: str
    source_text: str


class QuizResultRequest(BaseModel):
    concept: str
    course: str | None = None
    score: int
    total: int


class QuizBatchEvaluateItem(BaseModel):
    question: str
    student_answer: str
    source_text: str


class QuizBatchEvaluateRequest(BaseModel):
    items: list[QuizBatchEvaluateItem]


class PlanRequest(BaseModel):
    course: str | None = None
    days_until_exam: int
    target_grade: str = "5"
    limit_weak: int = 10


class HomeworkRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "question": "Miks kasutatakse alalisvooluliine pikkade kaabelliinide korral?",
        "course": "Elektrisüsteem",
    }}}

    question: str
    course: str | None = None
    top_k: int = 5


class SM2UpdateRequest(BaseModel):
    concept: str
    course: str
    quality: int  # 0-5


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: list[str] = []
    systemPrompt: str | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", summary="Upload and ingest a PDF")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    course: str = Form(...),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(MEDIA_DIR, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (file.filename or "upload.pdf"))
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    stored_path = os.path.abspath(os.path.join(MEDIA_DIR, stored_name))

    with open(stored_path, "wb") as target:
        shutil.copyfileobj(file.file, target)

    try:
        ingest(stored_path, title, course)
    except Exception:
        if os.path.exists(stored_path):
            os.unlink(stored_path)
        raise

    return {"status": "ok", "title": title, "course": course}


# ---------------------------------------------------------------------------
# Concepts & Courses
# ---------------------------------------------------------------------------

@app.get("/concepts", summary="List all concepts")
def get_concepts(course: str | None = None):
    return {"concepts": _list_concepts(course)}


@app.get("/courses", summary="List all courses")
def get_courses():
    return {"courses": list_course_names()}


@app.get("/courses/{course_id}/topics", summary="List topics for a course")
def get_course_topics(course_id: str):
    return {"topics": list_topics(course_id)}


@app.get("/stats", summary="Frontend stats card data")
def stats():
    return get_stats()


@app.get("/topics", summary="List topic cards for the frontend")
def get_topics(course: str | None = None):
    return {"topics": list_topics(course)}


@app.get("/topics/{topic_id}", summary="Get topic detail")
def get_topic(topic_id: int):
    payload = build_topic_payload(topic_id, include_materials=False)
    if not payload:
        raise HTTPException(status_code=404, detail="Topic not found.")
    return payload


@app.get("/topics/{topic_id}/materials", summary="Get topic source cards")
def get_topic_source_cards(topic_id: int):
    if not build_topic_payload(topic_id, include_materials=False):
        raise HTTPException(status_code=404, detail="Topic not found.")
    return get_topic_materials(topic_id)


@app.get("/topics/{topic_id}/search", summary="Search within a topic's source material")
def search_topic_source_cards(topic_id: int, q: str, limit: int = 5):
    if not q.strip():
        return {"matches": []}
    if not build_topic_payload(topic_id, include_materials=False):
        raise HTTPException(status_code=404, detail="Topic not found.")
    return {"matches": search_topic_materials(topic_id, q, limit=limit)}


@app.get("/documents/{document_id}/file", summary="Open original uploaded document")
def get_document_file(document_id: int, page: int | None = None):
    from db.connection import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT title, source_file FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        if row and row["source_file"] and os.path.exists(row["source_file"]):
            filename = os.path.basename(row["source_file"])
            return FileResponse(row["source_file"], media_type="application/pdf", filename=filename)

        cur.execute(
            """
            SELECT page_number, text
            FROM chunks
            WHERE document_id = ?
            ORDER BY COALESCE(page_number, 0), id
            """,
            (document_id,),
        )
        chunks = cur.fetchall()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Document file not found.")
    if not chunks:
        raise HTTPException(status_code=404, detail="Stored document file is missing.")

    pages: dict[int | None, list[str]] = {}
    for chunk in chunks:
        pages.setdefault(chunk["page_number"], []).append(chunk["text"])

    lines = [
        f"{row['title']}",
        "",
        "Original PDF is missing, so this is a reconstructed text export from stored chunks.",
        "",
    ]
    selected_pages = pages.items()
    if page is not None and page in pages:
        selected_pages = [(page, pages[page])]

    for page_number, page_chunks in selected_pages:
        label = page_number if page_number is not None else "-"
        lines.append(f"=== Page {label} ===")
        lines.append("")
        lines.append("\n\n".join(part.strip() for part in page_chunks if part and part.strip()))
        lines.append("")

    filename = f"{row['title']}.txt"
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return PlainTextResponse("\n".join(lines).strip(), headers=headers)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@app.post("/rag", response_model=RAGResponse, summary="Ask a question")
def ask(req: RAGRequest):
    result = rag(req.question, course=req.course, top_k=req.top_k, history=req.history)
    return result


@app.post("/chat", summary="Context-aware freeform chat used by the frontend editor")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    transcript = []
    for message in req.messages[-12:]:
        role = "User" if message.role == "user" else "Assistant"
        transcript.append(f"{role}: {message.content}")

    context_text = "\n\n".join(
        block.strip() for block in req.context if isinstance(block, str) and block.strip()
    )
    prompt_parts = []
    if context_text:
        prompt_parts.append(f"Context blocks:\n{context_text}")
    prompt_parts.append("Conversation:\n" + "\n".join(transcript))
    prompt_parts.append("Continue the conversation helpfully and concisely.")

    try:
        reply = call_llm(
            "\n\n".join(prompt_parts),
            system=req.systemPrompt
            or "You are a helpful study assistant. Use the provided context when it is relevant.",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {"reply": reply}


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

@app.post("/quiz/generate", summary="Generate quiz questions for a concept")
def quiz_generate(req: QuizGenerateRequest):
    chunks = get_chunks_for_concept(req.concept, req.course)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No material found for concept '{req.concept}'.",
        )
    questions = generate_questions(chunks, n=req.n_questions, concept=req.concept)
    if not questions:
        raise HTTPException(status_code=500, detail="Could not generate questions.")
    return {"questions": questions}


@app.post("/quiz/evaluate", summary="Evaluate a student answer")
def quiz_evaluate(req: QuizEvaluateRequest):
    result = evaluate_answer(req.question, req.student_answer, req.source_text)
    return result


@app.post("/quiz/evaluate-batch", summary="Evaluate multiple student answers")
def quiz_evaluate_batch(req: QuizBatchEvaluateRequest):
    return {"results": evaluate_answers_batch([item.model_dump() for item in req.items])}


@app.post("/quiz/result", summary="Save a completed quiz result")
def quiz_result(req: QuizResultRequest):
    pct, is_best = save_quiz_result(
        req.concept, req.course or "", req.score, req.total
    )
    return {"percentage": pct, "is_personal_best": bool(is_best)}


@app.get("/quiz/weak", summary="Get weakest concepts by quiz history")
def quiz_weak(course: str | None = None, limit: int = 5):
    return {"weak_concepts": get_weak_concepts(course, limit)}


# ---------------------------------------------------------------------------
# Summary + Audio
# ---------------------------------------------------------------------------

class SummaryAudioRequest(BaseModel):
    topic: str | None = None
    course: str | None = None
    top_k: int = 5
    chunk_ids: list[int] = []
    speaker: str = "mari"


class TTSRequest(BaseModel):
    text: str
    speaker: str = "mari"


@app.post("/summary", summary="Estonian text summary from retrieved chunks")
def summary(req: SummaryAudioRequest):
    result = build_summary(
        topic=req.topic,
        course=req.course,
        top_k=req.top_k,
        chunk_ids=req.chunk_ids or None,
    )
    return result


@app.post("/tts", summary="Convert text to Estonian speech (WAV)")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        wav_bytes = synthesize(req.text, speaker=req.speaker)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/topics/textToSpeech", summary="Alias for frontend text-to-speech requests")
def topic_tts(req: TTSRequest):
    return tts(req)


# ---------------------------------------------------------------------------
# Streaming RAG
# ---------------------------------------------------------------------------

@app.post("/rag/stream", summary="Stream RAG answer token by token")
def rag_stream_endpoint(req: RAGRequest):
    def event_stream():
        for chunk in rag_stream(req.question, course=req.course, top_k=req.top_k):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Study Plan
# ---------------------------------------------------------------------------

@app.post("/plan", summary="Generate a study plan based on weak concepts")
def plan(req: PlanRequest):
    weak = get_weak_concepts(req.course, req.limit_weak)
    concept_names = [w["concept"] for w in weak]
    result = generate_plan(
        weak_concepts=concept_names,
        days_until_exam=req.days_until_exam,
        target_grade=req.target_grade,
        course=req.course,
    )
    return result


# ---------------------------------------------------------------------------
# Homework Helper
# ---------------------------------------------------------------------------

@app.post(
    "/homework-helper",
    summary="Guided hints without direct answers",
    description=(
        "Use this instead of `/rag` when the student is working on a homework problem "
        "and should find the answer themselves. "
        "The AI will point to relevant material, ask guiding questions, and give hints — "
        "**but will never reveal the direct answer**. "
        "Response: `{guidance: str, sources: list}`."
    ),
)
def homework_helper(req: HomeworkRequest):
    return homework_help(req.question, course=req.course, top_k=req.top_k)


# ---------------------------------------------------------------------------
# Spaced Repetition (SM-2)
# ---------------------------------------------------------------------------

@app.get("/quiz/next", summary="Get next concept due for review (SM-2)")
def quiz_next(course: str | None = None):
    concept = get_next_concept(course)
    if not concept:
        return {"concept": None, "message": "No concepts due for review."}
    return concept


@app.post("/quiz/sm2", summary="Update SM-2 state after a quiz attempt")
def quiz_sm2(req: SM2UpdateRequest):
    if not 0 <= req.quality <= 5:
        raise HTTPException(status_code=400, detail="quality must be 0-5")
    return update_sm2(req.concept, req.course, req.quality)
