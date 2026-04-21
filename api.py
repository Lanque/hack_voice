"""
FastAPI application for web access to chunk summaries and summary audio.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from speech import synthesize_summary_audio
from summary import build_summary

load_dotenv()

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Easels API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


class SummaryAudioRequest(BaseModel):
    topic: str | None = Field(default=None, description="Search topic or natural-language summary target.")
    course: str | None = Field(default=None, description="Optional course filter.")
    top_k: int = Field(default=5, ge=1, le=10)
    chunk_ids: list[int] = Field(default_factory=list, description="Explicit chunk ids to summarize.")
    speaker: str | None = Field(default=None, description="Estonian TTS speaker name.")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/summary-audio")
def summary_audio(payload: SummaryAudioRequest) -> dict:
    if not payload.topic and not payload.chunk_ids:
        raise HTTPException(status_code=400, detail="Provide either 'topic' or 'chunk_ids'.")

    try:
        result = build_summary(
            topic=payload.topic,
            course=payload.course,
            top_k=payload.top_k,
            chunk_ids=payload.chunk_ids,
        )
        audio = synthesize_summary_audio(result["summary"], speaker=payload.speaker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summary audio generation failed: {exc}") from exc

    return {
        "summary": result["summary"],
        "sources": result["sources"],
        "audio_url": f"/media/{audio['filename']}",
    }
