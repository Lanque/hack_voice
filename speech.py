"""
Speech adapters for Estonian summary audio generation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TTS_ROOT = Path(os.getenv("ESTONIAN_TTS_ROOT", r"C:\txttospeech\text-to-speech"))
TTS_PYTHON = os.getenv("ESTONIAN_TTS_PYTHON") or sys.executable
DEFAULT_SPEAKER = os.getenv("ESTONIAN_TTS_SPEAKER", "albert")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))


def _ensure_media_dir() -> Path:
    media_dir = MEDIA_DIR.resolve()
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir


def _tts_script() -> Path:
    script = TTS_ROOT / "synthesizer.py"
    if not script.exists():
        raise FileNotFoundError(f"TTS script not found: {script}")
    return script


def synthesize_summary_audio(text: str, speaker: str | None = None, stem: str = "summary") -> dict:
    """
    Generate a wav file from summary text using the external Estonian TTS project.
    Returns {path, filename}.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Summary text is empty.")

    media_dir = _ensure_media_dir()
    filename = f"{stem}_{uuid.uuid4().hex}.wav"
    output_path = media_dir / filename
    chosen_speaker = speaker or DEFAULT_SPEAKER

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(clean_text)
        input_path = Path(handle.name)

    try:
        subprocess.run(
            [
                TTS_PYTHON,
                str(_tts_script()),
                "--speaker",
                chosen_speaker,
                str(input_path),
                str(output_path),
            ],
            cwd=str(TTS_ROOT),
            check=True,
        )
    finally:
        input_path.unlink(missing_ok=True)

    return {"path": str(output_path), "filename": filename}
