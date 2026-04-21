"""Estonian TTS via TartuNLP public API."""
from __future__ import annotations

import requests

TTS_API = "https://api.tartunlp.ai/text-to-speech/v2"
DEFAULT_SPEAKER = "mari"
REQUEST_TIMEOUT = 60  # seconds


def synthesize(text: str, speaker: str = DEFAULT_SPEAKER) -> bytes:
    """
    Call the TartuNLP public TTS API and return raw WAV bytes.
    Raises RuntimeError on failure.
    """
    response = requests.post(
        TTS_API,
        json={"text": text, "speaker": speaker},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"TTS API returned {response.status_code}: {response.text[:200]}"
        )
    return response.content
