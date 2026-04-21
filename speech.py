"""Estonian TTS via TartuNLP public API."""
from __future__ import annotations

import time

import requests

TTS_API = "https://api.tartunlp.ai/text-to-speech/v2"
DEFAULT_SPEAKER = "mari"
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 3


def _is_wav(payload: bytes) -> bool:
    return len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"


def synthesize(text: str, speaker: str = DEFAULT_SPEAKER) -> bytes:
    """
    Call the TartuNLP public TTS API and return raw WAV bytes.
    Raises RuntimeError on failure.
    """
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                TTS_API,
                json={"text": text, "speaker": speaker},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = f"TTS request failed: {exc}"
        else:
            if response.status_code != 200:
                last_error = (
                    f"TTS API returned {response.status_code}: {response.text[:200]}"
                )
            elif not response.content:
                last_error = "TTS API returned an empty audio payload."
            elif not _is_wav(response.content):
                content_type = response.headers.get("content-type", "unknown")
                last_error = (
                    "TTS API returned invalid audio data "
                    f"(content-type: {content_type})."
                )
            else:
                return response.content

        if attempt < MAX_RETRIES:
            time.sleep(0.5 * attempt)

    raise RuntimeError(last_error or "TTS synthesis failed.")
