# Easels RAG + Estonian Summary Audio API

This project is a study assistant backend that:

- ingests PDF study materials into SQLite
- retrieves relevant chunks with hybrid search
- answers questions with Azure OpenAI
- generates Estonian summary audio from retrieved chunks via external TTS

## Main Features

- `ingest.py`
  Loads PDFs, chunks text, creates embeddings, extracts concepts, and stores everything in SQLite.
- `rag.py`
  Retrieves relevant chunks and generates a grounded answer with citations.
- `quiz.py`
  Generates topic quizzes from stored chunks and evaluates answers.
- `api.py`
  Exposes a web API for summary generation and summary-to-speech.
- `summary.py`
  Builds an Estonian summary from retrieved chunks.
- `speech.py`
  Calls the external Estonian TTS project and saves a `.wav` file.

## Project Structure

```text
hack/
  api.py
  summary.py
  speech.py
  ingest.py
  query.py
  rag.py
  quiz.py
  concepts.py
  llm.py
  requirements.txt
  db/
    connection.py
    init_db.py
    schema.sql
```

## Local Setup

Create the virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Initialize the database:

```powershell
.venv\Scripts\python.exe -m db.init_db
```

## Environment Variables

Create a local `.env` file in the project root.

### Azure chat config

Used by `llm.py`.

```env
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your_chat_api_key
AZURE_DEPLOYMENT=gpt-4.1
AZURE_API_VERSION=2024-12-01-preview
```

### Azure embeddings config

Used by `query.py` and `ingest.py`.

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2023-05-15
```

### App config

```env
SQLITE_DB_PATH=easels.db

LLM_TEMPERATURE=0
LLM_TOP_P=1.0
LLM_PRESENCE_PENALTY=0.2
LLM_MAX_TOKENS=2000
```

### Estonian TTS config

Used by `speech.py`.

```env
ESTONIAN_TTS_ROOT=C:\txttospeech\text-to-speech
ESTONIAN_TTS_PYTHON=C:\Users\your-user\anaconda3\envs\transformer-tts\python.exe
ESTONIAN_TTS_SPEAKER=albert
MEDIA_DIR=media
```

## Important TTS Note

The summary audio endpoint does not include the TTS model itself.

The backend machine must separately have:

- the TartuNLP `text-to-speech` project
- its models
- a working Python or Conda environment that can run `synthesizer.py`

This project calls that TTS installation through `ESTONIAN_TTS_ROOT` and `ESTONIAN_TTS_PYTHON`.

## Ingesting PDFs

```powershell
.venv\Scripts\python.exe ingest.py "<file.pdf>" "<lecture title>" "<course name>"
```

Example:

```powershell
.venv\Scripts\python.exe ingest.py "konspekt.pdf" "Loeng 1" "Elektrisusteem"
```

## Asking Questions

```powershell
.venv\Scripts\python.exe rag.py "<question>" ["<course>"]
```

Example:

```powershell
.venv\Scripts\python.exe rag.py "Mis on nimipinge?" "Elektrisusteem"
```

## Running the API

Start the API locally:

```powershell
.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8010
```

Swagger UI:

```text
http://127.0.0.1:8010/docs
```

Health check:

```text
GET /api/health
```

## Summary Audio Endpoint

Endpoint:

```text
POST /api/summary-audio
```

Request body:

```json
{
  "topic": "nimipinge",
  "course": null,
  "top_k": 5,
  "chunk_ids": [],
  "speaker": "albert"
}
```

Notes:

- send `topic` to retrieve chunks through search
- or send `chunk_ids` if the frontend already knows which chunks to summarize
- `speaker` must match an available TTS voice such as `albert`, `kalev`, `kylli`, `mari`, `meelis`, or `vesta`

Example response:

```json
{
  "summary": "Kokkuvote ...",
  "sources": [
    {
      "id": 12,
      "text": "...",
      "page_number": 4,
      "document_title": "Loeng 1",
      "course": "Elektrisusteem"
    }
  ],
  "audio_url": "/media/summary_xxxxx.wav"
}
```

Generated audio files are served from:

```text
/media/<filename>.wav
```

## Notes

- `.env` is ignored and must never be committed
- `media/` is ignored because it contains generated audio
- `easels.db` is ignored because each environment may have different ingested materials
- if no matching chunks are found, the API returns a valid response with a fallback summary saying that no source material was found
