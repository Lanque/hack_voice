# Easels — RAG Study Companion

A retrieval-augmented study tool. Upload course materials, then query them in natural language, get grounded answers with source citations, and quiz yourself on specific topics.

---

## How It Works

```
PDF → parse → chunk → embed (Azure ada-002) → SQLite (sqlite-vec + FTS5)
                                                        ↓
                              LLM extracts master concepts per document
                                                        ↓
                              each chunk tagged to concepts (chunk_concepts)
```

When you ask a question:

- Your question is embedded and matched against chunks via **vector search + keyword search (hybrid RRF)**
- The top chunks are passed to **GPT-4.1** with a strict "cite sources only" instruction
- The answer references which document and page it came from

---

## Setup

**1. Create and activate virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. Configure environment**

```powershell
copy .env.example .env
```

Open `.env` and fill in:

- `AZURE_OPENAI_API_KEY` — key for the embedding model (ada-002)
- `AZURE_API_KEY` — key for the chat model (gpt-4.1)

Both keys currently point to the same Azure deployment (`teliaee-openai`).

**4. Initialise the database**

```powershell
.venv\Scripts\python.exe -m db.init_db
```

This creates `easels.db` (SQLite file, tracked by `.gitignore`).

---

## Ingesting Course Materials

```powershell
.venv\Scripts\python.exe ingest.py "<file.pdf>" "<Lecture title>" "<Course name>"
```

**Example:**

```powershell
.venv\Scripts\python.exe ingest.py "konspekt.pdf" "Loeng 1" "Elektrisüsteem"
.venv\Scripts\python.exe ingest.py "dbb.pdf" "Loeng 1" "Andmebaasid"
```

What happens during ingest:

1. PDF is parsed into pages and split into ~500 character chunks
2. All chunks are embedded via Azure ada-002
3. The full document text is sent to GPT-4.1 once → extracts 5–15 master concepts
4. Each chunk is assigned concepts from the master list (no new concepts created per chunk)
5. Everything stored in `easels.db` — chunks, embeddings, FTS index, concepts, links

---

## Asking Questions (RAG)

```powershell
.venv\Scripts\python.exe rag.py "<question>" ["<course>"]
```

**Example:**

```powershell
.venv\Scripts\python.exe rag.py "Mis on nimipinge?" "Elektrisüsteem"
```

Returns a GPT-4.1 answer grounded strictly in the source material, with source citations (document + page).

---

## Web Summary Audio

You can expose the project as a small web API that:

- retrieves relevant chunks for a topic
- generates an Estonian summary from those chunks
- converts the summary to speech using `C:\txttospeech\text-to-speech`

### Extra environment variables

- `ESTONIAN_TTS_ROOT` - path to the TTS project, default `C:\txttospeech\text-to-speech`
- `ESTONIAN_TTS_PYTHON` - optional Python executable for the TTS environment
- `ESTONIAN_TTS_SPEAKER` - default voice, e.g. `albert`
- `MEDIA_DIR` - folder where generated wav files are stored

### Run the API

```powershell
.venv\Scripts\python.exe -m uvicorn api:app --reload
```

### Summary audio endpoint

`POST /api/summary-audio`

Example body:

```json
{
  "topic": "nimipinge",
  "course": "Elektrisusteem",
  "top_k": 5,
  "speaker": "albert"
}
```

You can also send explicit `chunk_ids` instead of a topic if the frontend already knows which chunks the user selected.

---

## Quizzing on a Topic

```powershell
.venv\Scripts\python.exe quiz.py                                  # list available concepts
.venv\Scripts\python.exe quiz.py "<concept>" ["<course>"] [n]     # run a quiz
```

**Example:**

```powershell
.venv\Scripts\python.exe quiz.py "nimipinged" "Elektrisüsteem" 5
```

Flow:

1. Fetches chunks tagged to the concept
2. GPT-4.1 generates N questions from those chunks
3. You answer each in the terminal
4. Each answer is evaluated and feedback given with source citation
5. Score summary + weak concepts highlighted at the end
6. Result (percentage) saved to DB — personal best tracked per concept

---

## Project Structure

```
easels/
├── ingest.py         # PDF → chunks → embeddings → DB + concept extraction
├── query.py          # Hybrid vector + FTS5 search with RRF merging
├── rag.py            # query.py + LLM generation (full RAG)
├── quiz.py           # /quiz-topic mode: generate → ask → evaluate → score
├── concepts.py       # LLM concept extraction and DB management
├── llm.py            # Azure OpenAI chat completion client
├── requirements.txt
├── .env.example      # Copy to .env and fill in keys
└── db/
    ├── __init__.py
    ├── connection.py  # SQLite connection with sqlite-vec loaded
    ├── init_db.py     # Run once to create tables
    └── schema.sql     # Table definitions
```

## Database Schema (simplified)

```
documents        — title, course, source_file
chunks           — text, page_number, document_id
chunk_embeddings — vector(1536) per chunk (sqlite-vec)
chunks_fts       — FTS5 full-text index
concepts         — name, course, embedding (master concept list per document)
chunk_concepts   — many-to-many: chunks ↔ concepts
document_concepts— many-to-many: documents ↔ concepts
quiz_results     — concept, score, percentage, best (personal best flag)
```

---

## Notes

- `easels.db` is in `.gitignore` — each teammate runs their own local DB and ingests PDFs themselves
- PDF files are also gitignored — share them separately
- The `.env` file is gitignored — never commit keys
