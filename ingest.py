"""
PDF ingestion pipeline:
  1. Parse PDF into pages (text + image descriptions)
  2. Split pages into chunks
  3. Embed each chunk via Azure OpenAI
  4. Store document + chunks in Postgres
"""

import os
import re
import base64
import httpx
import pymupdf
from dotenv import load_dotenv
from db.connection import get_connection
from concepts import assign_concepts, extract_document_concepts

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

CHUNK_SIZE = 500      # characters
CHUNK_OVERLAP = 50   # characters

# Vision — uses the same gpt-4.1 deployment via llm.py config
AZURE_LLM_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_LLM_KEY = os.getenv("AZURE_API_KEY")
AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1")
AZURE_LLM_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
MIN_IMAGE_BYTES = 1024  # skip tiny images (icons, bullets)


# ---------------------------------------------------------------------------
# 1a. Vision: describe images on a page
# ---------------------------------------------------------------------------

def describe_image(image_bytes: bytes) -> str | None:
    """Send image to GPT-4.1 vision and return a text description."""
    b64 = base64.b64encode(image_bytes).decode()
    url = (
        f"{AZURE_LLM_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_LLM_DEPLOYMENT}/chat/completions?api-version={AZURE_LLM_API_VERSION}"
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image from a university lecture slide concisely. "
                            "Focus on any diagrams, charts, formulas, or tables. "
                            "Be specific about labels, values, and relationships shown. "
                            "If it is decorative or contains no meaningful content, respond with: SKIP"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0,
    }
    try:
        response = httpx.post(
            url,
            headers={"api-key": AZURE_LLM_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return None if content == "SKIP" else content
    except Exception as e:
        print(f"    [vision] error: {e}")
        return None


# ---------------------------------------------------------------------------
# 1. Parse PDF (text + image descriptions)
# ---------------------------------------------------------------------------

def parse_pdf(file_path: str) -> list[dict]:
    """Return list of {page_number, text} dicts.
    Text includes GPT-4.1 vision descriptions of any images on the page.
    """
    pages = []
    with pymupdf.open(file_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            # Extract and describe images on this page
            image_descriptions = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.n > 4:  # convert CMYK to RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    if len(img_bytes) < MIN_IMAGE_BYTES:
                        continue  # skip tiny images
                    desc = describe_image(img_bytes)
                    if desc:
                        image_descriptions.append(f"[Image on page {i}: {desc}]")
                except Exception as e:
                    print(f"    [vision] page {i} image error: {e}")

            combined = text
            if image_descriptions:
                combined = text + "\n" + "\n".join(image_descriptions)
                print(f"    page {i}: {len(image_descriptions)} image(s) described")

            if combined.strip():
                pages.append({"page_number": i, "text": combined})
    return pages


# ---------------------------------------------------------------------------
# 2. Chunk
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Return list of {page_number, text} dicts for all chunks."""
    result = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            result.append({"page_number": page["page_number"], "text": chunk})
    return result


# ---------------------------------------------------------------------------
# 3. Embed
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call Azure OpenAI embeddings API. Returns list of embedding vectors."""
    url = (
        f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_DEPLOYMENT}/embeddings?api-version={AZURE_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY,
    }
    # API supports up to 2048 inputs per request; batch to be safe
    batch_size = 100
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = httpx.post(
            url,
            headers=headers,
            json={"input": batch},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        # data is sorted by index
        embeddings.extend([item["embedding"] for item in sorted(data, key=lambda x: x["index"])])
    return embeddings


# ---------------------------------------------------------------------------
# 4. Store
# ---------------------------------------------------------------------------

def store_document(document_id: int, course: str, chunks: list[dict], embeddings: list[list[float]]):
    """Insert chunks into SQLite and assign concepts."""
    import struct

    conn = get_connection()
    try:
        cur = conn.cursor()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (document_id, text, page_number) VALUES (?, ?, ?)",
                (document_id, chunk["text"], chunk["page_number"]),
            )
            chunk_id = cur.lastrowid
            blob = struct.pack(f"{len(embedding)}f", *embedding)
            cur.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )
            cur.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk_id, chunk["text"]),
            )
            conn.commit()

            concepts = assign_concepts(chunk_id, chunk["text"], course, embed_texts)
            print(f"    chunk {i+1}/{len(chunks)}: {concepts}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest(file_path: str, title: str, course: str):
    print(f"Parsing {file_path}...")
    pages = parse_pdf(file_path)
    print(f"  {len(pages)} pages found")

    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks created")

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    def embed_one(text: str) -> list[float]:
        return embed_texts([text])[0]

    # Insert document row first so we have an ID for concept linking
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (title, course, source_file) VALUES (?, ?, ?)",
            (title, course, file_path),
        )
        document_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    print(f"  Extracting master concepts from full document...")
    full_text = "\n".join(p["text"] for p in pages)
    extract_document_concepts(full_text, course, embed_one, document_id)

    print(f"  Storing chunks in database...")
    store_document(document_id, course, chunks, embeddings)
    print(f"Done. Document '{title}' ingested.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python ingest.py <pdf_path> <title> <course>")
        sys.exit(1)

    ingest(
        file_path=sys.argv[1],
        title=sys.argv[2],
        course=sys.argv[3],
    )
